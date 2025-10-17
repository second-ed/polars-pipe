from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Self

import attrs

import polars_pipe.adapters.io_pl as io
import polars_pipe.core.transform as tf
import polars_pipe.core.validation as vl


def abs_path(path: str) -> str:
    return str(Path(path).absolute())


@attrs.define
class GeneralConfig:
    guid: str = attrs.field(validator=attrs.validators.instance_of(str))
    date_time: str = attrs.field(
        validator=attrs.validators.instance_of(str), converter=lambda x: x.strftime("%Y%m%d_%H%M")
    )
    process_name: str = attrs.field(validator=attrs.validators.instance_of(str))
    src_path: str = attrs.field(validator=attrs.validators.instance_of(str), converter=abs_path)
    src_file_type: str = attrs.field(
        validator=[
            attrs.validators.instance_of(str),
            attrs.validators.in_(io.FileType._member_map_),
        ],
        converter=str.upper,
    )
    valid_dst_path: str = attrs.field(
        validator=attrs.validators.instance_of(str), converter=abs_path
    )
    invalid_dst_path: str = attrs.field(
        validator=attrs.validators.instance_of(str), converter=abs_path
    )
    config_dst_dir: str = attrs.field(
        validator=attrs.validators.instance_of(str), converter=abs_path
    )
    validation: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    transformations: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    custom_transformations: dict = attrs.field(
        factory=dict, validator=attrs.validators.instance_of(dict)
    )

    def to_dict(self) -> dict:
        return attrs.asdict(self)


@attrs.define
class TransformConfig:
    drop_cols: list = attrs.field(factory=list, validator=attrs.validators.instance_of(list))
    rename_map: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    recast_map: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    fill_map: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    clip_map: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    filter_exprs: list = attrs.field(factory=list, validator=attrs.validators.instance_of(list))
    new_col_map: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    unnest_cols: list = attrs.field(factory=list, validator=attrs.validators.instance_of(list))
    nest_cols: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))

    @classmethod
    def from_dict(cls, config: dict) -> Self:
        config = deepcopy(config)
        config["filter_exprs"] = list(
            vl.parse_validation_config(config.get("filter_exprs", {})).values()
        )
        config["recast_map"] = {
            k: tf.POLARS_DTYPE_MAPPING[v] for k, v in config.get("recast_map", {}).items()
        }
        return cls(**config)


def run_pipeline(
    io_wrapper: io.IOBase,
    config: dict,
    custom_transformation_fns: dict[str, Callable] | None = None,
) -> None:
    custom_transformation_fns = custom_transformation_fns or {}

    guid = io_wrapper.get_guid()
    date_time = io_wrapper.get_datetime()
    config["guid"] = guid
    config["date_time"] = date_time

    parsed_config = GeneralConfig(**config)

    file_type = io.FileType._member_map_[parsed_config.src_file_type]
    lf = io_wrapper.read(parsed_config.src_path, file_type).lazy()

    rules = vl.parse_validation_config(parsed_config.validation)
    expected_cols = [val[0] for val in parsed_config.validation.values()]

    valid_lf, invalid_lf = (
        lf.pipe(vl.check_expected_cols, expected_cols=expected_cols)
        .pipe(
            tf.add_process_cols,
            guid=guid,
            date_time=date_time,
            process_name=parsed_config.process_name,
        )
        .pipe(vl.validate_df, rules=rules)
    )
    tf_config = TransformConfig.from_dict(parsed_config.transformations)

    pipeline_plan = (
        valid_lf.pipe(tf.normalise_str_cols)
        .pipe(tf.unnest_df_cols, tf_config.unnest_cols)
        .pipe(tf.filter_df, filter_exprs=tf_config.filter_exprs)
        .pipe(tf.fill_nulls_per_col, fill_map=tf_config.fill_map)
        .pipe(tf.recast_df_cols, recast_map=tf_config.recast_map)
        .pipe(tf.rename_df_cols, rename_map=tf_config.rename_map)
        .pipe(tf.clip_df_cols, clip_map=tf_config.clip_map)
        .pipe(tf.derive_new_cols, new_col_map=tf_config.new_col_map)
        .pipe(tf.nest_df_cols, nest_cols=tf_config.nest_cols)
        .pipe(tf.drop_df_cols, drop_cols=tf_config.drop_cols)
    )

    pipeline_plan = tf.pipe_custom_transformations(
        pipeline_plan, custom_transformation_fns, parsed_config.custom_transformations
    )

    transformed_df = pipeline_plan.collect()
    io_wrapper.write(
        parsed_config.to_dict(),
        str(
            Path(parsed_config.config_dst_dir).joinpath(
                f"{parsed_config.process_name}_{parsed_config.date_time}.yaml",
            )
        ),
        file_type=io.FileType.YAML,
    )
    io_wrapper.write(transformed_df, parsed_config.valid_dst_path, file_type=io.FileType.PARQUET)

    invalid_df = invalid_lf.collect()
    if not invalid_df.is_empty():
        io_wrapper.write(invalid_df, parsed_config.invalid_dst_path, file_type=io.FileType.PARQUET)
