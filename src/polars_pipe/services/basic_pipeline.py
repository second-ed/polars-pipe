import inspect
from copy import deepcopy
from types import MappingProxyType
from typing import Self

import attrs
import polars.datatypes.classes as pl_dtypes
from polars.datatypes._parse import is_polars_dtype

import polars_pipe.adapters.io_pl as io
import polars_pipe.core.transform as tf
import polars_pipe.core.validation as vl

POLARS_DTYPE_MAPPING = MappingProxyType(
    {k: v for k, v in inspect.getmembers(pl_dtypes) if is_polars_dtype(v)}
)


@attrs.define
class GeneralConfig:
    src_path: str = attrs.field(validator=attrs.validators.instance_of(str))
    src_file_type: str = attrs.field(
        validator=[
            attrs.validators.instance_of(str),
            attrs.validators.in_(io.FileType._member_map_),
        ],
        converter=str.upper,
    )
    valid_dst_path: str = attrs.field(validator=attrs.validators.instance_of(str))
    invalid_dst_path: str = attrs.field(validator=attrs.validators.instance_of(str))
    validation: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    transformations: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))


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
            k: POLARS_DTYPE_MAPPING[v] for k, v in config.get("recast_map", {}).items()
        }
        return cls(**config)


def run_pipeline(io_wrapper: io.IOBase, config: dict) -> None:
    parsed_config = GeneralConfig(**config)

    file_type = io.FileType._member_map_[parsed_config.src_file_type]
    lf = io_wrapper.read(parsed_config.src_path, file_type).lazy()

    rules = vl.parse_validation_config(parsed_config.validation)
    expected_cols = [val[0] for val in parsed_config.validation.values()]

    valid_lf, invalid_lf = (
        lf.pipe(vl.check_expected_cols, expected_cols=expected_cols)
        .pipe(tf.add_process_cols, guid=io_wrapper.get_guid(), date_time=io_wrapper.get_datetime())
        .pipe(vl.validate_df, rules=rules)
    )
    tf_config = TransformConfig.from_dict(parsed_config.transformations)

    transformed_df = (
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
        .collect()
    )
    io_wrapper.write(transformed_df, parsed_config.valid_dst_path, file_type=io.FileType.PARQUET)

    invalid_df = invalid_lf.collect()
    if not invalid_df.is_empty():
        io_wrapper.write(invalid_df, parsed_config.invalid_dst_path, file_type=io.FileType.PARQUET)
