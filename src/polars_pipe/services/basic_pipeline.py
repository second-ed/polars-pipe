from copy import deepcopy
from typing import Self

import attrs

import polars_pipe.core.transform as tf
import polars_pipe.core.validation as vl
from polars_pipe.adapters import io_pl


@attrs.define
class TransformConfig:
    drop_cols: list = attrs.field(factory=list, validator=attrs.validators.instance_of(list))
    rename_map: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    recast_map: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    fill_map: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    clip_map: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    filter_exprs: list = attrs.field(factory=list, validator=attrs.validators.instance_of(list))
    new_col_map: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))

    @classmethod
    def from_dict(cls, config: dict) -> Self:
        config = deepcopy(config)
        config["filter_exprs"] = list(
            vl.parse_validation_config(config.get("filter_exprs", {})).values()
        )
        config["new_col_map"] = vl.parse_validation_config(config.get("new_col_map", {}))
        return cls(**config)


def run_pipeline(io: io_pl.IOProtocol, config: dict) -> None:
    file_type = io_pl.FileType._member_map_[config["src_file_type"].upper()]
    lf = io.read(config["src_path"], file_type).lazy()

    rules = vl.parse_validation_config(config.get("validation", {}))
    expected_cols = [val[0] for val in config.get("validation", {}).values()]

    valid_lf, invalid_lf = lf.pipe(vl.check_expected_cols, expected_cols=expected_cols).pipe(
        vl.validate_df, rules=rules
    )

    tf_config = TransformConfig.from_dict(config.get("transformations", {}))

    tranformed_df = (
        valid_lf.pipe(tf.normalise_str_cols)
        .pipe(tf.filter_df, filter_exprs=tf_config.filter_exprs)
        .pipe(tf.drop_df_cols, drop_cols=tf_config.drop_cols)
        .pipe(tf.recast_df_cols, recast_map=tf_config.recast_map)
        .pipe(tf.fill_nulls_per_col, fill_map=tf_config.fill_map)
        .pipe(tf.rename_df_cols, rename_map=tf_config.rename_map)
        .pipe(tf.derive_cols, new_col_map=tf_config.new_col_map)
        .collect()
    )
    io.write(tranformed_df, config["valid_dst_path"], file_type=io_pl.FileType.PARQUET)
    io.write(invalid_lf.collect(), config["invalid_dst_path"], file_type=io_pl.FileType.PARQUET)
