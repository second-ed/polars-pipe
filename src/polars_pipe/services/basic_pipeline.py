from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars_pipe.adapters.io_pl as io
import polars_pipe.core.config as cf
import polars_pipe.core.transform as tf
import polars_pipe.core.validation as vl


def run_pipeline(
    io_wrapper: io.IOBase,
    config: dict,
    custom_transformation_fns: dict[str, Callable] | None = None,
) -> None:
    date_time = io_wrapper.get_datetime()
    config["guid"] = io_wrapper.get_guid()
    config["date_time"] = date_time

    parsed_config = cf.GeneralConfig.from_dict(config)

    lf = io_wrapper.read(
        parsed_config.src_path, file_type=io.FileType._member_map_[parsed_config.src_file_type]
    ).lazy()

    rules = vl.parse_validation_config(parsed_config.validation)
    expected_cols = vl.extract_expected_cols(parsed_config)

    valid_lf, invalid_lf = (
        lf.pipe(vl.check_expected_cols, expected_cols=expected_cols)
        .pipe(
            tf.add_process_cols,
            guid=parsed_config.guid,
            date_time=date_time,
            process_name=parsed_config.process_name,
        )
        .pipe(vl.validate_df, rules=rules)
    )
    tf_config = tf.TransformConfig.from_dict(parsed_config.transformations)

    transformed_lf = (
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

    custom_transformation_fns = custom_transformation_fns or {}
    transformed_lf = tf.pipe_custom_transformations(
        transformed_lf, custom_transformation_fns, parsed_config.custom_transformations
    )

    parsed_config.pipeline_plan = transformed_lf.explain()

    io_wrapper.write(
        parsed_config.to_dict(),
        str(
            Path(parsed_config.config_dst_dir).joinpath(
                f"{parsed_config.process_name}_{parsed_config.date_time}.yaml",
            )
        ),
        file_type=io.FileType.YAML,
    )

    dst_file_type = io.FileType._member_map_[parsed_config.dst_file_type]
    io_wrapper.write(transformed_lf, parsed_config.valid_dst_path, file_type=dst_file_type)

    if invalid_lf.limit(1).collect().height > 0:
        io_wrapper.write(invalid_lf, parsed_config.invalid_dst_path, file_type=dst_file_type)
