from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars.selectors as cs

import polars_pipe.adapters.io_pl as io
import polars_pipe.core.config as cf
import polars_pipe.core.inspect as ins
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
        parsed_config.src_path, file_type=io.FileType.from_str(parsed_config.src_file_type)
    ).lazy()

    valid_lf, invalid_lf = (
        lf.pipe(vl.check_expected_cols, expected_cols=vl.extract_expected_cols(parsed_config))
        .pipe(tf.add_hash_col)
        .pipe(
            tf.add_process_cols,
            guid=parsed_config.guid,
            date_time=date_time,
            process_name=parsed_config.process_name,
        )
        .pipe(vl.validate_df, rules=vl.parse_validation_config(parsed_config.validation))
    )

    io_wrapper.write(
        ins.describe_lf(valid_lf),
        Path(parsed_config.dst_root).joinpath(
            parsed_config.guid, parsed_config.desc_stats_stem, "pre_transform"
        ),
        file_type=io.FileType.PARQUET,
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
        .pipe(
            tf.pipe_custom_transformations,
            custom_transformation_fns=custom_transformation_fns or {},
            custom_transformation_map=parsed_config.custom_transformations,
        )
        .select(
            # this basically selects the given columns (all if given "*") and then grabs
            # all of the sys_cols and puts them at the end (retaining their original relative order).
            cs.by_name(*parsed_config.select_cols) - cs.starts_with("sys_col"),
            cs.starts_with("sys_col"),
        )
    )

    parsed_config.pipeline_plan = transformed_lf.explain().splitlines()

    io_wrapper.write(
        parsed_config.to_dict(),
        Path(parsed_config.dst_root).joinpath(
            parsed_config.guid,
            parsed_config.config_dst_stem,
            f"{parsed_config.process_name}_{parsed_config.date_time}.yaml",
        ),
        file_type=io.FileType.YAML,
    )

    dst_file_type = io.FileType.from_str(parsed_config.dst_file_type)
    io_wrapper.write(
        transformed_lf,
        Path(parsed_config.dst_root).joinpath(parsed_config.guid, parsed_config.valid_dst_stem),
        file_type=dst_file_type,
    )

    if invalid_lf.limit(1).collect().height > 0:
        io_wrapper.write(
            invalid_lf,
            Path(parsed_config.dst_root).joinpath(
                parsed_config.guid, parsed_config.invalid_dst_stem
            ),
            file_type=dst_file_type,
        )

    io_wrapper.write(
        ins.describe_lf(transformed_lf),
        Path(parsed_config.dst_root).joinpath(
            parsed_config.guid, parsed_config.desc_stats_stem, "post_transform"
        ),
        file_type=io.FileType.PARQUET,
    )
