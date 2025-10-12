from typing import Any

import polars as pl

from polars_pipe.core.logger import logger


def normalise_str_cols(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        [
            pl.col(col_name).str.strip_chars().str.to_lowercase().alias(col_name)
            for col_name, dtype in lf.collect_schema().items()
            if dtype == pl.Utf8
        ]
    )


def drop_df_cols(lf: pl.LazyFrame, drop_cols: list[str]) -> pl.LazyFrame:
    if not drop_cols:
        logger.info(f"No drop_cols provided: {drop_cols = }")
        return lf
    logger.info(f"Dropping: {drop_cols = }")
    return lf.drop(drop_cols)


def rename_df_cols(lf: pl.LazyFrame, rename_map: dict[str, str]) -> pl.LazyFrame:
    if not rename_map:
        logger.info(f"No rename_map provided: {rename_map = }")
        return lf
    logger.info(f"Renaming: {rename_map = }")
    return lf.rename(rename_map)


def recast_df_cols(lf: pl.LazyFrame, recast_map: dict[str, pl.DataType]) -> pl.LazyFrame:
    if not recast_map:
        logger.info(f"No recast_map provided: {recast_map = }")
        return lf
    logger.info(f"Recasting: {recast_map = }")
    return lf.with_columns([pl.col(col).cast(dtype) for col, dtype in recast_map.items()])


def fill_nulls_per_col(lf: pl.LazyFrame, fill_map: dict[str, Any]) -> pl.LazyFrame:
    if not fill_map:
        logger.info(f"No fill_map provided: {fill_map = }")
        return lf
    logger.info(f"Filling nulls: {fill_map = }")
    return lf.with_columns([pl.col(col).fill_null(value) for col, value in fill_map.items()])


def clip_df_cols(lf: pl.LazyFrame, clip_map: dict[str, tuple[float, float]]) -> pl.LazyFrame:
    if not clip_map:
        logger.info(f"No clip_map provided: {clip_map = }")
        return lf
    logger.info(f"Clipping values: {clip_map = }")
    return lf.with_columns(
        [pl.col(col).clip(lower, upper) for col, (lower, upper) in clip_map.items()]
    )


def filter_df(lf: pl.LazyFrame, filter_exprs: list[pl.Expr]) -> pl.LazyFrame:
    if not filter_exprs:
        logger.info(f"No filter_exprs provided: {filter_exprs = }")
        return lf
    logger.info(f"Filtering df: {filter_exprs = }")
    combined_filter = pl.all_horizontal(filter_exprs)
    return lf.filter(combined_filter)


def derive_cols(lf: pl.LazyFrame, new_col_map: dict[str, pl.Expr]) -> pl.LazyFrame:
    if not new_col_map:
        logger.info(f"No new_col_map provided: {new_col_map = }")
        return lf
    logger.info(f"Deriving new columns: {new_col_map = }")
    # have to call `expr` here because its a partial function object
    return lf.with_columns([expr().alias(name) for name, expr in new_col_map.items()])
