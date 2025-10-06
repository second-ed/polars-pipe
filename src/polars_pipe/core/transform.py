from functools import partial
from typing import Any
import polars as pl

def drop_df_cols(lf: pl.LazyFrame, drop_cols: list[str]) -> pl.LazyFrame:
    if not drop_cols:
        return lf
    return lf.drop(drop_cols)

def rename_df_cols(lf: pl.LazyFrame, rename_map: dict[str,str]) -> pl.LazyFrame:
    if not rename_map:
        return lf
    return lf.rename(rename_map)

def recast_df_cols(lf: pl.LazyFrame, recast_map: dict[str, pl.DataType]) -> pl.LazyFrame:
    if not recast_map:
        return lf
    return lf.with_columns([
            pl.col(col).cast(dtype) for col, dtype in recast_map.items()
        ])


def fill_nulls_per_col(lf: pl.LazyFrame, fill_map: dict[str, Any]) -> pl.LazyFrame:
    if not fill_map:
        return lf
    return lf.with_columns([
        pl.col(col).fill_null(value) for col, value in fill_map.items()
    ])


def clip_df_cols(lf: pl.LazyFrame, clip_map: dict[str, tuple[float, float]]) -> pl.LazyFrame:
    if not clip_map:
        return lf
    return lf.with_columns([
        pl.col(col).clip(lower, upper) for col, (lower, upper) in clip_map.items()
    ])


def filter_df(lf: pl.LazyFrame, filter_exprs: list[pl.Expr]) -> pl.LazyFrame:
    if not filter_exprs:
        return lf
    combined_filter = pl.all_horizontal(filter_exprs)
    return lf.filter(combined_filter)