import inspect
from collections.abc import Callable
from datetime import datetime
from functools import partial
from types import MappingProxyType
from typing import Any

import polars as pl
import polars.datatypes.classes as pl_dtypes
from polars.datatypes._parse import is_polars_dtype

from polars_pipe.core import derive_cols
from polars_pipe.core.logger import logger

POLARS_DTYPE_MAPPING = MappingProxyType(
    {k: v for k, v in inspect.getmembers(pl_dtypes) if is_polars_dtype(v)}
)


def add_process_cols(
    lf: pl.LazyFrame, guid: str, date_time: datetime, process_name: str = "process"
) -> pl.LazyFrame:
    return lf.with_columns(
        [
            pl.lit(guid).alias(f"{process_name}_guid"),
            pl.lit(date_time).alias(f"{process_name}_datetime"),
        ]
    )


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


def nest_df_cols(lf: pl.LazyFrame, nest_cols: dict[str, list[str]]) -> pl.LazyFrame:
    if not nest_cols:
        logger.info(f"No nest_cols provided: {nest_cols = }")
        return lf

    return lf.with_columns(
        [pl.struct(to_nest).alias(name) for name, to_nest in nest_cols.items()]
    ).drop(*[c for cols in nest_cols.values() for c in cols])


def unnest_df_cols(lf: pl.LazyFrame, unnest_cols: list[str]) -> pl.LazyFrame:
    if not unnest_cols:
        logger.info(f"No unnest_cols provided: {unnest_cols = }")
        return lf
    logger.info(f"Unnesting: {unnest_cols = }")
    return lf.unnest(unnest_cols)


def filter_df(lf: pl.LazyFrame, filter_exprs: list[pl.Expr]) -> pl.LazyFrame:
    if not filter_exprs:
        logger.info(f"No filter_exprs provided: {filter_exprs = }")
        return lf
    logger.info(f"Filtering df: {filter_exprs = }")
    combined_filter = pl.all_horizontal(filter_exprs)
    return lf.filter(combined_filter)


DERIVE_FNS = {
    k: v for k, v in inspect.getmembers(derive_cols, inspect.isfunction) if not k.startswith("_")
}
logger.info(f"{DERIVE_FNS = }")


def derive_new_cols(lf: pl.LazyFrame, new_col_map: dict[str, dict[str, str]]) -> pl.LazyFrame:
    if not new_col_map:
        logger.info(f"No new_col_map provided: {new_col_map = }")
        return lf

    logger.info(f"Deriving new columns: {new_col_map = }")
    derived_transforms = {
        derived_col_name: partial(DERIVE_FNS[fn_config["fn_name"]], **fn_config["fn_kwargs"])
        for derived_col_name, fn_config in new_col_map.items()
    }

    # have to call `expr` here because its a partial function object
    return lf.with_columns([expr().alias(name) for name, expr in derived_transforms.items()])


def pipe_custom_transformations(
    lf: pl.LazyFrame,
    custom_transformation_fns: dict[str, Callable[[pl.LazyFrame, Any], pl.LazyFrame]],
    custom_transformation_map: dict[str, dict[str, dict[str, Any]]],
) -> pl.LazyFrame:
    if not custom_transformation_map:
        logger.info(f"No custom_transformation_map provided: {custom_transformation_map = }")
        return lf

    for fn_name, kwargs in custom_transformation_map.items():
        logger.info(f"Applying custom transformation: {fn_name = } {kwargs = }")
        func = custom_transformation_fns[fn_name]
        lf = lf.pipe(func, **kwargs)
    return lf
