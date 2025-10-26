from __future__ import annotations

import inspect
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from functools import partial
from types import MappingProxyType
from typing import Any, Self

import attrs
import polars as pl
import polars.datatypes.classes as pl_dtypes
from polars.datatypes._parse import is_polars_dtype

import polars_pipe.core.validation as vl
from polars_pipe.core import derive_cols
from polars_pipe.core.logger import logger

POLARS_DTYPE_MAPPING = MappingProxyType(
    {k: v for k, v in inspect.getmembers(pl_dtypes) if is_polars_dtype(v)}
)


@attrs.define(frozen=True)
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


def add_hash_col(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf_schema = dict(lf.collect_schema().items())
    if "sys_col_row_hash" in lf_schema:
        return lf
    non_sys_cols = [c for c in lf_schema if not c.startswith("sys_col")]

    return lf.with_columns(
        pl.concat_str(
            [
                pl.col(c).struct.json_encode()
                if isinstance(lf_schema[c], (pl.Struct, pl.List))
                else pl.col(c).cast(pl.Utf8)
                for c in non_sys_cols
            ],
            separator="|",
        )
        .hash()
        .alias("sys_col_row_hash")
    )


def add_process_cols(
    lf: pl.LazyFrame, guid: str, date_time: datetime, process_name: str = "process"
) -> pl.LazyFrame:
    return lf.with_columns(
        [
            pl.lit(guid).alias(f"sys_col_{process_name}_guid"),
            pl.lit(date_time).alias(f"sys_col_{process_name}_datetime"),
        ]
    )


def normalise_str_cols(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        [
            pl.col(col_name).str.strip_chars().str.to_lowercase()
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
