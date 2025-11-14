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
logger.info(f"{POLARS_DTYPE_MAPPING = }")


@attrs.define(frozen=True)
class TransformConfig:
    drop_cols: list = attrs.field(factory=list, validator=attrs.validators.instance_of(list))
    rename_map: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    recast_map: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    fill_map: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    clip_map: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    filter_exprs: list = attrs.field(factory=list, validator=attrs.validators.instance_of(list))
    new_col_map: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    dedupe_cols: list = attrs.field(factory=list, validator=attrs.validators.instance_of(list))
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
    """Generate deterministic hashes of the values in non-system-columns for each row in the lazyframe.
    If the column `sys_col_row_hash` is already present in the lazyframe, it returns the lazyframe unchanged.
    This stage cannot be skipped.
    """
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
    lf: pl.LazyFrame,
    date_time: datetime,
    guid: str,
    src_path: str,
    process_name: str = "process",
) -> pl.LazyFrame:
    """Add columns for the given source file path and process time.
    The columns are called `f"sys_col_{process_name}_src_path"` and `f"sys_col_{process_name}_datetime"`
    to allow for multiple teams to use the same pipeline and not overwrite eachothers sys_cols,
    maintaining the lineage of the data as its passed between teams.
    This stage cannot be skipped.
    """
    return lf.with_columns(
        [
            pl.lit(guid).alias(f"sys_col_{process_name}_guid"),
            pl.lit(src_path).alias(f"sys_col_{process_name}_src_path"),
            pl.lit(date_time).alias(f"sys_col_{process_name}_datetime"),
        ]
    )


def normalise_str_cols(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Normalise the string columns by stripping whitespace and converting to lowercase.
    This stage cannot be skipped.
    """
    return lf.with_columns(
        [
            pl.col(col_name).str.strip_chars().str.to_lowercase()
            for col_name, dtype in lf.collect_schema().items()
            if dtype == pl.Utf8 and not col_name.startswith("sys_col_")
        ]
    )


def standardise_col_names_if_no_case_insensitive_dupes(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Standardise the column names, lowered and stripped.
    If there are case insensitive duplicates then this operation is skipped and the given lf is returned.
    This is run at the end before saving the transformed data to avoid having to translate all column references in the config.
    """
    cols = lf.collect_schema().names()
    lowered = [c.lower().strip() for c in cols]

    if len(lowered) != len(set(lowered)):
        logger.info("There are case insensitive duplicates in the column names, skipping")
        return lf

    return rename_df_cols(
        lf, {col: std_col for col in cols if (std_col := col.lower().strip()) != col}
    )


def drop_df_cols(lf: pl.LazyFrame, drop_cols: list[str]) -> pl.LazyFrame:
    """Drop the given columns from the lazyframe.
    If no drop_cols are provided, exits early returning the given lazyframe.
    Expects a list of `["col_to_drop_a", "col_to_drop_b"]`
    """
    if not drop_cols:
        logger.info(f"No drop_cols provided: {drop_cols = }")
        return lf
    logger.info(f"Dropping: {drop_cols = }")
    return lf.drop(drop_cols)


def rename_df_cols(lf: pl.LazyFrame, rename_map: dict[str, str]) -> pl.LazyFrame:
    """Rename columns to new names.
    If no rename_map is provided, exits early returning the given lazyframe.
    Expects a dict of `{"old_col_name": "new_col_name"}`.
    """
    if not rename_map:
        logger.info(f"No rename_map provided: {rename_map = }")
        return lf
    logger.info(f"Renaming: {rename_map = }")
    return lf.rename(rename_map)


def recast_df_cols(lf: pl.LazyFrame, recast_map: dict[str, pl.DataType]) -> pl.LazyFrame:
    """Recast columns to new datatypes.
    If no recast_map is provided, exits early returning the given lazyframe.
    Expects a dict of `{"col_name": pl.DataType}`.
    """
    if not recast_map:
        logger.info(f"No recast_map provided: {recast_map = }")
        return lf
    logger.info(f"Recasting: {recast_map = }")
    return lf.with_columns([pl.col(col).cast(dtype) for col, dtype in recast_map.items()])


def fill_nulls_per_col(lf: pl.LazyFrame, fill_map: dict[str, Any]) -> pl.LazyFrame:
    """Fill nulls for given columns with value.
    If no fill_map is provided, exits early returning the given lazyframe.
    Expects a dict of `{"col_name": "fill value"}`.
    """
    if not fill_map:
        logger.info(f"No fill_map provided: {fill_map = }")
        return lf
    logger.info(f"Filling nulls: {fill_map = }")
    return lf.with_columns([pl.col(col).fill_null(value) for col, value in fill_map.items()])


def clip_df_cols(lf: pl.LazyFrame, clip_map: dict[str, tuple[float, float]]) -> pl.LazyFrame:
    """Clip values for given columns with min/max.
    If no clip_map is provided, exits early returning the given lazyframe.
    Expects a dict of `{"col_name": (min_val, max_val)}`.
    """
    if not clip_map:
        logger.info(f"No clip_map provided: {clip_map = }")
        return lf
    logger.info(f"Clipping values: {clip_map = }")
    return lf.with_columns(
        [pl.col(col).clip(lower, upper) for col, (lower, upper) in clip_map.items()]
    )


def nest_df_cols(lf: pl.LazyFrame, nest_cols: dict[str, list[str]]) -> pl.LazyFrame:
    """Nest given columns into a struct column.
    If no nest_cols is provided, exits early returning the given lazyframe.
    Expects a dict of `{"struct_col_name": ["col_a", "col_b", "col_c"]}`.
    """
    if not nest_cols:
        logger.info(f"No nest_cols provided: {nest_cols = }")
        return lf

    return lf.with_columns(
        [pl.struct(to_nest).alias(name) for name, to_nest in nest_cols.items()]
    ).drop(*[c for cols in nest_cols.values() for c in cols])


def unnest_df_cols(lf: pl.LazyFrame, unnest_cols: list[str]) -> pl.LazyFrame:
    """Unnest given columns from struct columns.
    If no unnest_cols is provided, exits early returning the given lazyframe.
    Expects a list of `["struct_col_a", "struct_col_b", "struct_col_c"]`.
    """
    if not unnest_cols:
        logger.info(f"No unnest_cols provided: {unnest_cols = }")
        return lf
    logger.info(f"Unnesting: {unnest_cols = }")
    return lf.unnest(unnest_cols)


def filter_df(lf: pl.LazyFrame, filter_exprs: list[pl.Expr]) -> pl.LazyFrame:
    """Filter the lazyframe where all the given expressions are true.
    If no filter_exprs is provided, exits early returning the given lazyframe.
    Expects a list of polars expressions: `list[pl.Expr]`.
    """
    if not filter_exprs:
        logger.info(f"No filter_exprs provided: {filter_exprs = }")
        return lf
    logger.info(f"Filtering df: {filter_exprs = }")
    combined_filter = pl.all_horizontal(filter_exprs)
    return lf.filter(combined_filter)


def deduplicate_rows(lf: pl.LazyFrame, subset_cols: list[str]) -> pl.LazyFrame:
    """Deduplicate rows in the dataframe based on a subset of columns, can provide `["*"]` for all columns.
    If no subset_cols is provided, exits early returning the given lazyframe.
    Expects `list[str]`.
    """
    if not subset_cols:
        logger.info(f"No subset_cols provided: {subset_cols = }")
        return lf
    logger.info(f"Deduplicating df: {subset_cols = }")
    return lf.unique(subset=subset_cols, maintain_order=True)


CUSTOM_DERIVE_FNS = {
    k: v for k, v in inspect.getmembers(derive_cols, inspect.isfunction) if not k.startswith("_")
}
logger.info(f"{CUSTOM_DERIVE_FNS = }")
ALL_DERIVE_FNS = {**derive_cols.PL_EXPR_FNS, **CUSTOM_DERIVE_FNS}


def derive_new_cols(lf: pl.LazyFrame, new_col_map: dict[str, dict[str, str]]) -> pl.LazyFrame:
    """Derive new columns from existing columns.
    If no new_col_map is provided, exits early returning the given lazyframe.
    Expects a dict:
    ```
    {
        "new_col_name": {
            "fn_name": "add_cols",
            "fn_kwargs": {"cols": ["col_a", "col_b"]},
        },
        "other_new_name": {
            "fn_name": "mul_cols",
            "fn_kwargs": {"cols": ["col_a", "col_b", "col_c"]},
        }
    }
    ```
    As well as the row-wise operator columns, you can use any existing method on a pl.Expr:
    ```
    {
        "new_col_from_expression": {
            "fn_name": "col_method_name",
            "fn_kwargs": {"col": "some_col_name"},
        },
        # e.g:
        "cumulative_wind": {
            "fn_name": "cum_sum",
            "fn_kwargs": {"col": "wind_mph"},
        }
    }
    ```
    """
    if not new_col_map:
        logger.info(f"No new_col_map provided: {new_col_map = }")
        return lf

    logger.info(f"Deriving new columns: {new_col_map = }")
    derived_transforms = {
        derived_col_name: partial(ALL_DERIVE_FNS[fn_config["fn_name"]], **fn_config["fn_kwargs"])
        for derived_col_name, fn_config in new_col_map.items()
    }

    # have to call `expr` here because its a partial function object
    return lf.with_columns([expr().alias(name) for name, expr in derived_transforms.items()])


def pipe_custom_transformations(
    lf: pl.LazyFrame,
    custom_transformation_fns: dict[str, Callable[[pl.LazyFrame, Any], pl.LazyFrame]],
    custom_transformation_map: dict[str, dict[str, Any]],
) -> pl.LazyFrame:
    """Apply custom transformations to the lazyframe.
    If no custom_transformation_map is provided, exits early returning the given lazyframe.
    Must be given a dict of functions that meet this protocol: `dict[str, Callable[[pl.LazyFrame, Any], pl.LazyFrame]]`.
    `custom_transformation_map` is expected as
    ```
    {
        "custom_transformation_name": {
            "kwarg": "kwarg_value",
        },
        "other_custom_transformation_name": {
            "other_kwarg": "other_kwarg_value",
            "another_kwarg": 0,
        },
    }
    ```
    If a custom function is listed in the `custom_transformation_map` that isn't in the `custom_transformation_fns`
    will raise a KeyError.
    """
    if not custom_transformation_map:
        logger.info(f"No custom_transformation_map provided: {custom_transformation_map = }")
        return lf

    for fn_name, kwargs in custom_transformation_map.items():
        logger.info(f"Applying custom transformation: {fn_name = } {kwargs = }")
        func = custom_transformation_fns[fn_name]
        lf = lf.pipe(func, **kwargs)
    return lf
