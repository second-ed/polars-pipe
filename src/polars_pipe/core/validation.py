from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import polars as pl

from polars_pipe.core.config import GeneralConfig
from polars_pipe.core.logger import logger


def extract_expected_cols(parsed_config: GeneralConfig) -> set[str]:
    """Extract the columns that are referenced by name in the config.
    Returns a set of the expected_cols, accounts for columns that may be renamed.
    """

    def collect_original_names_dict_keys_or_list(
        stage_params: dict | list, reverse_rename_map: dict[str, str], new_cols: list[str]
    ) -> set[str]:
        return {reverse_rename_map.get(item, item) for item in stage_params if item not in new_cols}

    def collect_original_names_expr(
        stage_params: dict, reverse_rename_map: dict[str, str], new_cols: list[str]
    ) -> set[str]:
        return {
            reverse_rename_map.get(item[0], item[0])
            for item in stage_params.values()
            if item not in new_cols
        }

    def collect_original_names_dict_values_of_lists(
        stage_params: dict[str, list[str]], reverse_rename_map: dict[str, str], new_cols: list[str]
    ) -> set[str]:
        return {
            reverse_rename_map.get(item, item)
            for nest_cols in stage_params.values()
            for item in nest_cols
            if item not in new_cols
        }

    transformation_config_handlers = {
        "filter_exprs": collect_original_names_expr,
        "fill_map": collect_original_names_dict_keys_or_list,
        "recast_map": collect_original_names_dict_keys_or_list,
        "rename_map": collect_original_names_dict_keys_or_list,
        "clip_map": collect_original_names_dict_keys_or_list,
        "unnest_cols": collect_original_names_dict_keys_or_list,
        "nest_cols": collect_original_names_dict_values_of_lists,
        "drop_cols": collect_original_names_dict_keys_or_list,
    }

    reverse_rename_map = {
        new: old for old, new in parsed_config.transformations.get("rename_map", {}).items()
    }
    logger.info(f"{reverse_rename_map = }")

    new_cols = list(parsed_config.transformations.get("new_col_map", {}))

    expected_cols = set()

    for stage_name, stage_param_map in parsed_config.transformations.items():
        expected_cols.update(
            transformation_config_handlers.get(stage_name, lambda _, __, ___: set())(
                stage_param_map, reverse_rename_map, new_cols
            )
        )

    expected_cols.update(
        collect_original_names_expr(parsed_config.validation, reverse_rename_map, new_cols)
    )
    logger.info(f"{expected_cols = }")
    return expected_cols


def check_expected_cols(lf: pl.LazyFrame, expected_cols: Iterable[str]) -> pl.LazyFrame:
    """Check whether the expected columns are in the lazyframe schema.
    Raises a ValueError if any are missing.
    """
    logger.info(f"{expected_cols = }")
    actual_schema = lf.collect_schema().names()
    missing = [c for c in expected_cols if c not in actual_schema]
    if missing:
        msg = f"Missing required columns: {missing = } {actual_schema = }"
        logger.error(msg)
        raise ValueError(msg)
    return lf


def parse_validation_config(rules_config: dict[str, list[Any]]) -> dict[str, pl.Expr]:
    """Parse the validation config from a `dict[str, list[Any]]` to a `dict[str, pl.Expr]`.
    Expects a dict:
    ```
    {
        "human readable validation rule": ["col_name", "pl.col method name", Optional value],
        "greater than 0": ["some_col", "gt", 0],
        "is not null": ["some_other_col", "is_not_null", None],
    }
    ```
    For the `"is null"` rule, there is no need for a value in the last element of the list as
    pl.col().is_not_null() doesn't take an arg.
    Should be noted that the rules should describe what a valid record looks like.
    """
    logger.info(f"{rules_config = }")

    exprs = {}

    for rule_name, expression in rules_config.items():
        col_name, operation, value = expression
        exprs[rule_name] = (
            getattr(pl.col(col_name), operation)(value)
            if value is not None
            else getattr(pl.col(col_name), operation)()
        )

    logger.info(f"{exprs = }")
    return exprs


def validate_df(lf: pl.LazyFrame, rules: dict[str, pl.Expr]) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Validate the lazyframe.
    Applies the rules that were parsed by `parse_validation_config` to the lazyframe.
    Valid records are returned to the lazyframe that is then transformed.
    Invalid records are returned as an error records table with an added `error_reason` column.
    The `error_reason` column could include all of the validation errors that that record failed.
    The returned invalid lazyframe is not processed any further by the pipeline.
    """
    if not rules:
        logger.info(f"No rules provided: {rules = }")
        return lf, pl.LazyFrame()

    validated_lf = lf.with_columns(
        [
            pl.concat_str(
                [
                    pl.when(~expr).then(pl.lit(rule_name)).otherwise(pl.lit(""))
                    for rule_name, expr in rules.items()
                ],
                separator=",",
            )
            .str.strip_chars(",")
            .alias("error_reason")
        ]
    )

    valid_lf = validated_lf.filter(pl.col("error_reason") == "").drop("error_reason")
    invalid_lf = validated_lf.filter(pl.col("error_reason") != "")

    return valid_lf, invalid_lf
