from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import polars as pl

from polars_pipe.core.config import GeneralConfig
from polars_pipe.core.logger import logger


def extract_expected_cols(parsed_config: GeneralConfig) -> set[str]:
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

    expected_cols = set()

    reverse_rename_map = {
        new: old for old, new in parsed_config.transformations.get("rename_map", {}).items()
    }

    new_cols = list(parsed_config.transformations.get("new_col_map", {}))

    for stage_name, stage_param_map in parsed_config.transformations.items():
        expected_cols.update(
            transformation_config_handlers.get(stage_name, lambda _, __, ___: set())(
                stage_param_map, reverse_rename_map, new_cols
            )
        )

    expected_cols.update(
        collect_original_names_expr(parsed_config.validation, reverse_rename_map, new_cols)
    )
    return expected_cols


def check_expected_cols(lf: pl.LazyFrame, expected_cols: Iterable[str]) -> pl.LazyFrame:
    logger.info(f"{expected_cols = }")
    missing = [c for c in expected_cols if c not in lf.collect_schema().names()]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return lf


def parse_validation_config(rules_config: dict[str, list[Any]]) -> dict[str, pl.Expr]:
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
