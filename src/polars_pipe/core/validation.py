from typing import Any

import polars as pl

from polars_pipe.core.logger import logger


def check_expected_cols(lf: pl.LazyFrame, expected_cols: list[str]) -> pl.LazyFrame:
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
