from collections.abc import Callable

import polars as pl


def _apply_expr_to_each_col(
    lf: pl.LazyFrame, expr_func: Callable[[str], pl.Expr], expr_name: str
) -> pl.LazyFrame:
    schema_names = lf.collect_schema().names()
    return (
        lf.with_columns(pl.lit(expr_name).alias("statistic"))
        .with_columns([expr_func(name) for name in schema_names])
        .select("statistic", *schema_names)
        .first()
    )


def _calculate_null_proportion(name: str) -> pl.Expr:
    return (pl.col(name).null_count() / pl.len()).alias(name).cast(pl.String)


def _calculate_n_unique(name: str) -> pl.Expr:
    return pl.col(name).n_unique().cast(pl.String)


CUSTOM_STATISTICS = {"null_proportion": _calculate_null_proportion, "n_unique": _calculate_n_unique}


def describe_lf(
    lf: pl.LazyFrame, custom_statistics: dict[str, Callable[[str], pl.Expr]] = CUSTOM_STATISTICS
) -> pl.LazyFrame:
    described_df = lf.describe().cast(pl.String)

    for name, fn in custom_statistics.items():
        described_df = described_df.vstack(_apply_expr_to_each_col(lf, fn, name).collect())

    return described_df.lazy()
