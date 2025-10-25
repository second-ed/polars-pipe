import polars as pl


def describe_lf(lf: pl.LazyFrame) -> pl.DataFrame:
    return lf.describe().vstack(get_null_proportions(lf).collect())


def get_null_proportions(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Get the null count divided by the length of the dataframe.

    Args:
        lf (pl.LazyFrame): The lazyframe to analyse.

    Returns:
        pl.LazyFrame: The first row of the resulting lazyframe to avoid the same
        values repeated on every row.
    """
    schema_names = lf.collect_schema().names()
    return (
        lf.with_columns(pl.lit("null_proportion").alias("statistic"))
        .with_columns([(pl.col(name).null_count()).alias(name) / pl.len() for name in schema_names])
        .select("statistic", *schema_names)
        .first()
    )
