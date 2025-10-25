import polars as pl


def get_null_proportions(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Get the null count divided by the length of the dataframe.

    Args:
        lf (pl.LazyFrame): The lazyframe to analyse.

    Returns:
        pl.LazyFrame: The first row of the resulting lazyframe to avoid the same
        values repeated on every row.
    """
    return lf.with_columns(
        (pl.col(name).null_count() / pl.len()).alias(name) for name in lf.collect_schema().names()
    ).first()
