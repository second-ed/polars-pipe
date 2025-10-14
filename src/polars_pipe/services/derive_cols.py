import polars as pl


def add_cols(col1: str, col2: str) -> pl.Expr:
    return pl.col(col1).add(pl.col(col2))
