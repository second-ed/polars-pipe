import polars as pl


def mul_cols(col1: str, col2: str) -> pl.Expr:
    return pl.col(col1).mul(pl.col(col2))


def lin_reg(col_x: str, col_slope: str, col_intercept: str) -> pl.Expr:
    return pl.col(col_x) + pl.col(col_slope) + pl.col(col_intercept)
