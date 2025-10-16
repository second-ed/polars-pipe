import operator
from collections.abc import Callable
from functools import reduce

import polars as pl


def _reduce_horizontal(op: Callable, cols: list[str]) -> pl.Expr:
    if not cols:
        raise ValueError("Must supply at least one column")

    exprs = [pl.col(c) for c in cols]
    return reduce(op, exprs)


def add_cols(cols: list[str]) -> pl.Expr:
    return _reduce_horizontal(operator.add, cols)


def sub_cols(cols: list[str]) -> pl.Expr:
    return _reduce_horizontal(operator.sub, cols)


def mul_cols(cols: list[str]) -> pl.Expr:
    return _reduce_horizontal(operator.mul, cols)


def div_cols(cols: list[str]) -> pl.Expr:
    return _reduce_horizontal(operator.truediv, cols)
