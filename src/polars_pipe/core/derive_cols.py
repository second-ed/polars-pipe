import operator
from collections.abc import Callable
from functools import reduce

import polars as pl

from polars_pipe.core.logger import logger


def _reduce_horizontal(op: Callable, cols: list[str]) -> pl.Expr:
    if not cols:
        msg = f"Must supply at least one column. {op = } {cols = }"
        logger.error(msg)
        raise ValueError(msg)
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
