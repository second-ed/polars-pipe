from collections.abc import Callable
from functools import partial

import polars as pl


def mul_cols(col1: str, col2: str) -> pl.Expr:
    return pl.col(col1).mul(pl.col(col2))


DERIVE_FNS = {"mul_cols": mul_cols}


def create_transformations_for_derived_cols(
    new_col_map: dict[str, dict[str, str]],
) -> dict[str, Callable[[], pl.Expr]]:
    derived_transforms = {}

    for derived_col_name, fn_config in new_col_map.items():
        fn_name = fn_config["fn_name"]
        fn_params = fn_config["fn_kwargs"]
        derived_transforms[derived_col_name] = partial(DERIVE_FNS[fn_name], **fn_params)
    return derived_transforms
