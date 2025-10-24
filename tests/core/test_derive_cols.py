from contextlib import nullcontext

import polars as pl
import pytest

import polars_pipe.core.derive_cols as dc
from tests.conftest import BASIC_DF


@pytest.mark.parametrize(
    ("raw_data", "derive_fn", "cols", "expected_series_values", "expected_context"),
    [
        pytest.param(BASIC_DF, dc.add_cols, ["a", "a", "a"], [3, 3, 3], nullcontext()),
        pytest.param(BASIC_DF, dc.add_cols, ["a", "b"], [3, 3, 3], nullcontext()),
        pytest.param(BASIC_DF, dc.add_cols, ["b", "b"], [4, 4, 4], nullcontext()),
        pytest.param(BASIC_DF, dc.sub_cols, ["a", "a", "a"], [-1, -1, -1], nullcontext()),
        pytest.param(BASIC_DF, dc.sub_cols, ["a", "b"], [-1, -1, -1], nullcontext()),
        pytest.param(BASIC_DF, dc.sub_cols, ["b", "b"], [0, 0, 0], nullcontext()),
        pytest.param(BASIC_DF, dc.mul_cols, ["a", "a", "a"], [1, 1, 1], nullcontext()),
        pytest.param(BASIC_DF, dc.mul_cols, ["a", "b"], [2, 2, 2], nullcontext()),
        pytest.param(BASIC_DF, dc.mul_cols, ["b", "b"], [4, 4, 4], nullcontext()),
        pytest.param(BASIC_DF, dc.div_cols, ["a", "a", "a"], [1.0, 1.0, 1.0], nullcontext()),
        pytest.param(BASIC_DF, dc.div_cols, ["a", "b"], [0.5, 0.5, 0.5], nullcontext()),
        pytest.param(BASIC_DF, dc.div_cols, ["b", "b"], [1, 1, 1], nullcontext()),
        pytest.param(BASIC_DF, dc.div_cols, [], None, pytest.raises(ValueError)),
    ],
)
def test_derive_cols(raw_data, derive_fn, cols, expected_series_values, expected_context):
    with expected_context:
        assert (
            raw_data.with_columns(derive_fn(cols).alias("result")).to_dicts()
            == raw_data.hstack([pl.Series("result", expected_series_values)]).to_dicts()
        )
