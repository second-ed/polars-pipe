import polars as pl
import pytest

import polars_pipe.core.derive_cols as dc

BASIC_DF = pl.DataFrame(
    [
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
        {"a": 1, "b": 2},
    ]
)


@pytest.mark.parametrize(
    ("raw_data", "derive_fn", "cols", "expected_series_values"),
    [
        pytest.param(BASIC_DF, dc.add_cols, ["a", "a", "a"], [3, 3, 3]),
        pytest.param(BASIC_DF, dc.add_cols, ["a", "b"], [3, 3, 3]),
        pytest.param(BASIC_DF, dc.add_cols, ["b", "b"], [4, 4, 4]),
        pytest.param(BASIC_DF, dc.sub_cols, ["a", "a", "a"], [-1, -1, -1]),
        pytest.param(BASIC_DF, dc.sub_cols, ["a", "b"], [-1, -1, -1]),
        pytest.param(BASIC_DF, dc.sub_cols, ["b", "b"], [0, 0, 0]),
        pytest.param(BASIC_DF, dc.mul_cols, ["a", "a", "a"], [1, 1, 1]),
        pytest.param(BASIC_DF, dc.mul_cols, ["a", "b"], [2, 2, 2]),
        pytest.param(BASIC_DF, dc.mul_cols, ["b", "b"], [4, 4, 4]),
        pytest.param(BASIC_DF, dc.div_cols, ["a", "a", "a"], [1.0, 1.0, 1.0]),
        pytest.param(BASIC_DF, dc.div_cols, ["a", "b"], [0.5, 0.5, 0.5]),
        pytest.param(BASIC_DF, dc.div_cols, ["b", "b"], [1, 1, 1]),
    ],
)
def test_derive_cols(raw_data, derive_fn, cols, expected_series_values):
    expected_result = raw_data.hstack([pl.Series("result", expected_series_values)])
    assert (
        raw_data.with_columns(derive_fn(cols).alias("result")).to_dicts()
        == expected_result.to_dicts()
    )
