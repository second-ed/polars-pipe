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
    ("raw_data", "derive_fn", "cols", "expected_result"),
    [
        pytest.param(
            BASIC_DF,
            dc.add_cols,
            ["a", "a", "a"],
            BASIC_DF.hstack([pl.Series("result", [3, 3, 3])]),
        ),
        pytest.param(
            BASIC_DF, dc.add_cols, ["a", "b"], BASIC_DF.hstack([pl.Series("result", [3, 3, 3])])
        ),
        pytest.param(
            BASIC_DF, dc.add_cols, ["b", "b"], BASIC_DF.hstack([pl.Series("result", [4, 4, 4])])
        ),
        pytest.param(
            BASIC_DF,
            dc.sub_cols,
            ["a", "a", "a"],
            BASIC_DF.hstack([pl.Series("result", [-1, -1, -1])]),
        ),
        pytest.param(
            BASIC_DF, dc.sub_cols, ["a", "b"], BASIC_DF.hstack([pl.Series("result", [-1, -1, -1])])
        ),
        pytest.param(
            BASIC_DF, dc.sub_cols, ["b", "b"], BASIC_DF.hstack([pl.Series("result", [0, 0, 0])])
        ),
        pytest.param(
            BASIC_DF,
            dc.mul_cols,
            ["a", "a", "a"],
            BASIC_DF.hstack([pl.Series("result", [1, 1, 1])]),
        ),
        pytest.param(
            BASIC_DF, dc.mul_cols, ["a", "b"], BASIC_DF.hstack([pl.Series("result", [2, 2, 2])])
        ),
        pytest.param(
            BASIC_DF, dc.mul_cols, ["b", "b"], BASIC_DF.hstack([pl.Series("result", [4, 4, 4])])
        ),
        pytest.param(
            BASIC_DF,
            dc.div_cols,
            ["a", "a", "a"],
            BASIC_DF.hstack([pl.Series("result", [1.0, 1.0, 1.0])]),
        ),
        pytest.param(
            BASIC_DF,
            dc.div_cols,
            ["a", "b"],
            BASIC_DF.hstack([pl.Series("result", [0.5, 0.5, 0.5])]),
        ),
        pytest.param(
            BASIC_DF, dc.div_cols, ["b", "b"], BASIC_DF.hstack([pl.Series("result", [1, 1, 1])])
        ),
    ],
)
def test_derive_cols(raw_data, derive_fn, cols, expected_result):
    assert (
        raw_data.with_columns(derive_fn(cols).alias("result")).to_dicts()
        == expected_result.to_dicts()
    )
