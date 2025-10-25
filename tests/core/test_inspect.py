import polars as pl
import pytest

from polars_pipe.core.inspect import get_null_proportions


@pytest.mark.parametrize(
    ("inp_lf", "expected_result"),
    [
        pytest.param(
            pl.DataFrame(
                [
                    {"a": 1, "b": None},
                    {"a": 1, "b": None},
                    {"a": 1, "b": 2},
                    {"a": None, "b": 2},
                ]
            ).lazy(),
            [{"a": 0.25, "b": 0.5}],
        )
    ],
)
def test_get_null_proportions(inp_lf, expected_result):
    assert get_null_proportions(inp_lf).collect().to_dicts() == expected_result
