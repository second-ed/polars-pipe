import polars as pl
import pytest

from polars_pipe.core.inspect import describe_lf

BASIC_INSPECT_DF = pl.DataFrame(
    [
        {"a": 0, "b": None},
        {"a": 1, "b": None},
        {"a": 2, "b": 3},
        {"a": None, "b": 6},
    ]
).lazy()


@pytest.mark.parametrize(
    ("inp_lf", "expected_result"),
    [
        pytest.param(
            BASIC_INSPECT_DF,
            [
                {
                    "statistic": "count",
                    "a": 3.0,
                    "b": 2.0,
                },
                {
                    "statistic": "null_count",
                    "a": 1.0,
                    "b": 2.0,
                },
                {
                    "statistic": "mean",
                    "a": 1.0,
                    "b": 4.5,
                },
                {
                    "statistic": "std",
                    "a": 1.0,
                    "b": 2.1213203435596424,
                },
                {
                    "statistic": "min",
                    "a": 0.0,
                    "b": 3.0,
                },
                {
                    "statistic": "25%",
                    "a": 1.0,
                    "b": 3.0,
                },
                {
                    "statistic": "50%",
                    "a": 1.0,
                    "b": 6.0,
                },
                {
                    "statistic": "75%",
                    "a": 2.0,
                    "b": 6.0,
                },
                {
                    "statistic": "max",
                    "a": 2.0,
                    "b": 6.0,
                },
                {
                    "statistic": "null_proportion",
                    "a": 0.25,
                    "b": 0.5,
                },
            ],
        )
    ],
)
def test_describe_lf(inp_lf, expected_result):
    assert describe_lf(inp_lf).to_dicts() == expected_result
