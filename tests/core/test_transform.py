import polars as pl
import pytest

import polars_pipe.core.transform as tf
from tests.conftest import BASIC_DF

DUPE_COLS_DF = pl.DataFrame(
    [
        {"a": 1, "A": 2},
        {"a": 1, "A": 2},
        {"a": 1, "A": 2},
    ]
)


@pytest.mark.parametrize(
    ("in_lf", "out_lf"),
    [
        pytest.param(BASIC_DF.lazy(), BASIC_DF.lazy()),
        pytest.param(DUPE_COLS_DF.lazy(), DUPE_COLS_DF.lazy()),
    ],
)
def test_standardise_col_names_if_no_case_insensitive_dupes(in_lf, out_lf):
    assert (
        tf.standardise_col_names_if_no_case_insensitive_dupes(in_lf).collect().to_dicts()
        == out_lf.collect().to_dicts()
    )
