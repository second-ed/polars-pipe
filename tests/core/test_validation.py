from contextlib import nullcontext

import pytest

from polars_pipe.core.validation import check_expected_cols
from tests.conftest import BASIC_DF


@pytest.mark.parametrize(
    ("inp_lf", "expected_cols", "expected_result", "expected_context"),
    [
        pytest.param(BASIC_DF.lazy(), ["a"], BASIC_DF.lazy(), nullcontext()),
        pytest.param(BASIC_DF.lazy(), ["b"], BASIC_DF.lazy(), nullcontext()),
        pytest.param(BASIC_DF.lazy(), ["a", "b"], BASIC_DF.lazy(), nullcontext()),
        pytest.param(BASIC_DF.lazy(), [], BASIC_DF.lazy(), nullcontext()),
        pytest.param(BASIC_DF.lazy(), ["c"], None, pytest.raises(ValueError)),
    ],
)
def test_check_expected_cols(inp_lf, expected_cols, expected_result, expected_context):
    with expected_context:
        assert (
            check_expected_cols(inp_lf, expected_cols).collect().to_dicts()
            == expected_result.collect().to_dicts()
        )
