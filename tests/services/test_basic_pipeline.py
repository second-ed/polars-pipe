import polars as pl
import pytest

from polars_pipe.adapters.io_pl import FakeIOWrapper
from polars_pipe.services.basic_pipeline import run_pipeline


def basic_df():
    return pl.DataFrame(
        [
            {"name": "alice ", "salary": 30_000, "division": " B", "bonus": 10_000.0},
            {"name": "ben", "salary": 28_000, "division": "C", "bonus": 15000.0},
            {"name": "charlie", "salary": 75000, "division": "A", "bonus": None},
            {"name": None, "salary": 0, "division": "", "bonus": 0},
            {"name": "dani", "salary": 50_000, "division": "D", "bonus": 70_000.0},
            {"name": "emily", "salary": 80000, "division": "A", "bonus": -5000},
        ]
    )


def expected_transformed_df():
    return [
        {"name": "alice", "salary": 30000, "annual_bonus": 10000, "full_comp": 40000},
        {"name": "ben", "salary": 28000, "annual_bonus": 15000, "full_comp": 43000},
        {"name": "charlie", "salary": 75000, "annual_bonus": 0, "full_comp": 75000},
        {"name": "dani", "salary": 50000, "annual_bonus": 70000, "full_comp": 120000},
        {"name": "emily", "salary": 80000, "annual_bonus": 0, "full_comp": 80000},
    ]


def expected_error_records():
    return [
        {"name": None, "salary": 0, "division": "", "bonus": 0.0, "error_reason": "missing name"}
    ]


@pytest.mark.parametrize(
    ("raw_data", "config", "expected_result"),
    [
        pytest.param(
            {"path/to/raw_data.parquet": basic_df()},
            {
                "src_path": "path/to/raw_data.parquet",
                "src_file_type": "parquet",
                "valid_dst_path": "path/to/transformed_data.parquet",
                "invalid_dst_path": "path/to/error_records.parquet",
                "validation": {"missing name": ["name", "is_not_null", None]},
                "transformations": {
                    "filter_exprs": {"no d division": ["division", "ne", "D"]},
                    "fill_map": {"bonus": 0},
                    "recast_map": {"bonus": pl.Int64},
                    "rename_map": {"bonus": "annual_bonus"},
                    "clip_map": {"annual_bonus": (0, 500_000)},
                    "new_col_map": {
                        "full_comp": {
                            "fn_name": "add_cols",
                            "fn_kwargs": {"col1": "salary", "col2": "annual_bonus"},
                        },
                    },
                    "drop_cols": ["division"],
                },
            },
            {
                "path/to/transformed_data.parquet": expected_transformed_df(),
                "path/to/error_records.parquet": expected_error_records(),
            },
            id="transforms and filters dfs when given populated config",
        ),
        pytest.param(
            {"path/to/raw_data.parquet": basic_df()},
            {
                "src_path": "path/to/raw_data.parquet",
                "src_file_type": "parquet",
                "valid_dst_path": "path/to/transformed_data.parquet",
                "invalid_dst_path": "path/to/error_records.parquet",
                "validation": {},
                "transformations": {
                    "filter_exprs": {},
                    "fill_map": {},
                    "recast_map": {},
                    "rename_map": {},
                    "clip_map": {},
                    "new_col_map": {},
                    "drop_cols": [],
                },
            },
            {
                "path/to/transformed_data.parquet": [
                    {
                        "name": "alice",
                        "salary": 30000,
                        "division": "b",
                        "bonus": 10000.0,
                    },
                    {
                        "name": "ben",
                        "salary": 28000,
                        "division": "c",
                        "bonus": 15000.0,
                    },
                    {
                        "name": "charlie",
                        "salary": 75000,
                        "division": "a",
                        "bonus": None,
                    },
                    {
                        "name": None,
                        "salary": 0,
                        "division": "",
                        "bonus": 0.0,
                    },
                    {
                        "name": "dani",
                        "salary": 50000,
                        "division": "d",
                        "bonus": 70000.0,
                    },
                    {
                        "name": "emily",
                        "salary": 80000,
                        "division": "a",
                        "bonus": -5000.0,
                    },
                ],
            },
            id="should skip all stages if not given config",
        ),
    ],
)
def test_basic_pipeline(raw_data, config, expected_result):
    io = FakeIOWrapper(files=raw_data)
    run_pipeline(io, config)
    assert (
        io.files[config["valid_dst_path"]].to_dicts() == expected_result[config["valid_dst_path"]]
    )
    if config["invalid_dst_path"] in io.files:
        assert (
            io.files[config["invalid_dst_path"]].to_dicts()
            == expected_result[config["invalid_dst_path"]]
        )
