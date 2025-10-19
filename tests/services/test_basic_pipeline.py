import datetime

import polars as pl
import pytest

from polars_pipe.adapters.io_pl import FakeIOWrapper
from polars_pipe.core.config import abs_path
from polars_pipe.services.basic_pipeline import run_pipeline

BASIC_INPUT_DF = pl.DataFrame(
    [
        {
            "name": "alice ",
            "salary": 30_000,
            "division": " B",
            "bonus": 10_000.0,
            "projects": {"project a": 0.5, "project b": 0.5},
        },
        {
            "name": "ben",
            "salary": 28_000,
            "division": "C",
            "bonus": 15000.0,
            "projects": {"project c": 1.0, "project d": 0.0},
        },
        {
            "name": "charlie",
            "salary": 75000,
            "division": "A",
            "bonus": None,
            "projects": {"project a": 0.65, "project b": 0.35},
        },
        {
            "name": None,
            "salary": 0,
            "division": "",
            "bonus": 0,
            "projects": {"project c": 0.45, "project d": 0.55},
        },
        {
            "name": "dani",
            "salary": 50_000,
            "division": "D",
            "bonus": 70_000.0,
            "projects": {"project a": 0.95, "project b": 0.05},
        },
        {
            "name": "emily",
            "salary": 80000,
            "division": "A",
            "bonus": -5000,
            "projects": {"project c": 0.5, "project d": 0.5},
        },
    ]
)


EXPECTED_TRANSFORMED_DF = pl.DataFrame(
    [
        {
            "name": "alice",
            "comp": {"salary": 30000, "annual_bonus": 10000, "full_comp": 40000},
            "project a": 0.5,
            "project b": 0.5,
            "project c": None,
            "project d": None,
            "ingest_datetime": datetime.datetime(2025, 10, 16, 12, 0, tzinfo=datetime.UTC),
            "ingest_guid": "abc-123",
            "custom_transformation_status": "applied",
        },
        {
            "name": "ben",
            "comp": {"salary": 28000, "annual_bonus": 15000, "full_comp": 43000},
            "project a": None,
            "project b": None,
            "project c": 1.0,
            "project d": 0.0,
            "ingest_datetime": datetime.datetime(2025, 10, 16, 12, 0, tzinfo=datetime.UTC),
            "ingest_guid": "abc-123",
            "custom_transformation_status": "applied",
        },
        {
            "name": "charlie",
            "comp": {"salary": 75000, "annual_bonus": 0, "full_comp": 75000},
            "project a": 0.65,
            "project b": 0.35,
            "project c": None,
            "project d": None,
            "ingest_datetime": datetime.datetime(2025, 10, 16, 12, 0, tzinfo=datetime.UTC),
            "ingest_guid": "abc-123",
            "custom_transformation_status": "applied",
        },
        {
            "name": "dani",
            "comp": {"salary": 50000, "annual_bonus": 70000, "full_comp": 120000},
            "project a": 0.95,
            "project b": 0.05,
            "project c": None,
            "project d": None,
            "ingest_datetime": datetime.datetime(2025, 10, 16, 12, 0, tzinfo=datetime.UTC),
            "ingest_guid": "abc-123",
            "custom_transformation_status": "applied",
        },
        {
            "name": "emily",
            "comp": {"salary": 80000, "annual_bonus": 0, "full_comp": 80000},
            "project a": None,
            "project b": None,
            "project c": 0.5,
            "project d": 0.5,
            "ingest_datetime": datetime.datetime(2025, 10, 16, 12, 0, tzinfo=datetime.UTC),
            "ingest_guid": "abc-123",
            "custom_transformation_status": "applied",
        },
    ]
)


EXPECTED_ERROR_RECORDS = pl.DataFrame(
    [
        {
            "name": None,
            "salary": 0,
            "division": "",
            "bonus": 0.0,
            "projects": {
                "project a": None,
                "project b": None,
                "project c": 0.45,
                "project d": 0.55,
            },
            "ingest_datetime": datetime.datetime(2025, 10, 16, 12, 0, tzinfo=datetime.UTC),
            "ingest_guid": "abc-123",
            "error_reason": "missing name",
        }
    ]
)


def mock_custom_transformation(lf: pl.LazyFrame, status: str) -> pl.LazyFrame:
    return lf.with_columns(pl.lit(status).alias("custom_transformation_status"))


@pytest.mark.parametrize(
    ("raw_data", "config", "custom_transformation_fns", "expected_result"),
    [
        pytest.param(
            {abs_path("./path/to/raw_data.parquet"): BASIC_INPUT_DF},
            {
                "process_name": "ingest",
                "src_path": "./path/to/raw_data.parquet",
                "src_file_type": "parquet",
                "valid_dst_path": "./path/to/transformed_data",
                "invalid_dst_path": "./path/to/error_records",
                "config_dst_dir": "./path/to",
                "validation": {"missing name": ["name", "is_not_null", None]},
                "transformations": {
                    "filter_exprs": {"no d division": ["division", "ne", "D"]},
                    "fill_map": {"bonus": 0},
                    "recast_map": {"bonus": "Int64"},
                    "rename_map": {"bonus": "annual_bonus"},
                    "clip_map": {"annual_bonus": (0, 500_000)},
                    "new_col_map": {
                        "full_comp": {
                            "fn_name": "add_cols",
                            "fn_kwargs": {"cols": ["salary", "annual_bonus"]},
                        },
                    },
                    "unnest_cols": ["projects"],
                    "nest_cols": {"comp": ["salary", "annual_bonus", "full_comp"]},
                    "drop_cols": ["division"],
                },
                "custom_transformations": {
                    "mock_custom_transformation": {"status": "applied"},
                },
            },
            {"mock_custom_transformation": mock_custom_transformation},
            {
                abs_path(
                    "path/to/transformed_data/part-00000-abc-123.parquet"
                ): EXPECTED_TRANSFORMED_DF,
                abs_path(
                    "path/to/error_records/part-00000-abc-123.parquet"
                ): EXPECTED_ERROR_RECORDS,
                abs_path("./path/to/ingest_20251016_1200.yaml"): {
                    "guid": "abc-123",
                    "date_time": "20251016_1200",
                    "process_name": "ingest",
                    "src_path": abs_path("./path/to/raw_data.parquet"),
                    "src_file_type": "PARQUET",
                    "valid_dst_path": abs_path("./path/to/transformed_data"),
                    "invalid_dst_path": abs_path("./path/to/error_records"),
                    "config_dst_dir": abs_path("./path/to"),
                    "validation": {"missing name": ["name", "is_not_null", None]},
                    "transformations": {
                        "filter_exprs": {"no d division": ["division", "ne", "D"]},
                        "fill_map": {"bonus": 0},
                        "recast_map": {"bonus": "Int64"},
                        "rename_map": {"bonus": "annual_bonus"},
                        "clip_map": {"annual_bonus": (0, 500_000)},
                        "new_col_map": {
                            "full_comp": {
                                "fn_name": "add_cols",
                                "fn_kwargs": {"cols": ["salary", "annual_bonus"]},
                            },
                        },
                        "unnest_cols": ["projects"],
                        "nest_cols": {"comp": ["salary", "annual_bonus", "full_comp"]},
                        "drop_cols": ["division"],
                    },
                    "custom_transformations": {
                        "mock_custom_transformation": {"status": "applied"},
                    },
                    "pipeline_plan": " WITH_COLUMNS:\n"
                    ' ["applied".alias("custom_transformation_status")] \n'
                    '  simple π 8/8 ["name", "project a", ... 6 other columns]\n'
                    "     WITH_COLUMNS:\n"
                    '     [col("salary").as_struct([col("annual_bonus"), '
                    'col("full_comp")]).alias("comp")] \n'
                    "       WITH_COLUMNS:\n"
                    '       [[(col("salary")) + (col("annual_bonus"))].alias("full_comp")] \n'
                    "         WITH_COLUMNS:\n"
                    '         [col("annual_bonus").clip([dyn int: 0, dyn int: 500000])] \n'
                    '          SELECT [col("name"), col("salary"), '
                    'col("bonus").alias("annual_bonus"), col("project a"), col("project b"), '
                    'col("project c"), col("project d"), col("ingest_guid"), '
                    'col("ingest_datetime")]\n'
                    "             WITH_COLUMNS:\n"
                    '             [col("bonus").strict_cast(Int64)] \n'
                    "               WITH_COLUMNS:\n"
                    '               [col("project a"), col("project b"), col("project c"), '
                    'col("project d"), col("bonus").fill_null([0.0])] \n'
                    "                UNNEST by:[projects]\n"
                    '                  FILTER [(col("division")) != ("D")]\n'
                    "                  FROM\n"
                    "                     WITH_COLUMNS:\n"
                    "                     "
                    '[col("name").str.strip_chars([null]).str.lowercase(), '
                    'col("division").str.strip_chars([null]).str.lowercase(), '
                    'col("ingest_guid").str.strip_chars([null]).str.lowercase()] \n'
                    '                      simple π 7/7 ["name", "salary", "division", ... 4 '
                    "other columns]\n"
                    '                        FILTER [(col("error_reason")) == ("")]\n'
                    "                        FROM\n"
                    "                           WITH_COLUMNS:\n"
                    '                           ["abc-123".alias("ingest_guid"), 2025-10-16 '
                    '12:00:00.dt.replace_time_zone(["earliest"]).alias("ingest_datetime"), '
                    'when(col("name").is_null()).then("missing '
                    'name").otherwise("").str.concat_horizontal().str.strip_chars([","]).alias("error_reason")] \n'
                    '                            DF ["name", "salary", "division", "bonus", '
                    '...]; PROJECT["name", "salary", "division", "bonus", ...] 5/5 COLUMNS',
                },
            },
            id="transforms and filters dfs when given populated config",
        ),
        pytest.param(
            {abs_path("./path/to/raw_data.parquet"): BASIC_INPUT_DF},
            {
                "process_name": "ingest",
                "src_path": "./path/to/raw_data.parquet",
                "src_file_type": "parquet",
                "valid_dst_path": "./path/to/transformed_data",
                "invalid_dst_path": "./path/to/error_records",
                "config_dst_dir": "./path/to",
                "validation": {},
                "transformations": {},
            },
            None,
            {
                abs_path("./path/to/transformed_data/part-00000-abc-123.parquet"): pl.DataFrame(
                    [
                        {
                            "name": "alice",
                            "salary": 30000,
                            "division": "b",
                            "bonus": 10000.0,
                            "projects": {
                                "project a": 0.5,
                                "project b": 0.5,
                                "project c": None,
                                "project d": None,
                            },
                            "ingest_datetime": datetime.datetime(
                                2025, 10, 16, 12, 0, tzinfo=datetime.UTC
                            ),
                            "ingest_guid": "abc-123",
                        },
                        {
                            "name": "ben",
                            "salary": 28000,
                            "division": "c",
                            "bonus": 15000.0,
                            "projects": {
                                "project a": None,
                                "project b": None,
                                "project c": 1.0,
                                "project d": 0.0,
                            },
                            "ingest_datetime": datetime.datetime(
                                2025, 10, 16, 12, 0, tzinfo=datetime.UTC
                            ),
                            "ingest_guid": "abc-123",
                        },
                        {
                            "name": "charlie",
                            "salary": 75000,
                            "division": "a",
                            "bonus": None,
                            "projects": {
                                "project a": 0.65,
                                "project b": 0.35,
                                "project c": None,
                                "project d": None,
                            },
                            "ingest_datetime": datetime.datetime(
                                2025, 10, 16, 12, 0, tzinfo=datetime.UTC
                            ),
                            "ingest_guid": "abc-123",
                        },
                        {
                            "name": None,
                            "salary": 0,
                            "division": "",
                            "bonus": 0.0,
                            "projects": {
                                "project a": None,
                                "project b": None,
                                "project c": 0.45,
                                "project d": 0.55,
                            },
                            "ingest_datetime": datetime.datetime(
                                2025, 10, 16, 12, 0, tzinfo=datetime.UTC
                            ),
                            "ingest_guid": "abc-123",
                        },
                        {
                            "name": "dani",
                            "salary": 50000,
                            "division": "d",
                            "bonus": 70000.0,
                            "projects": {
                                "project a": 0.95,
                                "project b": 0.05,
                                "project c": None,
                                "project d": None,
                            },
                            "ingest_datetime": datetime.datetime(
                                2025, 10, 16, 12, 0, tzinfo=datetime.UTC
                            ),
                            "ingest_guid": "abc-123",
                        },
                        {
                            "name": "emily",
                            "salary": 80000,
                            "division": "a",
                            "bonus": -5000.0,
                            "projects": {
                                "project a": None,
                                "project b": None,
                                "project c": 0.5,
                                "project d": 0.5,
                            },
                            "ingest_datetime": datetime.datetime(
                                2025, 10, 16, 12, 0, tzinfo=datetime.UTC
                            ),
                            "ingest_guid": "abc-123",
                        },
                    ]
                ),
                abs_path("./path/to/ingest_20251016_1200.yaml"): {
                    "guid": "abc-123",
                    "date_time": "20251016_1200",
                    "process_name": "ingest",
                    "src_path": abs_path("./path/to/raw_data.parquet"),
                    "src_file_type": "PARQUET",
                    "valid_dst_path": abs_path("./path/to/transformed_data"),
                    "invalid_dst_path": abs_path("./path/to/error_records"),
                    "config_dst_dir": abs_path("./path/to"),
                    "validation": {},
                    "transformations": {},
                    "custom_transformations": {},
                    "pipeline_plan": " WITH_COLUMNS:\n"
                    ' [col("ingest_guid").str.strip_chars([null]).str.lowercase()] \n'
                    "   WITH_COLUMNS:\n"
                    '   ["abc-123".alias("ingest_guid"), 2025-10-16 '
                    '12:00:00.dt.replace_time_zone(["earliest"]).alias("ingest_datetime"), '
                    'col("name").str.strip_chars([null]).str.lowercase(), '
                    'col("division").str.strip_chars([null]).str.lowercase()] \n'
                    '    DF ["name", "salary", "division", "bonus", ...]; PROJECT */5 COLUMNS',
                },
            },
            id="should skip all stages if not given config",
        ),
    ],
)
def test_basic_pipeline(raw_data, config, custom_transformation_fns, expected_result):
    io = FakeIOWrapper(files=raw_data)
    run_pipeline(io, config, custom_transformation_fns)

    expected_result = {**raw_data, **expected_result}
    assert sorted(io.files) == sorted(expected_result)

    for key, actual_df in io.files.items():
        if isinstance(actual_df, pl.DataFrame):
            assert actual_df.to_dicts() == expected_result[key].to_dicts()
        else:
            assert actual_df == expected_result[key]
