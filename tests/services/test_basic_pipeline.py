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
            "sys_col_ingest_datetime": datetime.datetime(2025, 10, 16, 12, 0, tzinfo=datetime.UTC),
            "sys_col_ingest_guid": "abc-123",
            "sys_col_row_hash": 6792230521060155726,
            "custom_transformation_status": "applied",
        },
        {
            "name": "ben",
            "comp": {"salary": 28000, "annual_bonus": 15000, "full_comp": 43000},
            "project a": None,
            "project b": None,
            "project c": 1.0,
            "project d": 0.0,
            "sys_col_ingest_datetime": datetime.datetime(2025, 10, 16, 12, 0, tzinfo=datetime.UTC),
            "sys_col_ingest_guid": "abc-123",
            "sys_col_row_hash": 16546676625329609454,
            "custom_transformation_status": "applied",
        },
        {
            "name": "charlie",
            "comp": {"salary": 75000, "annual_bonus": 0, "full_comp": 75000},
            "project a": 0.65,
            "project b": 0.35,
            "project c": None,
            "project d": None,
            "sys_col_ingest_datetime": datetime.datetime(2025, 10, 16, 12, 0, tzinfo=datetime.UTC),
            "sys_col_ingest_guid": "abc-123",
            "sys_col_row_hash": 16397991471585692086,
            "custom_transformation_status": "applied",
        },
        {
            "name": "dani",
            "comp": {"salary": 50000, "annual_bonus": 70000, "full_comp": 120000},
            "project a": 0.95,
            "project b": 0.05,
            "project c": None,
            "project d": None,
            "sys_col_ingest_datetime": datetime.datetime(2025, 10, 16, 12, 0, tzinfo=datetime.UTC),
            "sys_col_ingest_guid": "abc-123",
            "sys_col_row_hash": 3979787036476502685,
            "custom_transformation_status": "applied",
        },
        {
            "name": "emily",
            "comp": {"salary": 80000, "annual_bonus": 0, "full_comp": 80000},
            "project a": None,
            "project b": None,
            "project c": 0.5,
            "project d": 0.5,
            "sys_col_ingest_datetime": datetime.datetime(2025, 10, 16, 12, 0, tzinfo=datetime.UTC),
            "sys_col_ingest_guid": "abc-123",
            "sys_col_row_hash": 16442676451241693571,
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
            "sys_col_ingest_datetime": datetime.datetime(2025, 10, 16, 12, 0, tzinfo=datetime.UTC),
            "sys_col_ingest_guid": "abc-123",
            "sys_col_row_hash": 16397991471585692086,
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
                "dst_file_type": "parquet",
                "dst_root": "./dst/root",
                "validation": {"missing name": ["name", "is_not_null", None]},
                "transformations": {
                    "filter_exprs": {"no d division": ["division", "ne", "D"]},
                    "fill_map": {"bonus": 0},
                    "recast_map": {"bonus": "Int64"},
                    "clip_map": {"bonus": (0, 500_000)},
                    "new_col_map": {
                        "full_comp": {
                            "fn_name": "add_cols",
                            "fn_kwargs": {"cols": ["salary", "bonus"]},
                        },
                    },
                    "unnest_cols": ["projects"],
                    "rename_map": {"bonus": "annual_bonus"},
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
                    "./dst/root/abc-123/transformed_data/part-00000-abc-123.parquet"
                ): EXPECTED_TRANSFORMED_DF,
                abs_path(
                    "./dst/root/abc-123/error_records/part-00000-abc-123.parquet"
                ): EXPECTED_ERROR_RECORDS,
                abs_path("./dst/root/abc-123/config/ingest_20251016_1200.yaml"): {
                    "guid": "abc-123",
                    "date_time": "20251016_1200",
                    "process_name": "ingest",
                    "src_path": abs_path("./path/to/raw_data.parquet"),
                    "src_file_type": "PARQUET",
                    "dst_root": abs_path("./dst/root"),
                    "dst_file_type": "PARQUET",
                    "valid_dst_stem": "transformed_data",
                    "invalid_dst_stem": "error_records",
                    "config_dst_stem": "config",
                    "desc_stats_stem": "desc_stats",
                    "validation": {"missing name": ["name", "is_not_null", None]},
                    "transformations": {
                        "filter_exprs": {"no d division": ["division", "ne", "D"]},
                        "fill_map": {"bonus": 0},
                        "recast_map": {"bonus": "Int64"},
                        "clip_map": {"bonus": (0, 500_000)},
                        "new_col_map": {
                            "full_comp": {
                                "fn_name": "add_cols",
                                "fn_kwargs": {"cols": ["salary", "bonus"]},
                            },
                        },
                        "unnest_cols": ["projects"],
                        "rename_map": {"bonus": "annual_bonus"},
                        "nest_cols": {"comp": ["salary", "annual_bonus", "full_comp"]},
                        "drop_cols": ["division"],
                    },
                    "custom_transformations": {
                        "mock_custom_transformation": {"status": "applied"},
                    },
                    "select_cols": "*",
                    "pipeline_plan": [
                        'simple π 10/10 ["name", "project a", ... 8 other columns]',
                        "   WITH_COLUMNS:",
                        '   ["applied".alias("custom_transformation_status")] ',
                        '    simple π 9/9 ["name", "project a", ... 7 other columns]',
                        "       WITH_COLUMNS:",
                        '       [col("salary").as_struct([col("annual_bonus"), col("full_comp")]).alias("comp")] ',
                        '        SELECT [col("name"), col("salary"), col("bonus").alias("annual_bonus"), col("project a"), col("project b"), col("project c"), col("project d"), col("sys_col_row_hash"), col("sys_col_ingest_guid"), col("sys_col_ingest_datetime"), col("full_comp")]',
                        "           WITH_COLUMNS:",
                        '           [[(col("salary")) + (col("bonus"))].alias("full_comp")] ',
                        "             WITH_COLUMNS:",
                        '             [col("bonus").clip([dyn int: 0, dyn int: 500000])] ',
                        "               WITH_COLUMNS:",
                        '               [col("bonus").strict_cast(Int64)] ',
                        "                 WITH_COLUMNS:",
                        '                 [col("project a"), col("project b"), col("project c"), col("project d"), col("bonus").fill_null([0.0])] ',
                        "                  UNNEST by:[projects]",
                        '                    FILTER [(col("division")) != ("D")]',
                        "                    FROM",
                        "                       WITH_COLUMNS:",
                        '                       [col("name").str.strip_chars([null]).str.lowercase(), col("division").str.strip_chars([null]).str.lowercase(), col("sys_col_ingest_guid").str.strip_chars([null]).str.lowercase()] ',
                        '                        simple π 8/8 ["name", "salary", "division", ... 5 other columns]',
                        '                          FILTER [(col("error_reason")) == ("")]',
                        "                          FROM",
                        "                             WITH_COLUMNS:",
                        '                             [col("name").str.concat_horizontal([col("salary").strict_cast(String), col("division"), col("bonus").strict_cast(String), col("projects").struct.to_json()]).hash().alias("sys_col_row_hash"), "abc-123".alias("sys_col_ingest_guid"), 2025-10-16 12:00:00.dt.replace_time_zone(["earliest"]).alias("sys_col_ingest_datetime"), when(col("name").is_null()).then("missing name").otherwise("").str.concat_horizontal().str.strip_chars([","]).alias("error_reason")] ',
                        '                              DF ["name", "salary", "division", "bonus", ...]; PROJECT["name", "salary", "division", "bonus", ...] 5/5 COLUMNS',
                    ],
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
                "dst_root": "./dst/root",
                "dst_file_type": "parquet",
                "validation": {},
                "transformations": {},
            },
            None,
            {
                abs_path(
                    "./dst/root/abc-123/transformed_data/part-00000-abc-123.parquet"
                ): pl.DataFrame(
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
                            "sys_col_ingest_datetime": datetime.datetime(
                                2025, 10, 16, 12, 0, tzinfo=datetime.UTC
                            ),
                            "sys_col_ingest_guid": "abc-123",
                            "sys_col_row_hash": 6792230521060155726,
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
                            "sys_col_ingest_datetime": datetime.datetime(
                                2025, 10, 16, 12, 0, tzinfo=datetime.UTC
                            ),
                            "sys_col_ingest_guid": "abc-123",
                            "sys_col_row_hash": 16546676625329609454,
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
                            "sys_col_ingest_datetime": datetime.datetime(
                                2025, 10, 16, 12, 0, tzinfo=datetime.UTC
                            ),
                            "sys_col_ingest_guid": "abc-123",
                            "sys_col_row_hash": 16397991471585692086,
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
                            "sys_col_ingest_datetime": datetime.datetime(
                                2025, 10, 16, 12, 0, tzinfo=datetime.UTC
                            ),
                            "sys_col_ingest_guid": "abc-123",
                            "sys_col_row_hash": 16397991471585692086,
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
                            "sys_col_ingest_datetime": datetime.datetime(
                                2025, 10, 16, 12, 0, tzinfo=datetime.UTC
                            ),
                            "sys_col_ingest_guid": "abc-123",
                            "sys_col_row_hash": 3979787036476502685,
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
                            "sys_col_ingest_datetime": datetime.datetime(
                                2025, 10, 16, 12, 0, tzinfo=datetime.UTC
                            ),
                            "sys_col_ingest_guid": "abc-123",
                            "sys_col_row_hash": 16442676451241693571,
                        },
                    ]
                ),
                abs_path("./dst/root/abc-123/config/ingest_20251016_1200.yaml"): {
                    "guid": "abc-123",
                    "date_time": "20251016_1200",
                    "process_name": "ingest",
                    "src_path": abs_path("./path/to/raw_data.parquet"),
                    "src_file_type": "PARQUET",
                    "dst_root": abs_path("./dst/root"),
                    "dst_file_type": "PARQUET",
                    "valid_dst_stem": "transformed_data",
                    "invalid_dst_stem": "error_records",
                    "config_dst_stem": "config",
                    "desc_stats_stem": "desc_stats",
                    "validation": {},
                    "transformations": {},
                    "custom_transformations": {},
                    "select_cols": "*",
                    "pipeline_plan": [
                        " WITH_COLUMNS:",
                        ' [col("sys_col_ingest_guid").str.strip_chars([null]).str.lowercase()] ',
                        "   WITH_COLUMNS:",
                        '   [col("name").str.concat_horizontal([col("salary").strict_cast(String), col("division"), col("bonus").strict_cast(String), col("projects").struct.to_json()]).hash().alias("sys_col_row_hash"), "abc-123".alias("sys_col_ingest_guid"), 2025-10-16 12:00:00.dt.replace_time_zone(["earliest"]).alias("sys_col_ingest_datetime"), col("name").str.strip_chars([null]).str.lowercase(), col("division").str.strip_chars([null]).str.lowercase()] ',
                        '    DF ["name", "salary", "division", "bonus", ...]; PROJECT["name", "salary", "division", "bonus", ...] 5/5 COLUMNS',
                    ],
                },
            },
            id="should skip all stages if not given config",
        ),
        pytest.param(
            {
                abs_path("./path/to/raw_data.parquet"): BASIC_INPUT_DF.with_columns(
                    pl.lit(0).alias("sys_col_row_hash")
                )
            },
            {
                "process_name": "ingest",
                "src_path": "./path/to/raw_data.parquet",
                "src_file_type": "parquet",
                "dst_root": "./dst/root",
                "dst_file_type": "parquet",
                "validation": {},
                "transformations": {},
                "select_cols": ["name", "salary"],
            },
            None,
            {
                abs_path(
                    "./dst/root/abc-123/transformed_data/part-00000-abc-123.parquet"
                ): pl.DataFrame(
                    [
                        {
                            "name": "alice",
                            "salary": 30000,
                            "sys_col_ingest_datetime": datetime.datetime(
                                2025, 10, 16, 12, 0, tzinfo=datetime.UTC
                            ),
                            "sys_col_ingest_guid": "abc-123",
                            "sys_col_row_hash": 0,
                        },
                        {
                            "name": "ben",
                            "salary": 28000,
                            "sys_col_ingest_datetime": datetime.datetime(
                                2025, 10, 16, 12, 0, tzinfo=datetime.UTC
                            ),
                            "sys_col_ingest_guid": "abc-123",
                            "sys_col_row_hash": 0,
                        },
                        {
                            "name": "charlie",
                            "salary": 75000,
                            "sys_col_ingest_datetime": datetime.datetime(
                                2025, 10, 16, 12, 0, tzinfo=datetime.UTC
                            ),
                            "sys_col_ingest_guid": "abc-123",
                            "sys_col_row_hash": 0,
                        },
                        {
                            "name": None,
                            "salary": 0,
                            "sys_col_ingest_datetime": datetime.datetime(
                                2025, 10, 16, 12, 0, tzinfo=datetime.UTC
                            ),
                            "sys_col_ingest_guid": "abc-123",
                            "sys_col_row_hash": 0,
                        },
                        {
                            "name": "dani",
                            "salary": 50000,
                            "sys_col_ingest_datetime": datetime.datetime(
                                2025, 10, 16, 12, 0, tzinfo=datetime.UTC
                            ),
                            "sys_col_ingest_guid": "abc-123",
                            "sys_col_row_hash": 0,
                        },
                        {
                            "name": "emily",
                            "salary": 80000,
                            "sys_col_ingest_datetime": datetime.datetime(
                                2025, 10, 16, 12, 0, tzinfo=datetime.UTC
                            ),
                            "sys_col_ingest_guid": "abc-123",
                            "sys_col_row_hash": 0,
                        },
                    ]
                ),
                abs_path("./dst/root/abc-123/config/ingest_20251016_1200.yaml"): {
                    "guid": "abc-123",
                    "date_time": "20251016_1200",
                    "process_name": "ingest",
                    "src_path": abs_path("./path/to/raw_data.parquet"),
                    "src_file_type": "PARQUET",
                    "dst_root": abs_path("./dst/root"),
                    "dst_file_type": "PARQUET",
                    "valid_dst_stem": "transformed_data",
                    "invalid_dst_stem": "error_records",
                    "config_dst_stem": "config",
                    "desc_stats_stem": "desc_stats",
                    "validation": {},
                    "transformations": {},
                    "custom_transformations": {},
                    "select_cols": ["name", "salary"],
                    "pipeline_plan": [
                        " WITH_COLUMNS:",
                        ' [col("sys_col_ingest_guid").str.strip_chars([null]).str.lowercase()] ',
                        "   WITH_COLUMNS:",
                        '   ["abc-123".alias("sys_col_ingest_guid"), 2025-10-16 12:00:00.dt.replace_time_zone(["earliest"]).alias("sys_col_ingest_datetime"), col("name").str.strip_chars([null]).str.lowercase()] ',
                        '    DF ["name", "salary", "division", "bonus", ...]; PROJECT["name", "salary", "sys_col_row_hash"] 3/6 COLUMNS',
                    ],
                },
            },
            id="should filter all other columns except system cols",
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
