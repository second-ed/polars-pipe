# polars-pipe
This is a basic pipeline for DE processing. The pipeline stages section describes the stages that the pipeline completes as part of its run.

# pipeline stages


## `check_expected_cols`
Check whether the expected columns are in the lf schema.
Raises a ValueError if any are missing.


## `add_hash_col`
Generate deterministic hashes of the values for each row in the lf.
If the column `sys_col_row_hash` is already present in the lf, it returns the lf unchanged.
This stage cannot be skipped.


## `add_process_cols`
Add columns for the given guid and process time.
The columns are called f"sys_col_{process_name}_guid" and f"sys_col_{process_name}_datetime"
to allow for multiple teams to use the same pipeline and not overwrite eachothers sys_cols,
maintaining the lineage of the data as its passed between teams.
This stage cannot be skipped.


## `validate_df`
Validate the lf.
Applies the rules that were parsed by `parse_validation_config` to the lf.
Valid records are returned to the lf that is then transformed.
Invalid records are returned as an error records table with an added `error_reason` column.
The `error_reason` column could include all of the validation errors that that record failed.
The returned invalid lf is not processed any further by the pipeline.


## `normalise_str_cols`
Normalise the string columns by stripping whitespace and converting to lowercase.
This stage cannot be skipped.


## `unnest_df_cols`
Unnest given columns from struct columns.
If no unnest_cols is provided, exits early returning the given lf.
Expects a list of `["struct_col_a", "struct_col_b", "struct_col_c"]`.


## `filter_df`
Filter the lf where all the given expressions are true.
If no filter_exprs is provided, exits early returning the given lf.
Expects a list of polars expressions: `list[pl.Expr]`.


## `fill_nulls_per_col`
Fill nulls for given columns with value.
If no fill_map is provided, exits early returning the given lf.
Expects a dict of `{"col_name": "fill value"}`.


## `recast_df_cols`
Recast columns to new datatypes.
If no recast_map is provided, exits early returning the given lf.
Expects a dict of `{"col_name": pl.DataType}`.


## `rename_df_cols`
Rename columns to new names.
If no rename_map is provided, exits early returning the given lf.
Expects a dict of `{"old_col_name": "new_col_name"}`.


## `clip_df_cols`
Clip values for given columns with min/max.
If no clip_map is provided, exits early returning the given lf.
Expects a dict of `{"col_name": (min_val, max_val)}`.


## `derive_new_cols`
Derive new columns from existing columns.
If no new_col_map is provided, exits early returning the given lf.
Expects a dict:
```
{
    "new_col_name": {
        "fn_name": "add_cols",
        "fn_kwargs": {"cols": ["col_a", "col_b"]},
    },
    "other_new_name": {
        "fn_name": "mul_cols",
        "fn_kwargs": {"cols": ["col_a", "col_b", "col_c"]},
    }
}
```
Currently the valid derived functions are pulled from the derive_cols.py file.


## `nest_df_cols`
Nest given columns into a struct column.
If no nest_cols is provided, exits early returning the given lf.
Expects a dict of `{"struct_col_name": ["col_a", "col_b", "col_c"]}`.


## `drop_df_cols`
Drop the given columns from the lf.
If no drop_cols are provided, exits early returning the given lf.
Expects a list of `["col_to_drop_a", "col_to_drop_b"]`


## `pipe_custom_transformations`
Apply custom transformations to the lf.
If no custom_transformation_map is provided, exits early returning the given lf.
Must be given a dict of functions that meet this protocol: `dict[str, Callable[[pl.LazyFrame, Any], pl.LazyFrame]]`.
`custom_transformation_map` is expected as
```
{
    "custom_transformation_name": {"kwarg": "kwarg_value"},
    "other_custom_transformation_name": {"other_kwarg": "other_kwarg_value", "another_kwarg": 0},
}
```
If a custom function is listed in the `custom_transformation_map` that isn't in the `custom_transformation_fns`
will raise a KeyError.

::

# Repo map
```
├── .github
│   └── workflows
│       └── ci_tests.yaml
├── dev_tools
│   ├── __init__.py
│   └── update_readme.py
├── src
│   └── polars_pipe
│       ├── adapters
│       │   ├── __init__.py
│       │   ├── io_funcs.py
│       │   └── io_pl.py
│       ├── core
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── derive_cols.py
│       │   ├── inspect.py
│       │   ├── logger.py
│       │   ├── transform.py
│       │   └── validation.py
│       ├── services
│       │   ├── __init__.py
│       │   └── basic_pipeline.py
│       ├── __init__.py
│       └── __main__.py
├── tests
│   ├── adapters
│   │   ├── __init__.py
│   │   └── test_wrapper_apis.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── test_derive_cols.py
│   │   ├── test_inspect.py
│   │   └── test_validation.py
│   ├── services
│   │   ├── __init__.py
│   │   └── test_basic_pipeline.py
│   ├── __init__.py
│   └── conftest.py
├── .pre-commit-config.yaml
├── README.md
├── pyproject.toml
├── ruff.toml
└── uv.lock
::
```