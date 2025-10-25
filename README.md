# polars-pipe


# Repo map
```
├── .github
│   └── workflows
│       └── ci_tests.yaml
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