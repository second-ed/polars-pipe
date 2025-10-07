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
│       │   └── io_pl.py
│       ├── core
│       │   ├── __init__.py
│       │   ├── logger.py
│       │   ├── transform.py
│       │   └── validation.py
│       ├── services
│       │   └── __init__.py
│       ├── __init__.py
│       └── __main__.py
├── tests
│   ├── adapters
│   │   ├── __init__.py
│   │   └── test_wrapper_apis.py
│   ├── core
│   │   └── __init__.py
│   ├── services
│   │   └── __init__.py
│   ├── __init__.py
│   └── test_main.py
├── .pre-commit-config.yaml
├── README.md
├── pyproject.toml
├── ruff.toml
└── uv.lock
::
```