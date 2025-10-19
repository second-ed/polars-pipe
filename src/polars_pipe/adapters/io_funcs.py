from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from pathlib import Path

import polars as pl
import yaml


class FileType(Enum):
    JSON = "json"
    PARQUET = "parquet"
    CSV = "csv"
    YAML = "yaml"


READ_FUNCS = {
    FileType.JSON: pl.read_json,
    FileType.PARQUET: pl.scan_parquet,
    FileType.CSV: pl.scan_csv,
}

WriteFn = Callable[[pl.DataFrame | dict, str, dict], None]


def write_parquet(df: pl.DataFrame, path: str, **kwargs: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, **kwargs)


def write_yaml(data: dict, path: str, **kwargs: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, **kwargs))


WRITE_FUNCS = {FileType.PARQUET: write_parquet, FileType.YAML: write_yaml}
