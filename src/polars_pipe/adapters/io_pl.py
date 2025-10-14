from __future__ import annotations

from enum import Enum
from types import MappingProxyType

import attrs
import polars as pl
from attrs.validators import instance_of

from polars_pipe.core.logger import logger


class FileType(Enum):
    JSON = "json"
    PARQUET = "parquet"
    CSV = "csv"


READ_FUNCS = {
    FileType.JSON: pl.read_json,
    FileType.PARQUET: pl.read_parquet,
    FileType.CSV: pl.read_csv,
}


def write_parquet(df: pl.DataFrame, path: str, **kwargs: dict) -> None:
    df.write_parquet(path, **kwargs)


WRITE_FUNCS = {
    FileType.PARQUET: write_parquet,
}


@attrs.define
class IOBase:
    def read(self, path: str, file_type: FileType | str, **kwargs: dict) -> pl.DataFrame:
        logger.debug(f"{path = } {file_type = } {kwargs = }")
        file_type = self._get_file_type(file_type)

        if file_type not in self._read_funcs:
            raise NotImplementedError(f"`read` is not implemented for {file_type}")
        return self._read_funcs[file_type](path, **kwargs)

    def write(self, df: pl.DataFrame, path: str, file_type: FileType | str, **kwargs: dict) -> None:
        logger.debug(f"{df.shape = } {path = } {file_type = } {kwargs = }")
        file_type = self._get_file_type(file_type)

        if file_type not in self._write_funcs:
            raise NotImplementedError(f"`write` is not implemented for {file_type}")
        return self._write_funcs[file_type](df, path, **kwargs)

    def _get_file_type(self, file_type: FileType | str) -> FileType:
        return (
            file_type
            if isinstance(file_type, FileType)
            else FileType._member_map_[file_type.strip().upper()]
        )


@attrs.define
class IOWrapper(IOBase):
    _read_funcs: MappingProxyType = attrs.field(
        default=READ_FUNCS, validator=instance_of(MappingProxyType), converter=MappingProxyType
    )
    _write_funcs: MappingProxyType = attrs.field(
        default=WRITE_FUNCS, validator=instance_of(MappingProxyType), converter=MappingProxyType
    )


@attrs.define
class FakeIOWrapper(IOBase):
    files: dict = attrs.field(factory=dict, validator=instance_of(dict))
    _read_funcs: MappingProxyType = attrs.field(
        factory=dict, validator=instance_of(MappingProxyType), converter=MappingProxyType
    )
    _write_funcs: MappingProxyType = attrs.field(
        factory=dict, validator=instance_of(MappingProxyType), converter=MappingProxyType
    )

    def __attrs_post_init__(self) -> None:
        self._read_funcs = MappingProxyType(dict.fromkeys(READ_FUNCS, self._read_fn))
        self._write_funcs = MappingProxyType(dict.fromkeys(WRITE_FUNCS, self._write_fn))

    def _read_fn(self, path: str) -> pl.DataFrame:
        return self.files[path]

    def _write_fn(self, df: pl.DataFrame, path: str) -> None:
        self.files[path] = df
