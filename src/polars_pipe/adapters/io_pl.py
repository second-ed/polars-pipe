from __future__ import annotations

import datetime
import math
from abc import ABC, abstractmethod
from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

import attrs
import polars as pl
from attrs.validators import instance_of

from polars_pipe.adapters.io_funcs import READ_FUNCS, WRITE_FUNCS, FileType, WriteFn
from polars_pipe.core.logger import logger


@attrs.define
class IOBase(ABC):
    def read(self, path: str, file_type: FileType | str, **kwargs: dict) -> pl.LazyFrame:
        logger.debug(f"{path = } {file_type = } {kwargs = }")
        file_type = self._get_file_type(file_type)

        if file_type not in self._read_funcs:
            raise NotImplementedError(f"`read` is not implemented for {file_type}")
        return self._read_funcs[file_type](path, **kwargs)

    def write(
        self, data: pl.LazyFrame | dict, path: str, file_type: FileType | str, **kwargs: dict
    ) -> None:
        logger.debug(f"{path = } {file_type = } {kwargs = }")
        file_type = self._get_file_type(file_type)

        if file_type not in self._write_funcs:
            raise NotImplementedError(f"`write` is not implemented for {file_type}")
        if file_type == FileType.YAML:
            return self._write_funcs[file_type](data, path, **kwargs)
        return self._sink_in_chunks(
            data,
            write_func=self._write_funcs[file_type],
            ext=file_type.value,
            base_path=path,
            fwd_kwargs=kwargs,
        )

    def _get_file_type(self, file_type: FileType | str) -> FileType:
        return (
            file_type
            if isinstance(file_type, FileType)
            else FileType._member_map_[file_type.strip().upper()]
        )

    def _sink_in_chunks(  # noqa: PLR0913
        self,
        lf: pl.LazyFrame,
        write_func: WriteFn,
        ext: str,
        base_path: str | Path,
        target_size_gb: float | None = 1.0,
        fwd_kwargs: dict | None = None,
    ) -> None:
        fwd_kwargs = fwd_kwargs or {}
        base_path: Path = Path(base_path)

        sample = lf.slice(0, 10_000).collect()
        avg_row_size = sample.estimated_size() / sample.height
        rows_per_chunk = int((target_size_gb * 1e9) / avg_row_size)

        total_rows = lf.select(pl.len()).collect().item()
        n_chunks = math.ceil(total_rows / rows_per_chunk)

        for i in range(n_chunks):
            offset = i * rows_per_chunk
            length = min(rows_per_chunk, total_rows - offset)
            chunk_df = lf.slice(offset, length).collect()
            part_path = base_path / f"part-{i:05d}-{self.get_guid()}.{ext.lower()}"
            write_func(chunk_df, part_path, **fwd_kwargs)

    @abstractmethod
    def get_guid(self) -> str:
        pass

    @abstractmethod
    def get_datetime(self) -> datetime.datetime:
        pass


@attrs.define
class IOWrapper(IOBase):
    _read_funcs: MappingProxyType = attrs.field(
        default=READ_FUNCS, validator=instance_of(MappingProxyType), converter=MappingProxyType
    )
    _write_funcs: MappingProxyType = attrs.field(
        default=WRITE_FUNCS, validator=instance_of(MappingProxyType), converter=MappingProxyType
    )

    def get_guid(self) -> str:
        return str(uuid4())

    def get_datetime(self) -> datetime.datetime:
        return datetime.datetime.now(datetime.UTC)


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

    def _read_fn(self, path: str) -> pl.LazyFrame | dict:
        data = self.files[path]
        return data.lazy() if isinstance(data, pl.DataFrame) else data

    def _write_fn(self, df: pl.DataFrame, path: str) -> None:
        self.files[str(path)] = df

    def get_guid(self) -> str:
        return "abc-123"

    def get_datetime(self) -> datetime.datetime:
        return datetime.datetime(2025, 10, 16, 12, tzinfo=datetime.UTC)
