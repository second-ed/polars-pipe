import attrs
import polars as pl
from enum import Enum, auto
from polars_pipe.core.logger import logger

class FileType(Enum):
    JSON = auto()
    PARQUET = auto()
    CSV = auto()

READ_FUNCS = {
    FileType.JSON: pl.read_json,
    FileType.PARQUET: pl.read_parquet,
    FileType.CSV: pl.read_csv,
}


def write_parquet(df: pl.DataFrame, path: str, **kwargs) -> None:
    df.write_parquet(path, **kwargs)

WRITE_FUNCS = {
    FileType.PARQUET: write_parquet,
}

@attrs.define
class IOWrapper:
    _read_funcs: dict = attrs.field(default=READ_FUNCS)
    _write_funcs: dict = attrs.field(default=WRITE_FUNCS)
    
    def read(self, path: str, file_type: FileType, **kwargs) -> pl.DataFrame:
        logger.debug(f"{path = } {file_type = } {kwargs = }")
        if file_type not in self._read_funcs:
            raise NotImplementedError(f"`read` is not implemented for {file_type}")
        return self._read_funcs[file_type](path, **kwargs)
    
    def write(self, df: pl.DataFrame, path: str, file_type: FileType, **kwargs) -> pl.DataFrame:
        logger.debug(f"{path = } {file_type = } {kwargs = }")
        if file_type not in self._write_funcs:
            raise NotImplementedError(f"`write` is not implemented for {file_type}")
        return self._write_funcs[file_type](df, path, **kwargs)