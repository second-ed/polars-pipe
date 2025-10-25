from __future__ import annotations

from pathlib import Path
from typing import Self

import attrs

import polars_pipe.adapters.io_pl as io


def abs_path(path: str) -> str:
    return str(Path(path).absolute())


@attrs.define
class GeneralConfig:
    guid: str = attrs.field(validator=attrs.validators.instance_of(str))
    date_time: str = attrs.field(
        validator=attrs.validators.instance_of(str), converter=lambda x: x.strftime("%Y%m%d_%H%M")
    )
    process_name: str = attrs.field(validator=attrs.validators.instance_of(str))
    src_path: str = attrs.field(validator=attrs.validators.instance_of(str), converter=abs_path)
    src_file_type: str = attrs.field(
        validator=[
            attrs.validators.instance_of(str),
            attrs.validators.in_(io.FileType._member_map_),
        ],
        converter=str.upper,
    )
    dst_root: str = attrs.field(validator=attrs.validators.instance_of(str), converter=abs_path)
    dst_file_type: str = attrs.field(
        validator=[
            attrs.validators.instance_of(str),
            attrs.validators.in_(io.FileType._member_map_),
        ],
        converter=str.upper,
    )
    validation: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    transformations: dict = attrs.field(factory=dict, validator=attrs.validators.instance_of(dict))
    custom_transformations: dict = attrs.field(
        factory=dict, validator=attrs.validators.instance_of(dict)
    )
    pipeline_plan: list = attrs.field(factory=list, validator=attrs.validators.instance_of(list))
    valid_dst_stem: str = attrs.field(
        default="transformed_data",
        validator=attrs.validators.instance_of(str),
    )
    invalid_dst_stem: str = attrs.field(
        default="error_records",
        validator=attrs.validators.instance_of(str),
    )
    config_dst_stem: str = attrs.field(
        default="config",
        validator=attrs.validators.instance_of(str),
    )
    desc_stats_stem: str = attrs.field(
        default="desc_stats",
        validator=attrs.validators.instance_of(str),
    )

    @classmethod
    def from_dict(cls, config: dict) -> Self:
        filtered = {f.name: config[f.name] for f in attrs.fields(cls) if f.name in config}
        return cls(**filtered)

    def to_dict(self) -> dict:
        return attrs.asdict(self)
