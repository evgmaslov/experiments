from datasets import Dataset
from dataclasses import dataclass, fields
from typing import Any, Dict
from .printers import VolumePrinter
from .converters import TesrConverter
from .base import Converter, Printer

@dataclass
class DataConfig:
    name: str
    path: str
    split: float = 0.2
    use_size: float = 1.0
    converter_type: str

    def to_dict(self):
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return d

STRING_TO_CONVERTER = {
    "Converter":Converter,
    "TesrConverter":TesrConverter,
}

STRING_TO_PRINTER = {
    "VolumePrinter":VolumePrinter,
}