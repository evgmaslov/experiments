from datasets import Dataset
from dataclasses import dataclass, fields
from typing import Any, Dict
from .printers import VolumePrinter
from .converters import TesrConverter
from .collators import TesrCollator
from .base import Converter, Printer, CollatorConfig

@dataclass
class DataConfig:
    name: str
    path: str
    converter_type: str
    collator_config: CollatorConfig
    split: float = 0.2
    use_size: float = 1.0
    

    def to_dict(self):
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return d

STRING_TO_CONVERTER = {
    "Converter":Converter,
    "TesrConverter":TesrConverter,
}

STRING_TO_COLLATOR = {
    "TesrCollator":TesrCollator,
}

STRING_TO_PRINTER = {
    "VolumePrinter":VolumePrinter,
}