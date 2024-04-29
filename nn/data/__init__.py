from datasets import Dataset
from dataclasses import dataclass, fields
from typing import Any, Dict
from printers import VolumePrinter
from converters import TesrConverter

@dataclass
class DataConfig:
    name: str
    path: str
    converter_type: str

    def to_dict(self):
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return d

class Converter():
    def __call__(self, *args, **kwargs) -> Dataset:
        pass

class Printer():
    def __call__(self, data: Dict):
        pass

STRING_TO_CONVERTER = {
    "Converter":Converter,
    "TesrConverter":TesrConverter,
}

STRING_TO_PRINTER = {
    "VolumePrinter":VolumePrinter,
}