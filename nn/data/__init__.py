from datasets import Dataset
from dataclasses import dataclass, fields
from typing import Any, Dict
from .printers import (
    VolumePrinter,
    VolumePrinterWithCondition
)
from .converters import (
    TesrConverter,
    LoadDatasetConverter,
)
from .collators import (
    TesrCollator,
    TesrCollatorWithCondition
)
from .base import Converter, Printer, CollatorConfig

STRING_TO_CONVERTER = {
    "Converter":Converter,
    "TesrConverter":TesrConverter,
    "LoadDatasetConverter":LoadDatasetConverter,
}

STRING_TO_COLLATOR = {
    "TesrCollator":TesrCollator,
    "TesrCollatorWithCondition":TesrCollatorWithCondition,
}

STRING_TO_PRINTER = {
    "VolumePrinter":VolumePrinter,
    "VolumePrinterWithCondition":VolumePrinterWithCondition,
}