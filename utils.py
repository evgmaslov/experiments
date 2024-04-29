from dataclasses import is_dataclass, fields
from typing import Dict, Type

def dict_to_dataclass(dictionary: Dict, class_type: Type):
    field_types = {field.name: field.type for field in fields(class_type)}
    for k in dictionary.keys():
        if k not in field_types or not is_dataclass(field_types[k]):
            continue
        dictionary[k] = dict_to_dataclass(dictionary[k], field_types[k])
    inst = class_type(**dictionary)
    return inst