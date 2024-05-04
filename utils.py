from dataclasses import is_dataclass, fields
from typing import Dict, Type

def dict_to_dataclass(dictionary: Dict, class_type: Type, custom_field_types: Dict = None):
    field_types = {field.name: field.type for field in fields(class_type)}
    for k in dictionary.keys():
        if k not in field_types or not is_dataclass(field_types[k]):
            continue
        field_type = field_types[k]
        if isinstance(custom_field_types, dict):
          if k in custom_field_types:
            field_type = custom_field_types[k]
        dictionary[k] = dict_to_dataclass(dictionary[k], field_type)
    inst = class_type(**dictionary)
    return inst