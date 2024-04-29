import torch
from typing import List, Any, Dict
from dataclasses import dataclass, fields
from methods import Method
from collections import OrderedDict
from generation import GenerationTaskInput, GenerationTaskOutput

class TaskInput(OrderedDict):
    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

class TaskOutput(OrderedDict):
    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

@dataclass
class TaskConfig:
    name: str
    input_type: str
    output_type: str

    def to_dict(self):
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return d

@dataclass
class MLTaskConfig(TaskConfig):
    data: Dict

class Task():
    def __init__(self, config: TaskConfig):
        self.config = config
    def report(self, methods: List[Method]):
        pass

STRING_TO_TASK_INPUT = {
    "TaskInput":TaskInput,
    "GenerationTaskInput":GenerationTaskInput,
}

STRING_TO_TASK_OUTPUT = {
    "TaskOutput":TaskOutput,
    "GenerationTaskOutput":GenerationTaskOutput,
}