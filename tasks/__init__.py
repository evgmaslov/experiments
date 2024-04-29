import torch
from typing import List, Any, Dict
from dataclasses import dataclass, fields
from collections import OrderedDict
from .generation import GenerationTaskInput, GenerationTaskOutput
from .base import TaskInput, TaskOutput

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
    def report(self, methods: List):
        pass

STRING_TO_TASK_INPUT = {
    "TaskInput":TaskInput,
    "GenerationTaskInput":GenerationTaskInput,
}

STRING_TO_TASK_OUTPUT = {
    "TaskOutput":TaskOutput,
    "GenerationTaskOutput":GenerationTaskOutput,
}