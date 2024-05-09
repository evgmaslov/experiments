import torch
from collections import OrderedDict
from .generation import GenerationTaskInput, GenerationTaskOutput
from .base import TaskInput, TaskOutput

STRING_TO_TASK_INPUT = {
    "TaskInput":TaskInput,
    "GenerationTaskInput":GenerationTaskInput,
}

STRING_TO_TASK_OUTPUT = {
    "TaskOutput":TaskOutput,
    "GenerationTaskOutput":GenerationTaskOutput,
}