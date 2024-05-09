from collections import OrderedDict
from typing import List, Any, Dict
from dataclasses import dataclass, fields
from ..nn.data import STRING_TO_CONVERTER
from ..nn.data.base import ConverterConfig
from datasets import get_dataset_split_names

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

class Task():
    def __init__(self, config: TaskConfig):
        self.config = config
    def report(self, methods: List):
        pass
    def get_task_input(self) -> TaskInput:
        pass

@dataclass
class MLTaskConfig(TaskConfig):
    train_data: Dict
    test_data: Dict
    test_data_converter_config: ConverterConfig

class MLTask(Task):
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.converter_type = STRING_TO_CONVERTER.get(config.data_converter, None)
        assert self.converter_type != None, f"Converter type {config.data_config.converter_type} isn't registered."
    
    def get_task_input(self, resize: float = None) -> TaskInput:
        converter = self.converter_type()
        dataset = converter(self.config.test_data)
        task_input = TaskInput()
        splits = get_dataset_split_names(self.config.test_data["path"])
        if len(splits) > 0:
            if "test" in splits:
                dataset = dataset["test"]
            else:
                dataset = dataset[splits[0]]
        for key in dataset.column_names:
            data = dataset[key]
            if resize != None:
                new_len = int(len(dataset)*resize)
                data = data[:new_len]
            task_input[key] = dataset[key]
        return task_input