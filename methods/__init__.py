from tasks import TaskInput, TaskOutput, TaskConfig, MLTaskConfig, STRING_TO_TASK_INPUT, STRING_TO_TASK_OUTPUT
from dataclasses import dataclass, fields, field
from typing import List, Any
from nn.data import Converter, DataConfig, STRING_TO_CONVERTER, STRING_TO_COLLATOR
from nn.model import STRING_TO_MODEL
from nn.model.base import ModelConfig
from nn.train import TrainerConfig, Trainer
import datasets
from datasets import load_dataset
from transformers import default_data_collator

@dataclass
class MethodConfig:
    steps: list = field(default_factory=lambda: ["solve"])
    task_config: TaskConfig = None

    def to_dict(self):
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        d["task_config"] = self.task_config.to_dict()
        return d

@dataclass
class MLMethodConfig(MethodConfig):
    steps: list = field(default_factory=lambda: ["prepare_data", "train", "solve"])
    task_config: MLTaskConfig = None
    data_config: DataConfig = None
    model_config: ModelConfig = None
    train_config: TrainerConfig = None

    def to_dict(self):
        d = {
            "steps": self.steps,
            "task_config": self.task_config.to_dict(),
            "data_config": self.data_config.to_dict(),
            "model_config": self.model_config.to_dict(),
            "train_config": self.train_config.to_dict(),
        }
        return d

class Method():
    def __init__(self, config: MethodConfig):
        self.config = config
        self.task_input_type = STRING_TO_TASK_INPUT.get(config.task_config.input_type, "None")
        self.task_output_type = STRING_TO_TASK_OUTPUT.get(config.task_config.output_type, "None")
    def solve(self, task_input: TaskInput) -> TaskOutput:
        pass
    def run_step(self, step_name: str):
        assert step_name in self.config.steps, f"Step {step_name} not in method steps: {', '.join(self.config.steps)}"
    def load_step(self, path: str, step_name: str):
        assert step_name in self.config.steps, f"Step {step_name} not in method steps: {', '.join(self.config.steps)}"
    def save_step(self, path: str, step_name: str):
        assert step_name in self.config.steps, f"Step {step_name} not in method steps: {', '.join(self.config.steps)}"

class MLMethod(Method):
    def __init__(self, config: MLMethodConfig):
        super().__init__(config)
        self.converter_type = STRING_TO_CONVERTER.get(config.data_config.converter_type, None)
        self.collator_type = STRING_TO_COLLATOR.get(config.data_config.collator_config.type, None)
        self.model_type = STRING_TO_MODEL.get(config.model_config.type, None)

        assert self.converter_type != None, f"Converter type {config.data_config.converter_type} isn't registered."
        assert self.model_type != None, f"Model type {config.model_config.type} isn't registered."

        self.dataset = None
        self.model = None
        self.trainer = None

    def solve(self, task_input: TaskInput) -> TaskOutput:
        super().solve(task_input)

    def run_step(self, step_name: str):
        super().run_step(step_name)
        if step_name == "prepare_data":
            self.prepare_data()
        elif step_name == "train":
            self.train()

    def load_step(self, path: str, step_name: str, load_from: str = "hub"):
        super().load_step(path, step_name)
        if step_name == "prepare_data":
            self.load_data()
        elif step_name == "train":
            self.model = self.model_type.from_pretrained(self.config.model_config.path, config=self.config.model_config)
            
    def save_step(self, path: str, step_name: str, save_to: str = "hub"):
        super().save_step(path, step_name)
        if step_name == "prepare_data":
            self.save_data(path, save_to)
        elif step_name == "train":
            if self.config.train_config.hub_model_id == None:
                self.trainer.args.hub_model_id = self.config.model_config.path
            self.trainer.push_to_hub()

    def prepare_data(self):
        converter = self.converter_type()
        self.dataset = converter(**self.config.task_config.data)
        self.dataset = self.dataset.train_test_split(test_size=self.config.data_config.split, shuffle=True, seed=42)
    def load_data(self):
        self.dataset = load_dataset(self.config.data_config.path).with_format("torch")
        if not isinstance(self.dataset, datasets.dataset_dict.DatasetDict):
            self.dataset = self.dataset.train_test_split(test_size=self.config.data_config.split, shuffle=True, seed=42)
        for split in self.dataset.keys():
            new_count = int(len(self.dataset[split])*self.config.data_config.use_size)
            self.dataset[split] = self.dataset[split].select(range(new_count))
    def save_data(self, path: str, save_to: str = "hub"):
        if save_to == "hub":
            self.dataset.push_to_hub(path)
        elif save_to == "dir":
            self.dataset.save_to_disk(path)
    
    def train(self):
        if self.model == None:
            self.model = self.model_type(self.config.model_config)

        if self.collator_type != None:
            collator = self.collator_type(self.config.data_config.collator_config)
        else:
            collator = None
        
        resume_from_checkpoint = self.config.train_config.resume_from_checkpoint

        self.trainer = Trainer(
            model=self.model,
            args=self.config.train_config,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            data_collator=collator
        )
        self.trainer.train(resume_from_checkpoint)