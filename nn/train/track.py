from ..metrics import STRING_TO_SCORE
from ..data import STRING_TO_PRINTER

from transformers.trainer_callback import TrainerCallback
from dataclasses import dataclass, fields
from typing import List
from tqdm.notebook import tqdm
import copy
from transformers.trainer_utils import EvalPrediction
import random

class DescriptionCallback(TrainerCallback):
    def __init__(self, tracker):
        self.tracker = tracker
        self.training_bar = None
        self.validation_bar = None
        self.current_step = 0
        self.epoch_counter = 0
        self.history = {
            "train":{
                "loss":[]
            },
            "val":{
                "loss":[]
            }
        }
        for score in self.tracker.metrics:
            for key in self.history.keys():
                self.history[key][score.name] = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            n_steps = int(state.max_steps/state.num_train_epochs)
            self.training_bar = tqdm(total=n_steps, desc="Training")
        self.current_step = 0
        for key in self.history.keys():
          self.history[key]["loss"].append([])
          for score in self.tracker.metrics:
            self.history[key][score.name].append([])

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
          return
        step = state.global_step - int(state.max_steps/state.num_train_epochs)*self.epoch_counter
        self.training_bar.update(step - self.current_step)
        self.current_step = step
        if self.tracker.last_train_batch != None:
            batch = self.tracker.last_train_batch
            loss = round(batch["loss"].item(), 5)
            description = f"Loss: {loss}"
            self.history["train"]["loss"][-1].append(loss)
            for score in self.tracker.metrics:
                labels = [batch["inputs"][key].detach().cpu().numpy() for key in score.label_names]
                preds = [batch["outputs"][key].detach().cpu().numpy() for key in score.output_names]
                value = score(EvalPrediction(preds, labels, None))
                description = description + f"; {score.name}: {value}"
                self.history["train"][score.name][-1].append(value)
            self.training_bar.set_description(description)

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if not state.is_world_process_zero  and len(eval_dataloader) == 0:
          return
        if self.validation_bar is None:
            self.validation_bar = tqdm(
                total=len(eval_dataloader), desc="Validation"
            )
        self.validation_bar.update(1)
        if self.tracker.last_val_batch != None:
            batch = self.tracker.last_val_batch
            loss = round(batch["loss"].item(), 5)
            description = f"Loss: {loss}"
            self.history["val"]["loss"][-1].append(loss)
            for score in self.tracker.metrics:
                labels = [batch["inputs"][key].detach().cpu().numpy() for key in score.label_names]
                preds = [batch["outputs"][key].detach().cpu().numpy() for key in score.output_names]
                value = score(EvalPrediction(preds, labels, None))
                description = description + f"; {score.name}: {value}"
                self.history["val"][score.name][-1].append(value)
            self.validation_bar.set_description(description)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.validation_bar is not None:
                self.validation_bar.close()
            self.validation_bar = None

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar.close()
            self.training_bar = None
            self.epoch_counter += 1
            train_report = ', '.join([f"{k}: {sum(v[-1])/len(v[-1])}" for k, v in self.history["train"].items()])
            print(f"Epoch {self.epoch_counter}. Train: {train_report}.")

class PrintOutputsCallback(TrainerCallback):
    def __init__(self, tracker):
        self.tracker = tracker
        self.epoch_counter = 0
    def on_evaluate(self, args, state, control, eval_dataloader, **kwargs):
        if not state.is_world_process_zero or not hasattr(state, "trainer") or self.tracker.printer == None:
            return
        if not (self.epoch_counter % self.tracker.config.print_outputs_freq == 0 or (self.epoch_counter == 1 and self.tracker.config.print_first)):
            return
        indexes = random.sample(range(len(eval_dataloader.dataset)), self.tracker.config.num_outputs)
        inputs = [eval_dataloader.dataset[i] for i in indexes]
        processed = eval_dataloader.collate_fn(inputs)
        self.tracker.printer(processed)
        device = "cuda"
        if state.trainer.args.use_cpu:
          device = "cpu"
        for key in processed.keys():
          processed[key] = processed[key].to(device)
        state.trainer.model.inference(processed, True)
        
    def device_dict(self, dictionary):
        for k in dictionary.keys():
            if type(dictionary[k]) == type({}):
                dictionary[k] = self.device_dict(dictionary[k])
            else:
                dictionary[k] = dictionary[k].to(self.device)
        return dictionary
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.epoch_counter += 1

@dataclass
class TrackerConfig:
    metrics: List[str]
    print_outputs: bool = True
    printer_type: str = ""
    print_outputs_freq: int = 1
    num_outputs: int = 1
    print_first: bool = True

    def to_dict(self):
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        return d

class Tracker():
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.metrics = []
        for metric_name in config.metrics:
            metric = STRING_TO_SCORE.get(metric_name, None)
            if metric != None:
                self.metrics.append(metric)
        
        printer_type = STRING_TO_PRINTER.get(config.printer_type, None)
        if printer_type == None:
            self.printer = None
        else:
            self.printer = printer_type()

        self.callbacks = [DescriptionCallback(self), PrintOutputsCallback(self)]
        self.last_train_batch = None
        self.last_val_batch = None

STRING_TO_TRACKER = {
    "Tracker":Tracker,
}