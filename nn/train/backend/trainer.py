from ..track import TrackerConfig, STRING_TO_TRACKER
from ..loss import STRING_TO_LOSS
from .checkpoint import CheckpointManager, SaveCheckpointCallback

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import Dataset
from transformers import Trainer as HfTrainer, TrainingArguments as HfTrainerConfig, TrainerState as HfTrainerState, TrainerControl as HfTrainerControl, ProgressCallback, PrinterCallback
from typing import Callable, Dict, List, Tuple, Any
from dataclasses import dataclass, fields
import dataclasses
from torch import nn 
import torch
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import ModelOutput 
import copy
import json

@dataclass
class TrainerConfig(HfTrainerConfig):
   tracker_config: TrackerConfig = None
   loss_fn: str = ""
   tracker_type: str = ""
   checkpoint_hub: str = ""

   def to_dict(self):
        d = super().to_dict()
        d["tracker_config"] = self.tracker_config.to_dict()
        d["tracker_config"] = self.tracker_type
        d["loss_fn"] = self.loss_fn
        return d

@dataclass
class TrainerState(HfTrainerState):
    trainer: Any = None
    def save_to_json(self, json_path: str):
        trainer = self.trainer
        self.trainer = ""
        super().save_to_json(json_path)
        self.trainer = trainer

class Trainer(HfTrainer):
    def __init__(self, model: PreTrainedModel | nn.Module = None, args: TrainerConfig = None, data_collator: Any | None = None, train_dataset: Dataset | None = None, eval_dataset: Dataset | Dict[str, Dataset] | None = None, tokenizer: PreTrainedTokenizerBase | None = None, model_init: Callable[[], PreTrainedModel] | None = None, compute_metrics: Callable[[EvalPrediction], Dict] | None = None, callbacks: List[TrainerCallback] | None = None, optimizers: Tuple[Optimizer, LambdaLR] = (None, None), preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None):
        if not isinstance(args, TrainerConfig):
            super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
            return
        
        self.loss_fn = STRING_TO_LOSS.get(args.loss_fn, None)
        
        tracker_type = STRING_TO_TRACKER.get(args.tracker_type, None)
        if tracker_type == None:
            self.tracker = None
        else:
            self.tracker = tracker_type(args.tracker_config)
            self._state = None
            if callbacks == None:
                callbacks = []
            for call in self.tracker.callbacks:
                callbacks.append(call)
        
        self.checkpoint_manager = CheckpointManager(args.output_dir, args.checkpoint_hub)
        callbacks.append(SaveCheckpointCallback(self.checkpoint_manager))
        
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        if self.tracker != None:
          self.remove_callback(ProgressCallback)
          self.remove_callback(PrinterCallback)
        
    @property
    def state(self):
        return self._state
    @state.setter
    def state(self, value):
        self._state = TrainerState(trainer=self, **dataclasses.asdict(value))

    def compute_loss(self, model: PreTrainedModel | nn.Module, inputs: Dict, return_outputs=False) -> Tuple[torch.Tensor, ModelOutput] | torch.Tensor:
        base_inputs = {k: copy.copy(v) for k, v in inputs.items()}
        if self.loss_fn == None:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        else:
            loss, outputs = self.loss_fn(model, inputs, return_outputs=True)
        if self.tracker != None:
            self.tracker.last_train_batch = {"loss":loss, "inputs":base_inputs, "outputs":outputs}
        return (loss, outputs) if return_outputs else loss
    def prediction_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor | Any], prediction_loss_only: bool, ignore_keys: List[str] | None = None) -> Tuple[torch.Tensor | None]:
        base_inputs = {k: copy.copy(v) for k, v in inputs.items()}
        loss, outputs, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        if self.tracker != None:
            self.tracker.last_val_batch = {"loss":loss, "inputs":base_inputs, "outputs":outputs}
        return (loss, base_inputs, outputs)