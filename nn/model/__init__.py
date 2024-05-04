from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput 
from tasks import TaskInput
from dataclasses import dataclass, fields
from typing import Callable, Dict, Any
from .volume_generation import DiffusersDDPM3D, SeisFusion
from .base import Model

STRING_TO_MODEL = {
    "Model":Model,
    "DiffusersDDPM3D":DiffusersDDPM3D,
    "SeisFusion":SeisFusion,
}