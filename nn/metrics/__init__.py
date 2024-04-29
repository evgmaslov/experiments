from typing import Union, Tuple
from transformers.trainer_utils import EvalPrediction
from .classification import AccuracyScore

STRING_TO_SCORE = {
    "AccuracyScore":AccuracyScore
}
    