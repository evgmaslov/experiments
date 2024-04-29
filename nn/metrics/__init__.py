from typing import Union, Tuple
from transformers.trainer_utils import EvalPrediction
from classification import AccuracyScore

class Score():
    def __init__(self):
        self.name = "Dummy score"
        self.label_names = []
    def __call__(self, pred: EvalPrediction) -> int:
        return 0

STRING_TO_SCORE = {
    "AccuracyScore":AccuracyScore
}
    