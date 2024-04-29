from transformers.trainer_utils import EvalPrediction

class Score():
    def __init__(self):
        self.name = "Dummy score"
        self.label_names = []
    def __call__(self, pred: EvalPrediction) -> int:
        return 0