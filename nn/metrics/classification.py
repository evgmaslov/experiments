from . import Score
from transformers.trainer_utils import EvalPrediction
import evaluate
import numpy as np

class AccuracyScore(Score):
    def __init__(self):
        self.name = "Accuracy"
        self.label_names = ["labels"]
        self.output_names = ["logits"]
        self.score = evaluate.load("accuracy")
    def __call__(self, pred: EvalPrediction) -> int:
        predictions, labels = pred
        predictions = np.argmax(predictions[0], axis=1)
        return self.score.compute(predictions=predictions, references=labels[0])["accuracy"]