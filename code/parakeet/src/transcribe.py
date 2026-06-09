from typing import Union

import torch
import librosa
import numpy as np

from parakeet.src.models.asr import ASRInference

class ParakeetEvaluator:
    def __init__(
            self,
            model_tag: str):

        self.name = 'NvidiaParakeet'
        self.batch_transcription = False
        self.asr_model = ASRInference(model_path=model_tag)
        self.target_sr = 16000

    def infer(self, filepath:str) -> str:
        """
        Main method to pass the audio array into the model
        """
        transcription = self.asr_model.transcribe(filepath)
        return transcription

    def infer_file(self, filepath: str) -> str:
        """
        Method to take in a filepath and convert into an array
        before passing into the model
        """
        return self.infer(filepath)

    def evaluate(self):
        """
        Method to take in a manifest_filepath and 
        ready test data for data ingestion into the model
        """
        return NotImplementedError

    def evaluate_batch(self):
        return NotImplementedError
