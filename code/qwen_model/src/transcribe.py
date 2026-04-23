from typing import Union

import torch
import torchaudio
import librosa
import numpy as np

from qwen_model.src.models.asr import ASRInference

class QwenEvaluator:
    def __init__(
            self,
            language: str,
            model_dir: str = "/pretrained_models"):
        
        self.name = 'Qwen3ASREvaluator'
        self.batch_transcription = False
        self.model = ASRInference(model_dir, language)

    def infer(self, filepath) -> str:
        """
        Main method to pass the audio array into the model
        """

        # if isinstance(arr, np.ndarray):
        #     arr = torch.tensor(arr)
        # assert isinstance(arr, torch.Tensor)

        transcription = self.model.transcribe(filepath)

        return transcription
    
    def batch_infer_file(self, filepaths: list[str]) -> list[str]:
        '''
        Method to take in multiple filepaths and convert to arrays
        before passing into model
        '''
        
        audio_tensors = []
        for path in filepaths:
            waveform, sample_rate = librosa.load(path, sr=16000, mono=True)
            
            assert len(waveform.shape) == 1
            
            # if isinstance(waveform, np.ndarray):
            #     waveform = torch.tensor(waveform)
            # assert isinstance(waveform, torch.Tensor)
            
            audio_tensors.append(waveform)
        
        return self.model.batch_transcribe(audio_tensors)
        

    def infer_file(self, filepath: str) -> str:
        """
        Method to take in a filepath and convert into an array
        before passing into the model
        """
        # audio, _ = librosa.load(filepath, sr=16000, mono=True)
        # assert len(audio.shape) == 1

        return self.infer(filepath)
