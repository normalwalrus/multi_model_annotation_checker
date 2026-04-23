from typing import Union

import torch
import torchaudio
import librosa
import numpy as np

from whisper_hf.src.models.asr import ASRInference

class WhisperEvaluator:
    def __init__(
            self,
            language: str,
            model_dir: str = "/pretrained_models",
            precision: str = "float16"):
        
        self.name = 'WhisperEvaluator'
        self.batch_transcription = True
        self.model = ASRInference(model_dir, precision, language)

    def infer(self, arr: Union[np.ndarray, torch.Tensor]) -> str:
        """
        Main method to pass the audio array into the model
        """

        transcription = self.model.transcribe(arr)

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
        audio, _ = librosa.load(filepath, sr=16000, mono=True)
        assert len(audio.shape) == 1

        return self.infer(audio)
