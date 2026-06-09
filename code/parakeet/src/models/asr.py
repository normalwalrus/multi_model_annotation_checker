import logging
import os
from typing import List, Union

import torch
import nemo.collections.asr as nemo_asr

class ASRInference:
    '''
    ASR Inference class
    '''
    def __init__(self, model_path: str = "nvidia/parakeet-tdt-0.6b-v2") -> None:

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # A `.nemo` checkpoint on disk is restored; an NGC/HF model tag
        # (e.g. "nvidia/parakeet-tdt-0.6b-v2") is downloaded via from_pretrained.
        if os.path.isfile(model_path):
            self.model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path)
        else:
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_path)

        self.model = self.model.to(self.device)
        self.model.eval()

    def transcribe(
            self,
            audio_path: str,
        ) -> str:
        '''
        Transcribe function
        '''

        output = self.model.transcribe([audio_path])

        predicted_text_1 = output[0].text
        
        return predicted_text_1
