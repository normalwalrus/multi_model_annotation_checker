import logging
from typing import List, Union

import torch
import numpy as np


class ASRInference:
    '''
    ASR Inference class
    '''
    def __init__(self, 
                 model_dir: str = "Qwen/Qwen3-ASR-1.7B", 
                 language: str = "ENGLISH") -> None:

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(self.device)
        from qwen_asr import Qwen3ASRModel
        
        self.model = Qwen3ASRModel.from_pretrained(
            model_dir,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            # attn_implementation="flash_attention_2",
            max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
            max_new_tokens=256, # Maximum number of tokens to generate. Set a larger value for long audio input.
        )
        
        self.language = language

    def transcribe(
            self,
            wav_path: str
        ) -> str:
        '''
        Transcribe function
        '''

        res = self.model.transcribe(
            audio=wav_path,
            language=self.language, # set "English" to force the language
        )
        try:
            text = res[0].text
        except:
            print(f"ERROR WITH TRANSCRIBE: Res = {res}")
            text = ''

        return text
    
    def batch_transcribe(
        self,
        audio_tensors: Union[List[np.ndarray]]
    ) -> List[str]:
        
        print("batch not implemented")
        
        return 