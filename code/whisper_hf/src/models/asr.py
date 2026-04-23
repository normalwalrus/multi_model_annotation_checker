import logging
from typing import List, Union

import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

class ASRInference:
    '''
    ASR Inference class
    '''
    def __init__(self, 
                 model_dir: str, 
                 precision: str = "float16", 
                 language: str = "ENGLISH") -> None:

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(self.device)

        if precision == "float16":
            self.torch_dtype = torch.float16
        elif precision == "float32":
            self.torch_dtype = torch.float32
        else:
            raise Exception(f"No such precision type: {precision}. Choose between float16 and float32")

        self.processor = WhisperProcessor.from_pretrained(model_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_dir, torch_dtype=self.torch_dtype)
        self.model.to(self.device)
        
        # Set for the specific LANGUAGE and task
        LANGUAGE = language
        TASK = 'transcribe'
        self.model.config.forced_decoder_ids = self.processor.tokenizer.get_decoder_prompt_ids(
            language=LANGUAGE, task=TASK
            )
        self.model.config.suppress_tokens = []
        self.model.generation_config.forced_decoder_ids = self.processor.tokenizer.get_decoder_prompt_ids(
            language=LANGUAGE, task=TASK
            )
        self.model.generation_config.suppress_tokens = []
        self.model.eval()
        
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def transcribe(
            self,
            audio_tensor: torch.tensor
        ) -> str:
        '''
        Transcribe function
        '''

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                text = self.pipeline(audio_tensor,
                                    max_new_tokens=255,
                                    return_timestamps=True)["text"]

        return text
    
    def batch_transcribe(
        self,
        audio_tensors: Union[List[np.ndarray]]
    ) -> List[str]:
        
        transcription_list = []
        
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                text_list = self.pipeline(audio_tensors,
                                          max_new_tokens=255,
                                          return_timestamps=True)
            
                for text in text_list:
                    
                    transcription_list.append(text['text'])

        return transcription_list