import logging
from typing import List, Union
from transformers import AutoProcessor, AutoModelForCausalLM

import torch
import numpy as np


class ASRInference:
    '''
    ASR Inference class
    '''
    def __init__(self, 
                 model_dir: str = "google/gemma-4-E4B-it", 
                 language: str = "ENGLISH") -> None:

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
                
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype="auto",
            device_map="auto"
        )

        self.language = language

    def transcribe(
            self,
            wav_path: str
        ) -> str:
        '''
        Transcribe function
        '''
        if self.language:
            prompt = f''' 
                    Transcribe the following speech segment in {self.language} into {self.language} text.

                    Follow these specific instructions for formatting the answer:
                    * Only output the transcription, with no newlines.
                    '''
        else:
            prompt = '''Transcribe the following speech segment in its original language. 
            Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.'''

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": wav_path},
                    {"type": "text", "text": prompt},
                ]
            }
        ]
        inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(**inputs, max_new_tokens=1024)
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        try:
            transcription = self.processor.parse_response(response)['content']
        except Exception as e:
            print(f"Exception occured at {wav_path}. \n Error: {e}")
            transcription = ''

        return transcription

    def translation(
            self,
            wav_path,
            source_language: str = None,
            target_language: str = 'en'
    ) -> str:
        '''
        Translation function
        '''
        if source_language:
            prompt = f"""Transcribe the following speech segment in {source_language}, then translate it into {target_language}.
                    When formatting the answer, first output the transcription in {source_language}, 
                    then one newline, then output the string '{target_language}: ', then the translation in {target_language}.
                    """
        else:
            prompt = f"""No language is given
                    Transcribe the following speech segment then translate it into {target_language}.
                    When formatting the answer, first output the transcription, 
                    then one newline, then output the string '{target_language}: ', then the translation in {target_language}.
                    """

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": wav_path},
                    {"type": "text", "text": prompt},
                ]
            }
        ]
        inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(**inputs, max_new_tokens=1024)
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        try:
            transcript, translation = self.processor.parse_response(response)['content'].split('\n')
        except Exception:
            transcript, translation  = '', ''

        return transcript, translation

    def batch_transcribe(
        self,
        audio_tensors: Union[List[np.ndarray]]
    ) -> List[str]:
        
        print("batch not implemented")
        
        return 