from gemma.src.models.asr import ASRInference

class GemmaEvaluator:
    def __init__(
            self,
            language: str,
            model_dir: str = "/pretrained_models"):
        
        self.name = 'GemmaASREvaluator'
        self.batch_transcription = False
        self.model = ASRInference(model_dir, language)

    def infer(self, filepath) -> str:
        """
        Main method to pass the audio array into the model
        """

        transcription = self.model.transcribe(filepath)

        return transcription
    
    def infer_translation(self, filepath, source_lang, target_lang) -> str:
        """
        Main method to pass the audio array into the model
        """

        translation = self.model.translation(
            filepath,
            source_language=source_lang,
            target_language=target_lang
        )

        return translation

    def batch_infer_file(self, filepaths: list[str]) -> list[str]:
        '''
        Method to take in multiple filepaths and convert to arrays
        before passing into model
        '''
        

        return 
        

    def infer_file(self, filepath: str) -> str:
        """
        Method to take in a filepath and convert into an array
        before passing into the model
        """
        # audio, _ = librosa.load(filepath, sr=16000, mono=True)
        # assert len(audio.shape) == 1

        return self.infer(filepath)
