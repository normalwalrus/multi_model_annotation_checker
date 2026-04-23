import librosa
import soundfile as sf
import numpy as np
from IPython.display import Audio
import scipy.signal

def get_duration(audio_filepath):
    ''' Get the duration of an audio file in seconds '''
    
    return librosa.get_duration(filename=audio_filepath)

def get_segment_of_audio(audio_filepath, start_time, end_time, sr = 16000):
    ''' Get a segment of an audio file '''
    
    y, sr = librosa.load(audio_filepath, sr=sr, offset=start_time, duration=end_time - start_time)
    
    return y, sr

def get_segment_of_audio_wav(wav,  start_time, end_time, sr = 16000):
    ''' Get segment of audio from audio wav'''
    
    start_frame = int(start_time * sr)
    end_frame = int(end_time * sr)
    
    if end_frame <= len(wav):
        sliced_wav = wav[start_frame:end_frame]
    else:
        sliced_wav = wav[start_frame:]
    
    return sliced_wav, sr

def get_audio_wav_and_sample_rate(audio_filepath, sr=16000):
    ''' Get the sample rate of an audio file '''
    
    y, sr = librosa.load(audio_filepath, sr=sr)
    
    return y, sr

def export_audio(y, sr, output_filepath):
    ''' Export audio to a file '''
    
    sf.write(output_filepath, y, sr)
    
def hear_one_audio_from_filepath(audio_filepath):
    
    return Audio(audio_filepath)

def hear_one_audio_from_wav(wav, sr):

    return Audio(wav, rate=sr)

def resample_audio(
    audio_wav: np.ndarray, original_sr: int, desired_sr: int
) -> tuple[np.ndarray, int]:
    """
    Using scipy, converts the audio to the desired samplerate

    Inputs:
        audio_wav (np.ndarray): waveform as a numpy array
        original_sr (int): original sample rate to change
        desired_sr (int): The desired sample rate to resample to

    Returns:
        y_desired (np.ndarray): Output resampled and rechanneled audio wav
        desired_sr (int): Pass through desired SR
    """
    number_of_samples = int(len(audio_wav) * desired_sr / original_sr)
    y_desired = scipy.signal.resample(audio_wav, number_of_samples)

    return y_desired, desired_sr

def rechannel_audio(audio_wav: np.ndarray) -> np.ndarray:
    """
    Using numpy, converts an audio array to mono

    Inputs:
        audio_wav (np.ndarray): waveform as a numpy array

    Returns:
        (np.ndarray): rechanneled audio wav
    """
    # Stereo
    if audio_wav.ndim == 2:
        if audio_wav.shape[0] < audio_wav.shape[1]:
            # Probably (n_channels, n_samples) — good
            print("Stereo input: (channels, samples)")
        else:
            # Probably (samples, channels) — need to transpose
            print("Stereo input in (samples, channels) format. Transposing...")
            audio_wav = audio_wav.T

        return np.mean(audio_wav, axis=0)
    # Mono
    elif audio_wav.ndim == 1:
        return audio_wav
    # Too many .ndim in array
    else:
        return "Error has been occured"
    
def low_pass_filter(
    wav: np.ndarray,
    sr: int,
    low_pass_threshold: int = 4000
):
    '''
    Put the audio through a low-pass filter of specified threshold 
    '''
    nyquist = sr / 2
    norm_cutoff = low_pass_threshold / nyquist

    b, a = scipy.signal.butter(N=4, Wn=norm_cutoff, btype='low', analog=False)

    # Apply the filter
    filtered = scipy.signal.filtfilt(b, a, wav)

    # Optionally scale the filtered output
    return filtered, sr

def high_pass_filter(
    wav: np.ndarray,
    sr: int,
    high_pass_threshold: int = 300
):
    '''
    Put the audio through a high-pass filter of specified threshold 
    '''
    nyquist = sr / 2
    norm_cutoff = high_pass_threshold / nyquist 

    # Design Butterworth high-pass filter
    b, a = scipy.signal.butter(N=4, Wn=norm_cutoff, btype='high', analog=False)

    # Apply zero-phase filtering
    filtered = scipy.signal.filtfilt(b, a, wav)

    # Optionally scale the filtered output
    return filtered, sr

def repeat_audio(
    wav: np.ndarray,
    num_to_repeat_by: int = 5
):
    '''
    Repeat audio numpy array by specified number of times
    '''
    
    return np.tile(wav, num_to_repeat_by)

def rechannel_mono_to_stereo(
    mono_wav: np.ndarray
):
    '''
    Mono to Stereo 
    '''
    if not isinstance(mono_wav, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(mono_wav)}")

    if mono_wav.ndim == 2:
        if mono_wav.shape[1] == 2:
            return mono_wav  # Already stereo, return as is
        elif mono_wav.shape[1] == 1:
            mono_wav = mono_wav.flatten()  # Convert (N, 1) to (N,)
        else:
            raise ValueError(f"Unsupported channel count: {mono_wav.shape[1]}")

    if mono_wav.ndim != 1:
        raise ValueError(f"Expected 1D array for mono, got {mono_wav.ndim}D")

    # Stack to Stereo
    stereo_wav = np.column_stack((mono_wav, mono_wav))
    
    return stereo_wav
    
    