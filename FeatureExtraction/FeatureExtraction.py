import os

from config import DEFAULT_FEATURE_EXTRACTION_LIBARY, ExtractionLibraries
from enum import Enum
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class FEATURES(Enum):
    LOG_MEL_SPECTOGRAM = 1
    WAVE_PLOT = 2
    CHROMA_GRAM = 3

class ExtractorFactory:

    @staticmethod
    def getExtractor(feature, library=None):
        if library is None:
            library = DEFAULT_FEATURE_EXTRACTION_LIBARY

        if(feature==FEATURES.LOG_MEL_SPECTOGRAM and library==ExtractionLibraries.LIBROSA):
            return LogMelSpectogramLibrosa()
        elif (feature == FEATURES.WAVE_PLOT and library == ExtractionLibraries.LIBROSA):
            return WavePlotLibrosa()
        elif (feature == FEATURES.CHROMA_GRAM and library == ExtractionLibraries.LIBROSA):
            return ChromagramLibrosa()

class BaseFeatureExtractor:

    def __init__(self):
        pass



class WavePlotLibrosa(BaseFeatureExtractor):
    def __init__(self):
        pass

    def plotSingleAudioFile(self, audio_file_path, save_file_path=None, trimSilentEdges=False):
        y, sr = librosa.load(audio_file_path)
        if trimSilentEdges:
            # trim silent edges
            plot, _ = librosa.effects.trim(y)
        else:
            plot = y
        librosa.display.waveplot(plot, sr=sr);
        if save_file_path:
            plt.savefig(save_file_path)
        else:
            plt.show()

    def convertSingleAudioFile(self, audio_file_path, save_file_path=None):
        y, sr = librosa.load(audio_file_path)
        if save_file_path:
            np.save(save_file_path, y)
        return y,sr

    def convertAllAudioInDir(self, source_dir_path, destination_dir_path):
        sampligRateList = []
        for root, directoris, files in os.walk(source_dir_path):
            for file in files:
                y, sr = self.convertSingleAudioFile(audio_file_path=os.path.join(root, file),
                                            save_file_path=os.path.join(destination_dir_path, file))
                sampligRateList.append([os.path.join(destination_dir_path, file), sr])

        sampligRateDF = pd.DataFrame(sampligRateList, columns=['file', 'sampling_rate'])



class LogMelSpectogramLibrosa(BaseFeatureExtractor):

    def __init__(self):
        pass

    def plotSingleAudioFile(self, audio_file_path, save_file_path=None, trimSilentEdges=False):
        y, sr = librosa.load(audio_file_path)
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        mel_spect = librosa.power_to_db(spect, ref=np.max)
        librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time');
        plt.title('Mel Spectrogram');
        plt.colorbar(format='%+2.0f dB')
        plt.show()

    def convertSingleAudioFile(self, audio_file_path, save_file_path=None):
        y, sr = librosa.load(audio_file_path)
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        #mel_spect = librosa.power_to_db(spect, ref=np.max)
        mel_spect = spect
        if not save_file_path:
            return mel_spect, sr
        else:
            np.save(save_file_path, mel_spect)

    def convertAllAudioInDir(self, source_dir_path, destination_dir_path):
        pass

class ChromagramLibrosa(BaseFeatureExtractor):

    def __init__(self):
        pass

    def plotSingleAudioFile(self, audio_file_path, save_file_path=None, trimSilentEdges=False):
        y, sr = librosa.load(audio_file_path)
        spect = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=1024)
        mel_spect = librosa.power_to_db(spect, ref=np.max)
        librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time');
        plt.title('Chromagram');
        plt.colorbar(format='%+2.0f dB')
        plt.show()

    def convertSingleAudioFile(self, audio_file_path, save_file_path=None):
        pass

    def convertAllAudioInDir(self, source_dir_path, destination_dir_path):
        pass