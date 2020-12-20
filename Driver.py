import librosa
import soundfile as sf

from FeatureExtraction.FeatureExtraction import ExtractorFactory, FEATURES

audio_file_path="/Users/ragarwal/PycharmProjects/vajra/data/intro25-sonicide.wav"

# featureExtractor = ExtractorFactory.getExtractor(feature=FEATURES.WAVE_PLOT)
# featureExtractor.plotSingleAudioFile(audio_file_path=audio_file_path)

featureExtractor = ExtractorFactory.getExtractor(feature=FEATURES.LOG_MEL_SPECTOGRAM)
featureExtractor.plotSingleAudioFile(audio_file_path=audio_file_path)
melSpect, sr = featureExtractor.convertSingleAudioFile(audio_file_path=audio_file_path)
audio = librosa.feature.inverse.mel_to_audio(melSpect)
sf.write('stereo_file.wav', audio, sr)
melSpect1, sr =featureExtractor.plotSingleAudioFile(audio_file_path='stereo_file.wav')


# featureExtractor = ExtractorFactory.getExtractor(feature=FEATURES.CHROMA_GRAM)
# featureExtractor.plotSingleAudioFile(audio_file_path=audio_file_path)

