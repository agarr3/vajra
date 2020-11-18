from FeatureExtraction import FEATURES, ExtractorFactory, WavePlotLibrosa

audio_file_path="/Users/ragarwal/Downloads/intro25-sonicide.wav"

featureExtractor = ExtractorFactory.getExtractor(feature=FEATURES.WAVE_PLOT)
featureExtractor.plotSingleAudioFile(audio_file_path=audio_file_path)

featureExtractor = ExtractorFactory.getExtractor(feature=FEATURES.LOG_MEL_SPECTOGRAM)
featureExtractor.plotSingleAudioFile(audio_file_path=audio_file_path)


featureExtractor = ExtractorFactory.getExtractor(feature=FEATURES.CHROMA_GRAM)
featureExtractor.plotSingleAudioFile(audio_file_path=audio_file_path)

