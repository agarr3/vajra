from FeatureExtraction import FEATURES, ExtractorFactory, WavePlotLibrosa

featureExtractor = ExtractorFactory.getExtractor(feature=FEATURES.WAVE_PLOT)
featureExtractor.plotSingleAudioFile(audio_file_path="/Users/ragarwal/Downloads/intro25-sonicide.wav")

#%%
wv= WavePlotLibrosa()
wv.plotSingleAudioFile(audio_file_path="/Users/ragarwal/Downloads/intro25-sonicide.wav")


#%%


