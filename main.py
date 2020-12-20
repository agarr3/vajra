from FeatureExtraction.FeatureExtraction import ExtractorFactory, FEATURES, WavePlotLibrosa

featureExtractor = ExtractorFactory.getExtractor(feature=FEATURES.WAVE_PLOT)
featureExtractor.plotSingleAudioFile(audio_file_path="/Users/ragarwal/Downloads/intro25-sonicide.wav")

#%%
wv= WavePlotLibrosa()
wv.plotSingleAudioFile(audio_file_path="/Users/ragarwal/Downloads/intro25-sonicide.wav")


#%%


import torch
a = torch.tensor([[2,3,4],[3,4,5]])
b = torch.tensor([[7,8,9],[1,2,3]])

c = torch.stack([a,b], dim=0)
print(c.shape)