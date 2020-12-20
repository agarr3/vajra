import numpy as np
import torch

batch_size = 5
input_images = np.random.rand(batch_size, 1, 28, 28)
input_images = torch.from_numpy(input_images).float()

labels = np.random.randint(0, 10, batch_size)
print('image size - ', input_images.size())
print('labels - ', labels)

conv_layer = torch.nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1, padding=0)
print('Weight matrix size- ', conv_layer.weight.data.size())

conv_layer_out = torch.F.relu(conv_layer(input_images))
print('Output size - ', conv_layer_out.size())

