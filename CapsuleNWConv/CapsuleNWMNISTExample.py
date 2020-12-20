import numpy as np
import torch


def safeNorm(tensor, dim, epsilon=1e-7, keepdim=True):
    squared_norm = tensor.pow(2).sum(dim=dim, keepdim=keepdim)
    safe_norm = torch.sqrt(squared_norm + epsilon)
    return safe_norm, squared_norm

def squash(tensor, dim, keepdim=True):
    safe_norm, squared_norm = safeNorm(tensor, dim=dim, keepdim=keepdim)
    squash_factor = squared_norm / (1. + squared_norm)
    unit_vector = tensor / safe_norm
    return squash_factor * unit_vector

batch_size = 5
input_images = np.random.rand(batch_size, 1, 28, 28)
input_images = torch.from_numpy(input_images).float()

labels = np.random.randint(0, 10, batch_size)
print('image size - ', input_images.size())
print('labels - ', labels)

conv_layer = torch.nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1, padding=0)
print('Weight matrix size- ', conv_layer.weight.data.size())


conv_layer_out = torch.nn.functional.relu(conv_layer(input_images))
print('Output size - ', conv_layer_out.size())

primary_capsule_dim = primary_num_conv = 8
num_feature = 32
primary_num_capsule = 6 * 6 * num_feature

conv_stack = torch.nn.ModuleList([torch.nn.Conv2d(in_channels=256,
                         out_channels=num_feature,
                         kernel_size=9, stride=2, padding=0)
                         for _ in range(primary_num_conv)])
print('Weight matrix shape - ', conv_stack[0].weight.data.size())

# shapeTest = conv_stack[0](conv_layer_out)
# print("shape of first element of stacked conv - ", shapeTest.size() )

primary_capsule_out = [conv(conv_layer_out) for conv in conv_stack]
print('Output shape of every conv layer - ', primary_capsule_out[0].size())

primary_capsule_out = torch.stack(primary_capsule_out, dim=1)
print('Initial output shape - ', primary_capsule_out.size())

primary_capsule_out = primary_capsule_out.view(primary_capsule_out.size(0), -1, primary_capsule_dim)
print('Final output shape - ', primary_capsule_out.size())