import torch
from mnist import MNIST
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from torch.utils.data.sampler import SequentialSampler


class CapsuleNetworkConfig:
    input_height = 28
    input_width = 28

    num_features_layer1 = 256
    kernel_size_layer1 = 9
    stride_layer1 = 1
    padding_layer1 = 0

    num_features_primary_cap = 32
    primary_cap_dim = 8
    kernel_size_layer2 = 9
    stride_layer2 = 2
    padding_layer2 = 0

    secondary_capsule_dim = 16
    secondary_capsule_number = 10

    routing_iterations = 5

    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5

    alpha = 0.0005

    TRAIN_BATCH_SIZE = 100
    EPOCHS = 20


class CapsuleNetwork(torch.nn.Module):

    def __init__(self, in_channels, configuration=None):

        super(CapsuleNetwork, self).__init__()
        if configuration:
            self.configuration = configuration
        else:
            self.configuration = CapsuleNetworkConfig()

        self.conv_layer_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=self.configuration.num_features_layer1, kernel_size=self.configuration.kernel_size_layer1, stride=self.configuration.stride_layer1, padding=self.configuration.padding_layer1)

        self.conv_stack = torch.nn.ModuleList([torch.nn.Conv2d(in_channels=self.configuration.num_features_layer1,
                         out_channels=self.configuration.num_features_primary_cap,
                         kernel_size=self.configuration.kernel_size_layer2, stride=self.configuration.stride_layer2, padding=self.configuration.padding_layer2)
                         for _ in range(self.configuration.primary_cap_dim)])

        self.configuration.first_filter_height = (int(self.configuration.input_height - self.configuration.kernel_size_layer1 + 2 * self.configuration.padding_layer1)/ self.configuration.stride_layer1 ) + 1
        self.configuration.first_filter_width = (int(self.configuration.input_width - self.configuration.kernel_size_layer1 + 2 * self.configuration.padding_layer1) / self.configuration.stride_layer1) + 1

        self.configuration.second_filter_height = int((int(self.configuration.first_filter_height - self.configuration.kernel_size_layer2 + 2 * self.configuration.padding_layer2) / self.configuration.stride_layer2) + 1)
        self.configuration.second_filter_width = int((int(self.configuration.first_filter_width - self.configuration.kernel_size_layer2 + 2 * self.configuration.padding_layer2) / self.configuration.stride_layer2) + 1)

        self.configuration.primary_capsule_num = int(self.configuration.second_filter_height * self.configuration.second_filter_width * self.configuration.num_features_primary_cap)

        self.prediction_w = torch.nn.parameter.Parameter(torch.randn([self.configuration.primary_capsule_num, self.configuration.secondary_capsule_number, self.configuration.secondary_capsule_dim, self.configuration.primary_cap_dim]))

        self.decoder_layer = torch.nn.Sequential(torch.nn.Linear(self.configuration.secondary_capsule_number * self.configuration.secondary_capsule_dim, 512),
                                      torch.nn.ReLU(inplace=True),
                                      torch.nn.Linear(512, 1024),
                                      torch.nn.ReLU(inplace=True),
                                      torch.nn.Linear(1024, self.configuration.input_height * self.configuration.input_width),
                                      torch.nn.Sigmoid())

    def forward(self, input, labels):
        conv_out_1 = torch.relu(self.conv_layer_1(input))
        conv_out_2 = [conv(conv_out_1) for conv in self.conv_stack]

        conv_out_2 = torch.stack(conv_out_2, dim =1)
        conv_out_2 = conv_out_2.view(conv_out_2.size(0),-1,self.configuration.primary_cap_dim)

        primary_caps_vec = self.squash(conv_out_2, dim=-1)

        primary_caps_vec_tiled = torch.stack([primary_caps_vec] * self.configuration.secondary_capsule_number, dim=2)
        primary_caps_vec_tiled = primary_caps_vec_tiled.unsqueeze(-1)

        batch_size = input.size(0)
        W = torch.stack([self.prediction_w] * batch_size, dim=0)
        u_hat = torch.matmul(W, primary_caps_vec_tiled)

        b_ij = torch.autograd.Variable(torch.zeros(batch_size, self.configuration.primary_capsule_num, self.configuration.secondary_capsule_number, 1, 1))

        for i in range(self.configuration.routing_iterations):
            c_ij = torch.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat)
            s_j = s_j.sum(dim=1, keepdim=True)
            v_j = self.squash(s_j, dim=-2)
            v_j_tiled = torch.cat([v_j] * self.configuration.primary_capsule_num, dim=1)
            a_ij = torch.matmul(v_j_tiled.transpose(3, 4), u_hat)
            b_ij = b_ij + a_ij

        v_k, _ = self.safeNorm(v_j, dim=-2, keepdim=True)
        prediction = torch.squeeze(v_k)

        predictionList = []
        for predictionForSingleInput in prediction:
            predictionList.append(torch.argmax(predictionForSingleInput))

        margin_loss = 0
        if labels is not None:
            T = labels
            T = torch.eye(10).index_select(dim=0, index=T)
            correct_loss = torch.relu(self.configuration.m_plus - v_k).pow(2).view(batch_size, -1)
            incorrect_loss = torch.relu(v_k - self.configuration.m_minus).pow(2).view(batch_size, -1)
            margin_loss = T * correct_loss + self.configuration.lambda_ * (1 - T) * incorrect_loss
            margin_loss = margin_loss.sum(dim=1).mean()


        v_j = v_j.squeeze(1)
        classes = torch.sqrt((v_j ** 2).sum(dim=2, keepdim=False))
        _, max_length_indices = classes.max(dim=1)
        masked = torch.autograd.Variable(torch.eye(10))
        masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)
        decoder_input = v_j * masked[:, :, None, None]

        decoder_input = decoder_input.view(batch_size, -1)
        decoder_output = self.decoder_layer(decoder_input)

        if labels is None:
            self.configuration.alpha = 1

        mse = torch.nn.MSELoss()
        reconstruction_loss = mse(decoder_output, input.view(batch_size, -1))
        final_loss = margin_loss + (self.configuration.alpha * reconstruction_loss)

        return torch.unsqueeze(max_length_indices, dim=-1), final_loss

    def predict(self, inputs):
        with torch.no_grad():
            predictions, loss = self.forward(inputs)
            return predictions



    def safeNorm(self, tensor, dim, epsilon=1e-7, keepdim=True):
        squared_norm = tensor.pow(2).sum(dim=dim, keepdim=keepdim)
        safe_norm = torch.sqrt(squared_norm + epsilon)
        return safe_norm, squared_norm

    def squash(self, tensor, dim, keepdim=True):
        safe_norm, squared_norm = self.safeNorm(tensor, dim=dim, keepdim=keepdim)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = tensor / safe_norm
        return squash_factor * unit_vector

class CustomDataset(Dataset):

    def __init__(self, dataList, labelList, channels, height, width):
        self.data = dataList
        self.labels = labelList
        self.height, self.width, self.channels = height, width, channels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        data = torch.tensor(data, dtype=torch.float)
        data = data.view(self.channels, self.height, self.width)

        label = torch.tensor(self.labels[index])

        return data, label

class CapsuleNWDriver:

    def __init__(self, mode, configuration= None):
        if configuration:
            self.configuration = configuration
        else:
            self.configuration = CapsuleNetworkConfig()

    def run_training_epoch(self, epoch, model, train_data_loader, test_data_loader, optimizer):
        model.train()
        model.zero_grad()

        for step, batch in torch.hub.tqdm(enumerate(train_data_loader),desc="running training for epoch {}".format(epoch)):
            data, label = batch

            inputs = {
                "input": data,
                "labels": label
            }

            predictions, loss = model(**inputs)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            print('Epoch: {}, step: {},  Loss:  {}'.format(epoch, step, loss.item()))

    def train(self, train_data_set, test_data_set):
        train_data_loader = DataLoader(train_data_set,
                                             batch_size=self.configuration.TRAIN_BATCH_SIZE,
                                             sampler=SequentialSampler(train_data_set), drop_last=False)

        test_data_loader = DataLoader(test_data_set,
                                       batch_size=self.configuration.TRAIN_BATCH_SIZE,
                                       sampler=SequentialSampler(test_data_set), drop_last=False)

        model = CapsuleNetwork(in_channels=1)
        optimizer = torch.optim.Adam(model.parameters())

        savedEpoch = 0
        for epoch in torch.hub.tqdm(range(savedEpoch, self.configuration.EPOCHS), desc="running outer train loop"):
            self.run_training_epoch(epoch, model, train_data_loader, test_data_loader, optimizer)


mndata = MNIST('/Users/ragarwal/PycharmProjects/vajra/mnist-data')
train_images, train_labels = mndata.load_training()
val_images, val_labels = mndata.load_testing()

train_dataset = CustomDataset(train_images, train_labels, 1, 28, 28)
val_dataset = CustomDataset(val_images, val_labels, 1, 28, 28)

driver = CapsuleNWDriver(mode = "train")
driver.train(train_dataset, val_dataset)

print("Hi")