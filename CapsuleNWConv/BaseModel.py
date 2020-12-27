import random

import torch
from mnist import MNIST
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
import numpy as np
import os

class CapsuleNWConfiguration:
    EPOCHS = 50
    TRAIN_BATCH_SIZE = 500
    TEST_BATCH_SIZE = 500
    LEARNING_RATE = 1e-05

    DETERMINISTIC = True

    in_channels = 1
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

class CapsuleNW(torch.nn.Module):
    def __init__(self, configuration = None):
        super(CapsuleNW, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if configuration:
            self.configuration = configuration
        else:
            self.configuration = CapsuleNWConfiguration()

        self.conv_layer_1 = torch.nn.Conv2d(in_channels=self.configuration.in_channels,
                                            out_channels=self.configuration.num_features_layer1,
                                            kernel_size=self.configuration.kernel_size_layer1,
                                            stride=self.configuration.stride_layer1,
                                            padding=self.configuration.padding_layer1)

        self.conv_stack = torch.nn.ModuleList([torch.nn.Conv2d(in_channels=self.configuration.num_features_layer1,
                                                               out_channels=self.configuration.num_features_primary_cap,
                                                               kernel_size=self.configuration.kernel_size_layer2,
                                                               stride=self.configuration.stride_layer2,
                                                               padding=self.configuration.padding_layer2)
                                               for _ in range(self.configuration.primary_cap_dim)])

        self.configuration.first_filter_height = (int(
            self.configuration.input_height - self.configuration.kernel_size_layer1 + 2 * self.configuration.padding_layer1) / self.configuration.stride_layer1) + 1
        self.configuration.first_filter_width = (int(
            self.configuration.input_width - self.configuration.kernel_size_layer1 + 2 * self.configuration.padding_layer1) / self.configuration.stride_layer1) + 1

        self.configuration.second_filter_height = int((int(
            self.configuration.first_filter_height - self.configuration.kernel_size_layer2 + 2 * self.configuration.padding_layer2) / self.configuration.stride_layer2) + 1)
        self.configuration.second_filter_width = int((int(
            self.configuration.first_filter_width - self.configuration.kernel_size_layer2 + 2 * self.configuration.padding_layer2) / self.configuration.stride_layer2) + 1)

        self.configuration.primary_capsule_num = int(
            self.configuration.second_filter_height * self.configuration.second_filter_width * self.configuration.num_features_primary_cap * self.configuration.primary_cap_dim)

        self.decoder_layer = torch.nn.Sequential(torch.nn.Linear(self.configuration.primary_capsule_num, 512),
                                                 torch.nn.Sigmoid(),
                                                 torch.nn.Linear(512, 10),
                                                 torch.nn.Softmax())
    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    def forward(self, inputs, labels=None):

        batch_size = inputs.size(0)
        conv_out_1 = torch.relu(self.conv_layer_1(inputs))
        conv_out_2 = [conv(conv_out_1) for conv in self.conv_stack]

        conv_out_2 = torch.stack(conv_out_2, dim=1)
        conv_out_2 = conv_out_2.view(batch_size, -1)

        outputs = self.decoder_layer(conv_out_2)

        if labels is not None:
            loss = self.loss_fn(outputs, labels)

        return outputs, loss


class CapsuleNWDriver:
    def __init__(self, configuration = None):
        if configuration is not None:
            self.configuration = configuration
        else:
            self.configuration = CapsuleNWConfiguration()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.configuration.DETERMINISTIC:
            self.seed_everything()

        self.model = CapsuleNW()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.configuration.LEARNING_RATE)
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

    def test_epoch(self, epoch, test_data_loader):
        self.model.eval()

        test_loss = 0.0
        correct_predictions = 0

        with torch.no_grad():
            for step, batch in enumerate(test_data_loader):
                data, labels = batch
                data = data.to(self.device)
                labels = labels.to(self.device)

                inputs = {
                    "inputs": data,
                    "labels": labels
                }

                outputs, loss = self.model(**inputs)
                test_loss = test_loss + loss.item()

                _, predictions = outputs.max(dim=1)
                _, actuals = labels.max(dim=1)
                correct_predictions = correct_predictions + (predictions == actuals).sum()

                print("for epoch {}, minibatch {}, the TEST loss is {}".format(epoch, step, loss.item()))

        accuracy = correct_predictions.data.item() / (len(test_data_loader)*test_data_loader.batch_size) * 100
        print("for epoch {}, the TOTAL TEST loss is {}, TEST accuracy is {}".format(epoch, test_loss, accuracy))
        return test_loss, accuracy

    def train_epoch(self, epoch, train_data_loader):
        self.model.train()
        self.model.zero_grad()

        train_loss = 0.0
        correct_predictions = 0

        for step, batch in enumerate(train_data_loader):
            data, labels = batch
            data = data.to(self.device)
            labels = labels.to(self.device)

            inputs = {
                "inputs": data,
                "labels": labels
            }

            outputs, loss = self.model(**inputs)
            train_loss = train_loss + loss.item()
            loss.backward()
            self.optimizer.step()
            self.model.zero_grad()

            _, predictions = outputs.max(dim=1)
            _, actuals = labels.max(dim=1)
            correct_predictions = correct_predictions + (predictions == actuals).sum()

            print("for epoch {}, minibatch {}, the TRAIN loss is {}".format(epoch, step, loss.item()))

        accuracy = correct_predictions.data.item()/(len(train_data_loader)*train_data_loader.batch_size) * 100

        print("for epoch {}, the TOTAL TRAIN loss is {}, accuracy is {}".format(epoch, train_loss, accuracy))
        return train_loss, accuracy

    def seed_everything(self, seed=1234):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True

    def train(self, train_data_set, test_data_set):

        train_data_loader = DataLoader(train_data_set, batch_size=self.configuration.TRAIN_BATCH_SIZE,
                                       sampler=SequentialSampler(train_data_set))
        test_data_loader = DataLoader(test_data_set, batch_size=self.configuration.TEST_BATCH_SIZE,
                                      sampler=SequentialSampler(test_data_set))

        for epoch in range(0, self.configuration.EPOCHS):
            train_loss, train_accuracy = self.train_epoch(epoch, train_data_loader)
            test_loss, test_accuracy  = self.test_epoch(epoch, test_data_loader)
            self.train_losses.append(train_loss)
            self.test_losses.append((test_loss))
            self.train_accuracies.append(train_accuracy)
            self.test_accuracies.append(test_accuracy)





class CustomDatasetMnist(Dataset):

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

        label = self.labels[index].float()
        return data, label

mndata = MNIST('.././mnist-data')
train_images, train_labels = mndata.load_training()
val_images, val_labels = mndata.load_testing()

train_labels = torch.nn.functional.one_hot(torch.tensor(train_labels))
val_labels = torch.nn.functional.one_hot(torch.tensor(val_labels))

train_dataset = CustomDatasetMnist(train_images, train_labels, 1, 28, 28)
val_dataset = CustomDatasetMnist(val_images, val_labels, 1, 28, 28)

#driver = CapsuleNWDriver(mode = "train", modelPath='/Users/ragarwal/PycharmProjects/vajra/models/mnist')
driver = CapsuleNWDriver()
driver.train(train_dataset, val_dataset)
print("train loss - {}".format(driver.train_losses))
print("train accuracy - {}".format(driver.train_accuracies))
print("test loss - {}".format(driver.test_losses))
print("test accuracy - {}".format(driver.test_accuracies))