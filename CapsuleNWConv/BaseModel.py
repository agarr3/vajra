import random

import torch
from mnist import MNIST
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
import numpy as np
import os

class CapsuleNW(torch.nn.Module):
    def __init__(self):
        super(CapsuleNW, self).__init__()
        self.USE_CUDA = True if torch.cuda.is_available() else False
        self.decoder_layer = torch.nn.Sequential(torch.nn.Linear(28*28, 32),
                                                 torch.nn.Sigmoid(),
                                                 torch.nn.Linear(32, 10),
                                                 torch.nn.Softmax())
    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    def forward(self, inputs, labels=None):
        outputs = self.decoder_layer(inputs)

        if labels is not None:
            loss = self.loss_fn(outputs, labels)

        return outputs, loss


class CapsuleNWConfiguration:
    EPOCHS = 50
    TRAIN_BATCH_SIZE = 500
    TEST_BATCH_SIZE = 500
    LEARNING_RATE = 1e-05

    DETERMINISTIC = True


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
        #data = data.view(self.channels, self.height, self.width)

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