import os
import shutil

import torch
from mnist import MNIST
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from torch.utils.data.sampler import SequentialSampler
import numpy as np
from torchvision.utils import save_image

from pytorchtools import ModelCheckPointer, EarlyStoppingAndCheckPointer

USE_CUDA = True if torch.cuda.is_available() else False


class CapsuleNetworkConfig:
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

    secondary_capsule_dim = 16
    secondary_capsule_number = 10

    routing_iterations = 3

    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5

    alpha = 0.0005

    TRAIN_BATCH_SIZE = 500
    EPOCHS = 2
    PATIENCE = 5
    LEARNING_RATE = 1e-01

    base_dir = ".././models/mnist"


class CapsuleNetwork(torch.nn.Module):

    def __init__(self, configuration=None):

        super(CapsuleNetwork, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if configuration:
            self.configuration = configuration
        else:
            self.configuration = CapsuleNetworkConfig()

        self.conv_layer_1 = torch.nn.Conv2d(in_channels=self.configuration.in_channels, out_channels=self.configuration.num_features_layer1, kernel_size=self.configuration.kernel_size_layer1, stride=self.configuration.stride_layer1, padding=self.configuration.padding_layer1)

        self.conv_stack = torch.nn.ModuleList([torch.nn.Conv2d(in_channels=self.configuration.num_features_layer1,
                         out_channels=self.configuration.num_features_primary_cap,
                         kernel_size=self.configuration.kernel_size_layer2, stride=self.configuration.stride_layer2, padding=self.configuration.padding_layer2)
                         for _ in range(self.configuration.primary_cap_dim)])

        self.configuration.first_filter_height = (int(self.configuration.input_height - self.configuration.kernel_size_layer1 + 2 * self.configuration.padding_layer1)/ self.configuration.stride_layer1 ) + 1
        self.configuration.first_filter_width = (int(self.configuration.input_width - self.configuration.kernel_size_layer1 + 2 * self.configuration.padding_layer1) / self.configuration.stride_layer1) + 1

        self.configuration.second_filter_height = int((int(self.configuration.first_filter_height - self.configuration.kernel_size_layer2 + 2 * self.configuration.padding_layer2) / self.configuration.stride_layer2) + 1)
        self.configuration.second_filter_width = int((int(self.configuration.first_filter_width - self.configuration.kernel_size_layer2 + 2 * self.configuration.padding_layer2) / self.configuration.stride_layer2) + 1)

        self.configuration.primary_capsule_num = int(self.configuration.second_filter_height * self.configuration.second_filter_width * self.configuration.num_features_primary_cap)

        self.prediction_w = torch.nn.parameter.Parameter(torch.ones([self.configuration.primary_capsule_num, self.configuration.secondary_capsule_number, self.configuration.secondary_capsule_dim, self.configuration.primary_cap_dim]).to(self.device))

        self.decoder_layer = torch.nn.Sequential(torch.nn.Linear(self.configuration.secondary_capsule_number * self.configuration.secondary_capsule_dim, 512),
                                      torch.nn.ReLU(inplace=True),
                                      torch.nn.Linear(512, 1024),
                                      torch.nn.ReLU(inplace=True),
                                      torch.nn.Linear(1024, self.configuration.input_height * self.configuration.input_width),
                                      torch.nn.Sigmoid())

    def forward(self, input, labels = None):

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

        if USE_CUDA:
            b_ij = b_ij.cuda()

        for i in range(self.configuration.routing_iterations):
            c_ij = torch.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat)
            s_j = s_j.sum(dim=1, keepdim=True)
            v_j = self.squash(s_j, dim=-2)
            v_j_tiled = torch.cat([v_j] * self.configuration.primary_capsule_num, dim=1)
            a_ij = torch.matmul(v_j_tiled.transpose(3, 4), u_hat)
            b_ij = b_ij + a_ij

        v_k, _ = self.safeNorm(v_j, dim=-2, keepdim=True)
        # prediction = torch.squeeze(v_k)
        #
        # predictionList = []
        # for predictionForSingleInput in prediction:
        #     predictionList.append(torch.argmax(predictionForSingleInput))

        margin_loss = 0
        if labels is not None:
            T = torch.eye(10)
            if USE_CUDA:
                T = T.cuda()
            T = T.index_select(dim=0, index=labels)
            correct_loss = torch.relu(self.configuration.m_plus - v_k).pow(2).view(batch_size, -1)
            incorrect_loss = torch.relu(v_k - self.configuration.m_minus).pow(2).view(batch_size, -1)
            margin_loss = T * correct_loss + self.configuration.lambda_ * (1 - T) * incorrect_loss
            margin_loss = margin_loss.sum(dim=1).mean()


        v_j = v_j.squeeze(1)
        classes = torch.sqrt((v_j ** 2).sum(dim=2, keepdim=False))
        _, max_length_indices = classes.max(dim=1)
        masked = torch.autograd.Variable(torch.eye(10))
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)
        decoder_input = v_j * masked[:, :, None, None]

        decoder_input = decoder_input.view(batch_size, -1)
        decoder_output = self.decoder_layer(decoder_input)

        if labels is None:
            self.configuration.alpha = 1

        mse = torch.nn.MSELoss()
        reconstruction_loss = mse(decoder_output, input.view(batch_size, -1))
        final_loss = margin_loss + (self.configuration.alpha * reconstruction_loss)

        return torch.squeeze(max_length_indices, dim=-1), final_loss, decoder_output

    def predict(self, inputs):
        with torch.no_grad():
            input = inputs['input']
            if "labels" in inputs.keys():
                label = inputs['labels']
            else:
                label = None
            predictions, loss, decoder_output = self.forward(input, label)
            return predictions, loss, decoder_output



    def safeNorm(self, tensor, dim, epsilon=1e-7, keepdim=True):
        squared_norm = tensor.pow(2).sum(dim=dim, keepdim=keepdim)
        safe_norm = torch.sqrt(squared_norm + epsilon)
        return safe_norm, squared_norm

    def squash(self, tensor, dim, keepdim=True):
        safe_norm, squared_norm = self.safeNorm(tensor, dim=dim, keepdim=keepdim)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = tensor / safe_norm
        return squash_factor * unit_vector

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

        label = torch.tensor(self.labels[index])

        return data, label, True

class CustomDatasetMelTest(Dataset):

    def __init__(self, dataList, labelList):
        self.data = dataList
        self.labels = labelList

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fileName = self.data[index]
        # featureExtractor = ExtractorFactory.getExtractor(feature=FEATURES.LOG_MEL_SPECTOGRAM)
        # mel_spect, sr = featureExtractor.convertSingleAudioFile(audio_file_path=fileName)

        mel_spect = np.load(fileName)
        data = torch.tensor(mel_spect[:24,:80], dtype=torch.float)
        data = torch.unsqueeze(data, dim =0)

        label = torch.tensor(self.labels[index])

        return data, label, False

class CapsuleNWDriver:

    def __init__(self, mode,  configuration= None, modelPath=None):


        if configuration:
            self.configuration = configuration
        else:
            self.configuration = CapsuleNetworkConfig()

        self.modelCheckpointer = ModelCheckPointer()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.modelPath = modelPath
        self.training_loss = []
        self.test_loss = []
        self.test_accuracy = []
        self.train_acccuracy = []

        if mode == "train":
            self.model = CapsuleNetwork(configuration=self.configuration)
            optimizer_grouped_parameters = [{
                "params": [p for n, p in self.model.named_parameters()],
            }]
            self.optimizer = torch.optim.SGD(params=optimizer_grouped_parameters, lr=self.configuration.LEARNING_RATE,
                                             momentum=0.9)
            if USE_CUDA:
                self.model = self.model.cuda()
            self.savedEpoch = 0
            if self.modelPath is not None:
                self.savedEpoch = self.modelCheckpointer.load_checkpoint(self.modelPath, self.model, self.device, optimizer=self.optimizer)
            else:
                pass
        elif mode == "eval":
            if self.modelPath is not None:
                self.modelCheckpointer.load_checkpoint(self.modelPath, self.model, self.device)
            else:
                self.modelCheckpointer.loadBestModel(self.configuration.base_dir, self.model, self.device)

            self.model.eval()


    def predict(self, input):
        self.model.eval()
        predictions, loss, decoder_output = self.model.predict(input)
        return predictions.detach(), loss.detach(), decoder_output.detach()

    def run_evaluation(self, epoch, test_data_loader, save_reconstruction=False):

        total_test_loss = 0.0
        total_test_accurate_matches = 0.0
        global_step = 0

        if save_reconstruction:
            if self.modelPath is not None:
                dir = self.modelPath
            else:
                dir = self.configuration.base_dir
            target_dir = os.path.join(dir, "reconstruction_epoch_{}".format(epoch))
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
                os.makedirs(target_dir)
            else:
                os.makedirs(target_dir)

        for step, batch in torch.hub.tqdm(enumerate(test_data_loader),desc="running evaluation for epoch {}".format(epoch)):
            data, label, labelFlag = batch
            if USE_CUDA:
                data, label = data.cuda(), label.cuda()

            if all(labelFlag):
                inputs = {
                    "input": data,
                    "labels": label
                }
            else:
                inputs = {
                    "input": data
                }

            predictions, loss, decoder_output = self.predict(inputs)
            total_test_loss = total_test_loss + loss.item()

            if all(labelFlag):
                total_test_accurate_matches = total_test_accurate_matches + (label == predictions).sum()

            print('Epoch: {}, step: {},  Loss:  {}'.format(epoch, step, loss.item()))
            global_step = global_step + 1
            local_step=0
            if save_reconstruction:
                for image, label in zip(decoder_output, label):
                    local_step = local_step + 1
                    image = image.view(self.configuration.input_height, self.configuration.input_width)
                    img_name = os.path.join(target_dir, "global_step_{}_local_step_{}_label_{}.png".format(global_step,local_step, label))
                    save_image(image, str(img_name))

        accuracy = total_test_accurate_matches / len(test_data_loader)
        return total_test_loss,accuracy.cpu().numpy() , all(labelFlag)


    def run_training_epoch(self, epoch, train_data_loader):
        self.model.train()
        self.model.zero_grad()

        total_train_loss = 0.0
        total_train_accurate_matches = 0.0

        for step, batch in torch.hub.tqdm(enumerate(train_data_loader),desc="running training for epoch {}".format(epoch)):
            data, label, labelFlag = batch

            if USE_CUDA:
                data, label = data.cuda(), label.cuda()

            if all(labelFlag):
                inputs = {
                    "input": data,
                    "labels": label
                }
            else:
                inputs = {
                    "input": data
                }

            predictions, loss, decoder_output = self.model(**inputs)
            total_train_loss = total_train_loss + loss.item()
            loss.backward()
            self.optimizer.step()
            self.model.zero_grad()

            if all(labelFlag):
                total_train_accurate_matches = total_train_accurate_matches + (label == predictions).sum()

            print('Epoch: {}, step: {},  Loss:  {}'.format(epoch, step, loss.item()))

        accuracy = total_train_accurate_matches/len(train_data_loader)
        return total_train_loss, accuracy.detach().cpu().numpy(), all(labelFlag)

    def train(self, train_data_set, test_data_set):
        train_data_loader = DataLoader(train_data_set,
                                             batch_size=self.configuration.TRAIN_BATCH_SIZE,
                                             sampler=SequentialSampler(train_data_set), drop_last=False)

        test_data_loader = DataLoader(test_data_set,
                                       batch_size=self.configuration.TRAIN_BATCH_SIZE,
                                       sampler=SequentialSampler(test_data_set), drop_last=False)

        if self.modelPath is not None:
            dir = self.modelPath
        else:
            dir = self.configuration.base_dir

        early_stopping = EarlyStoppingAndCheckPointer(patience=self.configuration.PATIENCE, verbose=True,
                                                      basedir=dir)

        for epoch in torch.hub.tqdm(range(self.savedEpoch, self.configuration.EPOCHS), desc="running outer train loop"):
            train_loss, train_accuracy, train_accuracyFlag = self.run_training_epoch(epoch, train_data_loader)
            test_loss, test_accuracy, test_accuracyFlag = self.run_evaluation(epoch, test_data_loader, True)
            early_stopping(test_loss, self.model, self.optimizer, epoch)
            self.training_loss.append(train_loss)
            self.train_acccuracy.append(train_accuracy)
            self.test_loss.append(test_loss)
            self.test_accuracy.append(test_accuracy)
            print('End of Epoch: {}, train accuracy: {},  train Loss:  {}'.format(epoch, self.train_acccuracy, self.training_loss))
            print('End of Epoch: {}, test accuracy: {},  test Loss:  {}'.format(epoch, self.test_accuracy, self.test_loss))
            if early_stopping.early_stop:
                print("Early stopping")
                #break


dataset = "MNIST"
#dataset = "MELTEST"

if dataset == "MNIST":
    mndata = MNIST('.././mnist-data')
    train_images, train_labels = mndata.load_training()
    val_images, val_labels = mndata.load_testing()

    train_dataset = CustomDatasetMnist(train_images, train_labels, 1, 28, 28)
    val_dataset = CustomDatasetMnist(val_images, val_labels, 1, 28, 28)

    driver = CapsuleNWDriver(mode = "train", modelPath='/Users/ragarwal/PycharmProjects/vajra/models/mnist')
    #driver = CapsuleNWDriver(mode="train")
    driver.train(train_dataset, val_dataset)
    print("test loss - {}".format(driver.test_loss))
    print("test accuracy - {}".format(driver.test_accuracy))
elif dataset == "MELTEST":
    train_data = [".././data/intro25-sonicide.npy"]*50
    labels = [1] * 50

    train_dataset = CustomDatasetMelTest(train_data, labels)

    configuration = CapsuleNetworkConfig()
    configuration.input_height = 24
    configuration.input_width = 80

    driver = CapsuleNWDriver(mode="train", configuration= configuration)
    driver.train(train_dataset, train_dataset)



print("Hi")