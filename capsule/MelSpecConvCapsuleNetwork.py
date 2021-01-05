import os

import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler

from capsuleclassification import DenseCapsule
from feature_extraction_capsule_layers import FeatureExtractionConvolution
import numpy as np

class MNISTCapsuleClassifierConfiguration:
    epochs = 2
    batch_size = 2
    lr = 0.001
    lr_decay = 0.9
    lam_recon = 0.0005 * 784
    routings = 3
    shift_pixels = 2
    data_dir = './data'
    download = True
    save_dir = './result/mel'

    input_size=[1, 24, 776]
    classes = 10
    routings = 3

    first_layer_kernel_size = 9
    first_layer_stride = 1
    first_layer_padding = 0
    first_layer_num_filters = 256

    second_layer_kernel_size = 9
    second_layer_stride = 2
    second_layer_padding = 0
    second_layer_num_filters = 256

    feature_dimension = 8
    output_dimension = 16

    supervised = False

class _CapsuleNet(nn.Module):
    """
    A Capsule Network on MNIST.
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
    """

    def __init__(self, input_size, classes, routings, configuration=None):
        super(_CapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings
        self.USE_CUDA = torch.cuda.is_available()

        if configuration:
            self.config = configuration
        else:
            self.config = MNISTCapsuleClassifierConfiguration()

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(input_size[0], self.config.first_layer_num_filters,
                               kernel_size=self.config.first_layer_kernel_size, stride=self.config.first_layer_stride,
                               padding=self.config.first_layer_padding)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = FeatureExtractionConvolution(self.config.first_layer_num_filters,
                                                        self.config.second_layer_num_filters,
                                                        self.config.feature_dimension,
                                                        kernel_size=self.config.second_layer_kernel_size,
                                                        stride=self.config.second_layer_stride,
                                                        padding=self.config.second_layer_padding)

        first_layer_op_height = (int(
            input_size[1] - self.config.first_layer_kernel_size + 2 * self.config.first_layer_padding) / self.config.first_layer_stride) + 1

        first_layer_op_width = (int(
            input_size[2] - self.config.first_layer_kernel_size + 2 * self.config.first_layer_padding) / self.config.first_layer_stride) + 1

        seconf_layer_op_height = int((int(
            first_layer_op_height - self.config.second_layer_kernel_size + 2 * self.config.second_layer_padding) / self.config.second_layer_stride) + 1)

        seconf_layer_op_width = int((int(
            first_layer_op_width - self.config.second_layer_kernel_size + 2 * self.config.second_layer_padding) / self.config.second_layer_stride) + 1)

        primary_capsule_num = int(
            seconf_layer_op_height * seconf_layer_op_width * self.config.second_layer_num_filters / self.config.feature_dimension)


        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(in_num_caps=primary_capsule_num, in_dim_caps=self.config.feature_dimension,
                                      out_num_caps=classes, out_dim_caps=self.config.output_dimension, routings=routings)

        # Decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(self.config.output_dimension * classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = self.relu(self.conv1(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        length = x.norm(dim=-1)
        if y is None:  # during testing, no label given. create one-hot coding using `length`
            index = length.max(dim=1)[1]
            if self.USE_CUDA:
                y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.).cuda())
            else:
                y = Variable(torch.zeros(length.size()).scatter_(1, index.view(-1, 1).cpu().data, 1.))
        if self.config.supervised:
            reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        else:
            reconstruction = self.decoder(x.view(x.size(0), -1))

        return length, reconstruction.view(-1, *self.input_size)


class MNISTCapsuleClassifier:
    def __init__(self, input_size=[1, 28, 28], classes=10, routings=3, configuration=None, model_path=None):

        if configuration is None:
            self.config = MNISTCapsuleClassifierConfiguration()
        else:
            self.config = configuration

        print(self.config)
        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)

        self.model = _CapsuleNet(input_size=input_size, classes=classes, routings=routings)
        self.use_cuda = torch.cuda.is_available()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.use_cuda:
            self.model.cuda()
        print(self.model)

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def caps_loss(self, y_true, y_pred, x, x_recon, lam_recon, supervised = True):
        """
        Capsule loss = Margin loss + lam_recon * reconstruction loss.
        :param y_true: true labels, one-hot coding, size=[batch, classes]
        :param y_pred: predicted labels by CapsNet, size=[batch, classes]
        :param x: input data, size=[batch, channels, width, height]
        :param x_recon: reconstructed data, size is same as `x`
        :param lam_recon: coefficient for reconstruction loss
        :return: Variable contains a scalar loss value.
        """
        L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
            0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
        L_margin = L.sum(dim=1).mean()

        L_recon = nn.MSELoss()(x_recon, x)
        if supervised:
            return L_margin + lam_recon * L_recon
        else:
            return L_recon

    def show_reconstruction(self, test_loader, n_images):
        import matplotlib.pyplot as plt
        from utils import combine_images
        from PIL import Image
        import numpy as np

        self.model.eval()
        for x, _ in test_loader:
            if self.use_cuda:
                x = Variable(x[:min(n_images, x.size(0))].cuda(), volatile=True)
            else:
                x = Variable(x[:min(n_images, x.size(0))], volatile=True)
            _, x_recon = self.model(x)
            data = np.concatenate([x.data.cpu(), x_recon.data.cpu()])
            img = combine_images(np.transpose(data, [0, 2, 3, 1]))
            image = img * 255
            Image.fromarray(image.astype(np.uint8)).save(self.config.save_dir + "/real_and_recon.png")
            print()
            print('Reconstructed images are saved to %s/real_and_recon.png' % config.save_dir)
            print('-' * 70)
            plt.imshow(plt.imread(config.save_dir + "/real_and_recon.png", ))
            plt.show()
            break

    def test(self,test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        for x, y in test_loader:
            y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)
            if self.use_cuda:
                x, y = Variable(x.cuda(), volatile=True), Variable(y.cuda())
            else:
                x, y = Variable(x, volatile=True), Variable(y)
            y_pred, x_recon = self.model(x)
            test_loss += self.caps_loss(y, y_pred, x, x_recon, self.config.lam_recon, self.config.supervised).data.item() * x.size(0)  # sum up batch loss
            y_pred = y_pred.data.max(1)[1]
            y_true = y.data.max(1)[1]
            correct += y_pred.eq(y_true).cpu().sum()

        test_loss /= len(test_loader.dataset)
        return test_loss, correct.item() / len(test_loader.dataset)

    def train(self, train_loader, test_loader):
        """
        Training a CapsuleNet
        :param model: the CapsuleNet model
        :param train_loader: torch.utils.data.DataLoader for training data
        :param test_loader: torch.utils.data.DataLoader for test data
        :param args: arguments
        :return: The trained model
        """
        print('Begin Training' + '-' * 70)
        from time import time

        t0 = time()
        optimizer = Adam(self.model.parameters(), lr=self.config.lr)
        lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=self.config.lr_decay)
        best_val_acc = 0.
        best_val_loss = np.inf
        for epoch in range(self.config.epochs):
            self.model.train()  # set to training mode
            lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
            ti = time()
            training_loss = 0.0
            for i, (x, y) in enumerate(train_loader):  # batch training
                y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)  # change to one-hot coding
                if self.use_cuda:
                    x, y = Variable(x.cuda()), Variable(y.cuda())  # convert input data to GPU Variable
                else:
                    x, y = Variable(x), Variable(y)

                optimizer.zero_grad()  # set gradients of optimizer to zero
                y_pred, x_recon = self.model(x, y)  # forward
                loss = self.caps_loss(y, y_pred, x, x_recon, self.config.lam_recon, self.config.supervised)  # compute loss
                loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
                training_loss += loss.data.item() * x.size(0)  # record the batch loss
                optimizer.step()  # update the trainable parameters with computed gradients
                print("epoch {}, minibatch {}, loss {}".format(epoch, i, loss.data.item()))

            # compute validation loss and acc
            val_loss, val_acc = self.test(test_loader)
            print("==> Epoch %02d: loss=%.5f, val_loss=%.5f, val_acc=%.4f, time=%ds"
                  % (epoch, training_loss / len(train_loader.dataset),
                     val_loss, val_acc, time() - ti))
            if self.config.supervised:
                if val_acc > best_val_acc:  # update best validation acc and save model
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), self.config.save_dir + '/epoch%d.pkl' % epoch)
                    print("best val_acc increased to %.4f" % best_val_acc)
            else:
                if val_loss < best_val_loss:  # update best validation acc and save model
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.config.save_dir + '/epoch%d.pkl' % epoch)
                    print("best val_loss decreased to %.4f" % best_val_loss)

        torch.save(self.model.state_dict(), self.config.save_dir + '/trained_model.pkl')
        print('Trained model saved to \'%s/trained_model.h5\'' % self.config.save_dir)
        print("Total time = %ds" % (time() - t0))
        print('End Training' + '-' * 70)

class CustomDatasetMelTest(Dataset):

    def __init__(self, dataList, labelList):
        self.data = dataList
        self.labels = labelList

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        import numpy as np
        fileName = self.data[index]
        # featureExtractor = ExtractorFactory.getExtractor(feature=FEATURES.LOG_MEL_SPECTOGRAM)
        # mel_spect, sr = featureExtractor.convertSingleAudioFile(audio_file_path=fileName)

        mel_spect = np.load(fileName)
        data = torch.tensor(mel_spect[:,:], dtype=torch.float)
        data = torch.unsqueeze(data, dim =0)

        label = torch.tensor(self.labels[index])

        return data, label

if __name__ == "__main__":
    def load_mnist(path='./data', download=False, batch_size=100, shift_pixels=2):
        """
        Construct dataloaders for training and test data. Data augmentation is also done here.
        :param path: file path of the dataset
        :param download: whether to download the original data
        :param batch_size: batch size
        :param shift_pixels: maximum number of pixels to shift in each direction
        :return: train_loader, test_loader
        """
        kwargs = {'num_workers': 1, 'pin_memory': True}

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=download,
                           transform=transforms.Compose([transforms.RandomCrop(size=28, padding=shift_pixels),
                                                         transforms.ToTensor()])),
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False, download=download,
                           transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)

        return train_loader, test_loader


    mode = 'train'
    data = "mel"
    config = MNISTCapsuleClassifierConfiguration()

    if data == "mel":
        train_data = [".././data/intro25-sonicide.npy"] * 500
        labels = [1] * 500

        train_dataset = CustomDatasetMelTest(train_data, labels)
        test_data_set = CustomDatasetMelTest(train_data, labels)

        train_loader = DataLoader(train_dataset,
                                       batch_size=config.batch_size,
                                       sampler=SequentialSampler(train_dataset), drop_last=False)

        test_loader = DataLoader(test_data_set,
                                      batch_size=config.batch_size,
                                      sampler=SequentialSampler(test_data_set), drop_last=False)
    elif data == "mnist":
        config.input_size = [1, 28, 28]
        config.save_dir = './result/mnist'
        config.batch_size=100
        config.supervised = True
        train_loader, test_loader = load_mnist(config.data_dir, download=True, batch_size=config.batch_size)

    if mode == 'train':
        classifier = MNISTCapsuleClassifier(input_size=config.input_size, classes=config.classes,routings=config.routings,configuration=config)
        classifier.train(train_loader,test_loader)
        classifier.show_reconstruction(test_loader, 50)
    elif mode == 'test':
        classifier = MNISTCapsuleClassifier(input_size=config.input_size, classes=config.classes,
                                            routings=config.routings, configuration=config, model_path=config.save_dir + '/trained_model.pkl')
        classifier.test(test_loader)
        classifier.show_reconstruction(test_loader, 50)

