'''
@author - ragarwal
@date - 07-oct-2020
'''

import numpy as np
import torch
import os
import shutil


class EarlyStoppingAndCheckPointer:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, basedir='.', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.basedir = basedir
        self.trace_func = trace_func
        self.modelCheckPointer = ModelCheckPointer()

    def __call__(self, val_loss, model, optimizer, epoch, scheduler= None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if scheduler is not None:
                self.modelCheckPointer.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict' : optimizer.state_dict(),
                                    'sched_dict' : scheduler.state_dict()},
                                   is_best=True,
                                   checkpoint=self.basedir)
            else:
                self.modelCheckPointer.save_checkpoint({'epoch': epoch + 1,
                                                        'state_dict': model.state_dict(),
                                                        'optim_dict': optimizer.state_dict()},
                                                       is_best=True,
                                                       checkpoint=self.basedir)
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.val_loss_min = val_loss

        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if scheduler is not None:
                self.modelCheckPointer.save_checkpoint({'epoch': epoch + 1,
                                                        'state_dict': model.state_dict(),
                                                        'optim_dict': optimizer.state_dict(),
                                                        'sched_dict' : scheduler.state_dict()},
                                                       is_best=False,
                                                       checkpoint=self.basedir)
            else:
                self.modelCheckPointer.save_checkpoint({'epoch': epoch + 1,
                                                        'state_dict': model.state_dict(),
                                                        'optim_dict': optimizer.state_dict()},
                                                       is_best=False,
                                                       checkpoint=self.basedir)
            if self.counter >= self.patience:
                self.early_stop = True
            self.trace_func(
                f'Validation loss did not decrease. The patience counter is ({self.counter} ).  Saving model as a resume checkpoint ...')
        else:
            self.best_score = score
            if scheduler is not None:
                self.modelCheckPointer.save_checkpoint({'epoch': epoch + 1,
                                                        'state_dict': model.state_dict(),
                                                        'optim_dict': optimizer.state_dict(),
                                                        'sched_dict' : scheduler.state_dict()},
                                                       is_best=True,
                                                       checkpoint=self.basedir)
            else:
                self.modelCheckPointer.save_checkpoint({'epoch': epoch + 1,
                                                        'state_dict': model.state_dict(),
                                                        'optim_dict': optimizer.state_dict()},
                                                       is_best=True,
                                                       checkpoint=self.basedir)
            self.counter = 0
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.val_loss_min = val_loss

    # def save_checkpoint(self, val_loss, model):
    #     '''Saves model when validation loss decrease.'''
    #     if self.verbose:
    #         self.trace_func(
    #             f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    #     torch.save(model.state_dict(), self.basedir)
    #     self.val_loss_min = val_loss


class ModelCheckPointer:
    def __init__(self):
        pass

    def save_checkpoint(self, state, is_best, checkpoint):
        """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
        checkpoint + 'best.pth.tar'
        Args:
            state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
            is_best: (bool) True if it is the best model seen till now
            checkpoint: (string) folder where parameters are to be saved
        """
        filepath = os.path.join(checkpoint, 'last.pth.tar')
        if not os.path.exists(checkpoint):
            print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
            os.mkdirs(checkpoint)
        else:
            print("Checkpoint Directory exists! ")
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

    def load_checkpoint(self, checkpoint, model, device, optimizer=None, scheduler=None):
        """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
        optimizer assuming it is present in checkpoint.
        Args:
            checkpoint: (string) filename which needs to be loaded
            model: (torch.nn.Module) model for which the parameters are loaded
            optimizer: (torch.optim) optional: resume optimizer from checkpoint
        """
        filepath = os.path.join(checkpoint, 'last.pth.tar')
        if not os.path.exists(filepath):
            raise ("File doesn't exist {}".format(filepath))
        checkpoint = torch.load(filepath, map_location=torch.device(device))
        model.load_state_dict(checkpoint['state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optim_dict'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['sched_dict'])

        return checkpoint['epoch']

    def loadBestModel(self,checkpoint, model, device):
        filepath = os.path.join(checkpoint, 'best.pth.tar')
        if not os.path.exists(filepath):
            raise ("File doesn't exist {}".format(filepath))
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

class CustomLossUtils:
    def __init__(self):
        pass

    def getL1RegularizationLossTerm(self, model, params=None):
        regularization_loss = 0
        if params is None:
            for param in model.parameters():
                regularization_loss += torch.sum(torch.abs(param))
        else:
            for param in params:
                regularization_loss += torch.sum(torch.abs(param))
