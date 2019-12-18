import numpy as np
from builtins import object
from utils import *
import update_method


class CaptionTrain(object):
    """
    This class defines the training of the caption generator using SGD.
    """
    def __init__(self, data, model, **kwargs):
        """
        Initialization of CaptionTrain instance.
        Arguments:
            data: dictionary of training and validation dataset
            model: model object from RNNImageCaption
            optional arguments:
                update: string of update method of either 'sgd' or 'adam'
                update_params: dictionary of hyper-parameters for update method
                lr_decay: float of learning rate decay
                batch_size: integer of batch size for loss and gradient computation
                num_epochs: integer of number of epochs
                print_freq: integer of loss printing frequency steps
        """
        self.data = data
        self.model = model

        self.update = kwargs.pop('update', 'sgd')
        self.update_params = kwargs.pop('update_params', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.print_freq = kwargs.pop('print_freq', 10)
        # throw error if more parameters are detected
        if len(kwargs) > 0:
            unreg_args = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % unreg_args)
        # throw error if the update method is not supported
        if self.update not in ['sgd', 'adam']:
            raise ValueError('Unsupported update method %s' % self.update)
        self.update_method = getattr(update_method, self.update)  # get update method from file "update_method.py"

        # initialize training parameters
        self.epoch = 0
        self.best_params = {}
        self.best_val_acc = 0.0
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # perform a deep copy of the update method parameters for each model parameter
        self.update_params_all = {}
        for param in self.model.params:
            self.update_params_all[param] = {k:v for k, v in self.update_params.items()}

    def train(self):
        """
        Train the model.
        """
        num_train = self.data['train_features'].shape[0]
        num_iter_epoch = max(num_train // self.batch_size, 1)
        num_iters = num_iter_epoch * self.num_epochs

        for t in range(num_iters):
            self._gradient_update()
            if t % self.print_freq == 0:
                print('(Iteration %d / %d) loss: %f' % (t + 1, num_iters, self.loss_history[-1]))
            if (t + 1) % num_iter_epoch == 0:
                self.epoch += 1
                for param in self.update_params_all:
                    self.update_params_all[param]['lr'] *= self.lr_decay

    def _gradient_update(self):
        """
        Conduct a gradient update for training.
        """
        # sample minibatch
        captions, image_features, urls = sample_coco_minibatch(self.data, self.batch_size, split='train')
        # compute loss and gradient
        loss, gradients = self.model.loss(image_features, captions)
        self.loss_history.append(loss)
        # parameter update
        for para_name, param in self.model.params.items():
            dparam = gradients[para_name]
            next_param, params = self.update_method(param, dparam, self.update_params_all[para_name])
            self.model.params[para_name] = next_param
            self.update_params_all[para_name] = params
