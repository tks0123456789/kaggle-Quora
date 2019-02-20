import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler, WeightedRandomSampler


def get_param_size(model, trainable=True):
    if trainable:
        psize = np.sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    else:
        psize = np.sum([np.prod(p.size()) for p in model.parameters()])
    return psize


def get_dataloader(x, y=None, weights=None, num_samples=None, batch_size=32,
                   dtype_x=torch.float, dtype_y=torch.float, training=True):
    x_tensor = torch.tensor([x_1 for x_1 in x], dtype=dtype_x)
    if y is None:
        data = TensorDataset(x_tensor)
    else:
        y_tensor = None if y is None else torch.tensor([y_1 for y_1 in y], dtype=dtype_y)
        data = TensorDataset(x_tensor, y_tensor)
    if training:
        if weights is None:
            sampler = RandomSampler(data)
        else:
            sampler = WeightedRandomSampler(weights, num_samples)
    else:
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, shuffle=False, batch_size=batch_size)
    return dataloader


# https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856
class EMA():
    def __init__(self, model, mu, level='batch', n=1):
        """
        level: 'batch' or 'epoch'
          'batch': Update params every n batches.
          'epoch': Update params every epoch.
        """
        # self.ema_model = copy.deepcopy(model)
        self.mu = mu
        self.level = level
        self.n = n
        self.cnt = self.n
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data

    def _update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1 - self.mu) * param.data + self.mu * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def set_weights(self, ema_model):
        for name, param in ema_model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]

    def on_batch_end(self, model):
        if self.level is 'batch':
            self.cnt -= 1
            if self.cnt == 0:
                self._update(model)
                self.cnt = self.n

    def on_epoch_end(self, model):
        if self.level is 'epoch':
            self._update(model)
