import json

import torch
import torch.nn as nn
import math
import torch.optim as optim
import torch.nn.functional as F


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()

        self.rnn1 = nn.LSTM(input_size=50, hidden_size=50, num_layers=2, batch_first=True,bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=200, hidden_size=32, num_layers=2, batch_first=True,bidirectional=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x_mirna, x_mrna):
        y, (h_n, h_c) = self.rnn1(x_mirna, None)
        h_mirna = y
        h_mirna = F.relu(h_mirna)

        y, (h_n, h_c) = self.rnn1(x_mrna, None)
        h_mrna = y
        h_mrna = F.relu(h_mrna)

        h = torch.cat((h_mirna, h_mrna), dim=2)
        # h = h.transpose(2, 1)

        y, (h_n, h_c) = self.rnn2(h)
        y = torch.cat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1)
        #y=y[:, -1, :]
        y = F.relu(y)
        #y = self.drop(y)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        return y

    def size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HyperParam:
    #for CNN1d
    def __init__(self, filters=None, kernels=None, model_json=None):
        self.dictionary = dict()
        self.name_postfix = str()

        if (filters is not None) and (kernels is not None) and (model_json is None):
            for i, (f, k) in enumerate(zip(filters, kernels)):
                setattr(self, 'f{}'.format(i + 1), f)
                setattr(self, 'k{}'.format(i + 1), k)
                self.dictionary.update({'f{}'.format(i + 1): f, 'k{}'.format(i + 1): k})
            self.len = i + 1

            for key, value in self.dictionary.items():
                self.name_postfix = "{}_{}-{}".format(self.name_postfix, key, value)
        elif model_json is not None:
            # self.dictionary = json.loads(model_json)
            for i, (key, value) in enumerate(self.dictionary.items()):
                setattr(self, key, value)
                self.name_postfix = "{}_{}-{}".format(self.name_postfix, key, value)
            self.len = (i + 1) // 2

    def __len__(self):
        return self.len



class CNN1d(nn.Module):
    # Change input shape in functions/Dataset and TrainDataset if using.
    def __init__(self, hparams=None, hidden_units=30, input_shape=(1, 4, 30), name_prefix="model"):
        super(CNN1d, self).__init__()

        if hparams is None:
            filters, kernels = [32, 16, 64, 16], [3, 3, 3, 3]
            hparams = HyperParam(filters, kernels)
        self.name = "{}{}".format(name_prefix, hparams.name_postfix)

        if (isinstance(hparams, HyperParam)) and (len(hparams) == 4):
            self.embd1 = nn.Conv1d(4, hparams.f1, kernel_size=hparams.k1, padding=((hparams.k1 - 1) // 2))
            self.conv2 = nn.Conv1d(hparams.f1 * 2, hparams.f2, kernel_size=hparams.k2)
            self.conv3 = nn.Conv1d(hparams.f2, hparams.f3, kernel_size=hparams.k3)
            self.conv4 = nn.Conv1d(hparams.f3, hparams.f4, kernel_size=hparams.k4)

            """ out_features = ((in_length - kernel_size + (2 * padding)) / stride + 1) * out_channels """
            flat_features = self.forward(torch.rand(input_shape), torch.rand(input_shape), flat_check=True)
            self.fc1 = nn.Linear(flat_features, hidden_units)
            self.fc2 = nn.Linear(hidden_units, 2)
        else:
            raise ValueError("not enough hyperparameters")

    def forward(self, x_mirna, x_mrna, flat_check=False):
        h_mirna = F.relu(self.embd1(x_mirna))
        h_mrna = F.relu(self.embd1(x_mrna))

        h = torch.cat((h_mirna, h_mrna), dim=1)

        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))

        h = h.view(h.size(0), -1)
        if flat_check:
            return h.size(1)
        h = self.fc1(h)
        y = self.fc2(h)  # y = F.softmax(self.fc2(h), dim=1)

        return y

    def size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
