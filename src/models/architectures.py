import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEmbedding(nn.Module):

    def __init__(self, input_shape=(), embedding_size=4):
        super(CNNEmbedding, self).__init__()
        self.cnn1 = nn.Conv2d(input_shape[0], 128, (7,7))
        self.cnn2 = nn.Conv2d(128, 256, (5,5))
        self.pooling = nn.MaxPool2d((2,2), (2,2))
        self.output_shape = self._get_output_shape(input_shape)
        self.linear = nn.Linear(self.output_shape, embedding_size)

    def _get_output_shape(self, input_shape):
        bs = 1
        x = torch.empty(bs, *input_shape)
        x = self._cnn(x).flatten(1)
        output_shape = x.size(1)
        return output_shape
    
    def _cnn(self, x):
        x = F.relu(self.cnn1(x))
        x = self.pooling(x)
        x = F.relu(self.cnn2(x))
        x = self.pooling(x)
        return x

    def _embedding(self, x):
        x = self.linear(x)
        return x

    def forward(self, x):
        embedding = self._embedding(self._cnn(x).flatten(1))

        return embedding
