import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, input_channels, conv_channels, embedding_dim):
        super(ResNet, self).__init__()

        self.embedding_dim = embedding_dim

        # residual 
        self.res1 = Conv2dStack(input_channels, conv_channels, 2)
        self.res2 = Conv2dRes(conv_channels, conv_channels, 2)
        self.res3 = Conv2dRes(conv_channels, conv_channels, 2)
        self.res4 = Conv2dRes(conv_channels, conv_channels, 2)
        self.res5 = Conv2dStack(conv_channels, conv_channels*2, 2)
        self.res6 = Conv2dRes(conv_channels*2, conv_channels*2, (3, 3))
        self.res7 = Conv2dRes(conv_channels*2, conv_channels*2, (3, 3))

        # fully connected
        self.embedding = nn.Linear(conv_channels * 2, conv_channels * 2)
        self.bn = nn.BatchNorm1d(conv_channels * 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)

        x = x.squeeze(2)

        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # fully connected
        x = self.embedding(x)
        x = self.bn(x)

        
        return x

class Conv2dStack(nn.Module):
    
    def __init__(self, input_channels, output_channels, pooling=2): 
        super(Conv2dStack, self).__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.bn   = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp   = nn.MaxPool2d(pooling)

    def forward(self, x):
        return self.mp(self.relu(self.bn(self.conv(x))))

class Conv2dRes(nn.Module):
   def __init__(self, input_channels, output_channels, pooling=2):          
       super(Conv2dRes, self).__init__()
                                                                            
       self.conv_1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
       self.bn_1   = nn.BatchNorm2d(output_channels)
       self.conv_2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
       self.bn_2   = nn.BatchNorm2d(output_channels)
       self.relu = nn.ReLU()
       self.mp   = nn.MaxPool2d(pooling)
                                                                            
   def forward(self, x):
      out = self.bn_2(self.conv_2(self.bn_1(self.conv_1(x))))
      out = x + out
      out = self.mp(self.relu(out))
      return out