import torch
import torch.nn.functional as F
import torch.nn as nn

class FCN(nn.Module):
    """
    Proposed Fully Convolutional Network
    This function/module uses fully convolutional blocks to extract pixel-wise image features.
    Tested on 1024*1024, 512*512 resolution; RGB, Immunohistochemical color channels

    Keyword arguments:
    input_dim -- input channel, 3 for RGB images (default)
    """
    def __init__(self, input_dim, output_classes, p_mode = 'replicate'):
        super(FCN, self).__init__()
        #self.Dropout = nn.Dropout(p=0.05)
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1 ,padding_mode=p_mode)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, padding_mode=p_mode)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, padding_mode=p_mode)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode=p_mode)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, output_classes, kernel_size=1, stride=1, padding=0)

        #self.Dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        #x = self.Dropout(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        #x = self.Dropout(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn3(x)
        #x = self.Dropout(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn4(x)

        x = self.conv5(x)
        return x

class Predictor(nn.Module):
    def __init__(self, dim, p_mode = 'replicate'):
        super(Predictor, self).__init__()
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, padding_mode=p_mode)
        self.bn4 = nn.BatchNorm2d(dim)

        self.conv5 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn4(x)

        x = self.conv5(x)
        return x


class GCN(torch.nn.Module):
    """
    Proposed Graph Convolutional Network
    This function/module uses classic GCN layers to generate superpixels(nodes) classification.
    --"Semi-Supervised Classification with Graph Convolutional Networks",
    --Thomas N. Kipf, Max Welling, ICLR2017

    Keyword arguments:
    input_dim -- input channel, aligns with output channel from FCN
    output_classes --output channel, default 1 for our proposed loss function
    """
    def __init__(self, input_dim, output_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)
        self.conv4 = GCNConv(256, 64)
        self.conv5 = GCNConv(64, output_classes)
        #self.Dropout = nn.Dropout(p=0.5)

        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(256)
        # self.bn4 = nn.BatchNorm1d(64)
        #
        # self.lin1 = Linear(64, 256)
        # self.lin2 = Linear(256, 128)
        # self.lin3 = Linear(128, output_classes)

    def forward(self, data):
        x = self.conv1(data.x, edge_index = data.edge_index, edge_weight = data.edge_weight)
        x = F.relu(x)
        #x = self.Dropout(x)
        #x = self.bn1(x)

        x = self.conv2(x, edge_index = data.edge_index, edge_weight = data.edge_weight)
        x = F.relu(x)
        #x = self.Dropout(x)
        #x = self.bn2(x)

        x = self.conv3(x, edge_index = data.edge_index, edge_weight = data.edge_weight)
        x = F.relu(x)
        #x = self.Dropout(x)
        #x = self.bn3(x)

        x = self.conv4(x, edge_index = data.edge_index, edge_weight = data.edge_weight)
        x = F.relu(x)
        #x = self.bn4(x)

        x = self.conv5(x, edge_index = data.edge_index, edge_weight = data.edge_weight)


        return torch.tanh(x)
