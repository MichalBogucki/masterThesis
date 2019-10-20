from torch import nn


class ModelBinaryLayerAfterFC(nn.Module):
    def __init__(self, pretrained_model):
        super(ModelBinaryLayerAfterFC, self).__init__()
        self.pretrained_model = pretrained_model
        self.last_layer = nn.Linear(1000, 2)

    def forward(self, x):
        return self.last_layer(self.pretrained_model(x))