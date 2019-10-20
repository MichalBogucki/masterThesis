from torch import nn


class ManualFineTuneModel(nn.Module):
    def __init__(self, original_model, inFeatuers, num_classes):
        super(ManualFineTuneModel, self).__init__()
        # Everything except the last linear layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(inFeatuers, num_classes)
        )

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y