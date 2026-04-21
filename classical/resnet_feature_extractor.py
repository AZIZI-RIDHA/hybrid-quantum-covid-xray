import torch, torchvision.models as models
class ResNetExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet50(pretrained=True)
        self.features = torch.nn.Sequential(*list(model.children())[:-1])
    def forward(self, x):
        return self.features(x).view(x.size(0), -1)
