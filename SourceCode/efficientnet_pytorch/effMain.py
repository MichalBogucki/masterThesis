from __future__ import print_function
import torch
from torch import nn

from compareImages import compareImages
from setParserArguments import setParserArgumentsMnist
from showExecutionTime import *
from modelSourceCode import EfficientNet

import json
import PIL
from PIL import Image
from torchvision import transforms


from visualizeGraphWithOnnxToNetron import visualizeGraphWithOnnxToNetron


def main():
    startTime = datetime.now()

    #performAugmentation()

    # Training settings
    args = setParserArgumentsMnist()

    useCuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if useCuda else "cpu")
    kwargs = {'num_workers': 3, 'pin_memory': True} if useCuda else {}


    modelName = 'efficientnet-b0'

    imageSize = EfficientNet.get_image_size(modelName)
    print("imgSize " + str(imageSize))
    #model = EfficientNet.pretrained(modelName)
    model = EfficientNet.pretrained(modelName).cuda()
    model.eval()
    print("Original")
    print(model)

    model2 = ModelLastBinaryLayer(model)
    print("ModelLastBinaryLayer")
    print(model2)

    model3 = FineTuneModel(model,model._fc.in_features, 2)
    print("FineTuneModel")
    print(model3)
    return
    print("dupa")
    print(model._fc.in_features)


    # ----------
    # for epoch in range(1, args.epochs + 1):

    # Preprocess image
    tfms = transforms.Compose([
        transforms.Resize(imageSize, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(imageSize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # trainDataset = datasets.ImageFolder(root='jpgImages/aug/train',
    #                                     transform=tfms)
    # testDataset = datasets.ImageFolder(root='jpgImages/aug/test',
    #                                    transform=tfms)
    # trainLoader = torch.utils.data.DataLoader(trainDataset,
    #                                           batch_size=4, shuffle=True,
    #                                           num_workers=4)
    # testLoader = torch.utils.data.DataLoader(testDataset,
    #                                          batch_size=4, shuffle=True,
    #                                          num_workers=4)
    # Open image
    nameViews = [
        'view_d15.jpg',
        'view_g15.jpg',
        'view_m15.jpg',
        'view_p15.jpg',
        'view_rl15.jpg',
        'view_rp15.jpg']


    #return visualizeGraphWithOnnxToNetron(model)

    compareImages(model, nameViews, tfms)

    # ---------------------------------

    nameImg = [
        'building.jpg',
        'dresKolano.jpg',
        'dresTatu.jpg',
        'dresidzie.jpg',
        'pandaSiedzi.jpg',
        'pandaSiedzi128.jpg',
        'pandaStoi.jpg',
        'pandaStoi200.jpg',
        'pies.jpg'
        ]

    compareImages(model, nameImg, tfms)

    # ---------------------------------


    # Load class names
    labels_map = json.load(open('labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]

    # Classify with EfficientNet
    # model.eval()
    # with torch.no_grad():
    # logits1 = model(img1)
    # logits2 = model(img2)
    # logits3 = model(img3)
    # logits4 = model(img4)
    #     print(logits1.shape)
    # preds1 = torch.topk(logits1, k=5).indices.squeeze(0).tolist()
    # preds2 = torch.topk(logits2, k=5).indices.squeeze(0).tolist()
    # preds3 = torch.topk(logits2, k=5).indices.squeeze(0).tolist()
    # preds4 = torch.topk(logits2, k=5).indices.squeeze(0).tolist()

    # features1 = model.extract_features(img1)
    # features2 = model.extract_features(img2)
    # features3 = model.extract_features(img3)
    # features4 = model.extract_features(img4)
    # features5 = model.extract_features(img5)
    # features6 = model.extract_features(img6)
    #
    # print("\n-mse_loss--sum----")
    # print(features1.shape)
    # loss1 = F.mse_loss(features1, features2, reduction='mean').item()
    # print("panda " + str(loss1))
    # loss2 = F.mse_loss(features1, features3, reduction='mean').item()
    # print("panda " + str(loss2))
    # loss3 = F.mse_loss(features4, features5, reduction='mean').item()
    # print("dres " + str(loss3))
    # loss4 = F.mse_loss(features4, features6, reduction='mean').item()
    # print("dres " + str(loss4))
    # loss5 = F.mse_loss(features1, features4, reduction='mean').item()
    # print("dres-panda " + str(loss5))
    # loss6 = F.mse_loss(features2, features6, reduction='mean').item()
    # print("dres-panda " + str(loss6))
    #
    # print("\n------------------\n")
    #
    # print('-----')
    # for idx in preds1:
    #     label = labels_map[idx]
    #     prob = torch.softmax(logits1, dim=1)[0, idx].item()
    #     print('{:<75} ({:.2f}%)'.format(label, prob * 100))
    #
    # print('-----2')
    # for idx in preds2:
    #     label = labels_map[idx]
    #     prob = torch.softmax(logits2, dim=1)[0, idx].item()
    #     print('{:<75} ({:.2f}%)'.format(label, prob * 100))
    #
    # print('-----3')
    # for idx in preds3:
    #     label = labels_map[idx]
    #     prob = torch.softmax(logits3, dim=1)[0, idx].item()
    #     print('{:<75} ({:.2f}%)'.format(label, prob * 100))
    #
    # print('-----4')
    # for idx in preds4:
    #     label = labels_map[idx]
    #     prob = torch.softmax(logits4, dim=1)[0, idx].item()
    #     print('{:<75} ({:.2f}%)'.format(label, prob * 100))

    showExecutionTime(startTime)

class ModelLastBinaryLayer(nn.Module):
    def __init__(self, pretrained_model):
        super(ModelLastBinaryLayer, self).__init__()
        self.pretrained_model = pretrained_model
        self.last_layer = nn.Linear(1000, 2)

    def forward(self, x):
        return self.last_layer(self.pretrained_model(x))


class FineTuneModel(nn.Module):
    def __init__(self, original_model, inFeatuers, num_classes):
        super(FineTuneModel, self).__init__()
        # Everything except the last linear layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(inFeatuers, num_classes)
        )
        self.modelName = 'LightCNN-29'
        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False


    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y



if __name__ == '__main__':
    main()
