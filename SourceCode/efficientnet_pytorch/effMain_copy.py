from __future__ import print_function
import torch
from torch.nn import functional as F

from ManualFineTuneModel import ManualFineTuneModel
from ModelBinaryLayerAfterFC import ModelBinaryLayerAfterFC
from compareImages import compareImages
from setParserArguments import setParserArgumentsMnist
from showExecutionTime import *
from modelSourceCode import EfficientNet

import json
import PIL
from PIL import Image
from torchvision import transforms, datasets


def main():
    startTime = datetime.now()

    #performAugmentation()

    # Training settings
    args = setParserArgumentsMnist()

    useCuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if useCuda else "cpu")
    kwargs = {'num_workers': 3, 'pin_memory': True} if useCuda else {}


    modelName = 'efficientnet-b7'

    imageSize = EfficientNet.get_image_size(modelName)
    print("imgSize " + str(imageSize))

    model = EfficientNet.pretrained(modelName, num_classes=2).cuda()
    model.eval()

    #model2 = ModelBinaryLayerAfterFC(model)
    #model3 = ManualFineTuneModel(model, model._fc.in_features, 2)


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
    testDataset = datasets.ImageFolder(root='jpgImages/aug/test',
                                       transform=tfms)
    # trainLoader = torch.utils.data.DataLoader(trainDataset,
    #                                           batch_size=4, shuffle=True,
    #                                           num_workers=4)
    testLoader = torch.utils.data.DataLoader(testDataset,
                                             batch_size=4, shuffle=False,
                                             num_workers=4, pin_memory=True)

    # #----
    # model.eval()
    # test_loss = 0
    # correct = 0
    # with torch.no_grad():
    #     for data, target in testLoader:
    #         data, target = data.to(device), target.to(device)
    #         output = model(data)
    #         # print("output")
    #         # print(output)
    #         # print("target")
    #         # print(target)
    #         test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
    #         pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    #         correct += pred.eq(target.view_as(pred)).sum().item()
    #         # print('Test set [x/y]: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
    #         #     .format(test_loss, correct, len(testLoader.dataset), 100. * correct / len(testLoader.dataset)))
    #
    # test_loss /= len(testLoader.dataset)
    #
    # print('Test set [x/y]: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
    #       .format(test_loss, correct, len(testLoader.dataset), 100. * correct / len(testLoader.dataset)))
    #----
    # Open image
    nameViews = [
        'view_d15.jpg',
        'view_g15.jpg',
        'view_m15.jpg',
        'view_p15.jpg',
        'view_rl15.jpg',
        'view_rp15.jpg']


    #return visualizeGraphWithOnnxToNetron(model)

    #compareImages(model, nameViews, tfms)

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

    #compareImages(model, nameImg, tfms)

    # ---------------------------------


    #---Binary---
    labels_map = json.load(open('Binary.txt'))
    labels_map = [labels_map[str(i)] for i in range(2)]
    #Classify with EfficientNet
    model.eval()
    with torch.no_grad():
        for data, target in testLoader:
                data, target = data.to(device), target.to(device)
                logits1 = model(data)
                for item in logits1:
                    #print(item.shape)
                    #print(item)
                    preds1 = torch.topk(item, k=2).indices.squeeze(0).tolist()
                    #print(preds1)
                    print('-----')
                    for idx in preds1:
                        #print(idx)
                        label = labels_map[idx]
                        prob = torch.softmax(item, dim=0)[idx].item()
                        print('{:<75} ({:.2f}%)'.format(label, prob * 100))
    return

    showExecutionTime(startTime)


if __name__ == '__main__':
    main()
