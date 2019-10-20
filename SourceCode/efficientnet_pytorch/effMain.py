from __future__ import print_function
import torch

from ImageFolderWithPaths import ImageFolderWithPaths
#from ModelBinaryLayerAfterFC import ModelBinaryLayerAfterFC
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


    modelName = 'efficientnet-b0'

    imageSize = EfficientNet.get_image_size(modelName)
    print("imgSize " + str(imageSize))

    model = EfficientNet.pretrained(modelName, num_classes=2).cuda()
    model.eval()



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
                                             batch_size=8, shuffle=True,
                                             num_workers=4, pin_memory=True)

    # EXAMPLE USAGE:
    # instantiate the dataset and dataloader
    data_dir = "jpgImages/aug/test"
    dataset = ImageFolderWithPaths(data_dir, transform=tfms)  # our custom dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True,
                                             num_workers=4, pin_memory=True)

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

    # ---Binary---
    labels_map = json.load(open('Binary.txt'))
    labels_map = [labels_map[str(i)] for i in range(2)]
    # Classify with EfficientNet
    with torch.no_grad():
        for data, target, paths in dataloader:
            data, target = data.to(device), target.to(device)
            print(target)
            logits1 = model(data)
            index = 0
            for item in logits1:
                #print(paths[index])
                #index += 1
                preds1 = torch.topk(item, k=2).indices.squeeze(0).tolist()
                print('-----')
                for idx in preds1:
                    label = labels_map[idx]
                    prob = torch.softmax(item, dim=0)[idx].item()
                    print('{:<75} ({:.2f}%)'.format(label, prob * 100))
    # ---Binary---


    #---labels_map---
    labels_map = json.load(open('labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]
    img1 = Image.open('jpgImages/pandaSiedzi.jpg')
    img1 = tfms(img1).unsqueeze(0).cuda()
    #Classify with EfficientNet
    model.eval()
    with torch.no_grad():
        logits1 = model(img1)
        preds1 = torch.topk(logits1, k=2).indices.squeeze(0).tolist()
        print('-----')
        for idx in preds1:
            label = labels_map[idx]
            prob = torch.softmax(logits1, dim=1)[0, idx].item()
            print('{:<75} ({:.2f}%)'.format(label, prob * 100))
    # ---labels_map---

    showExecutionTime(startTime)


if __name__ == '__main__':
    main()
