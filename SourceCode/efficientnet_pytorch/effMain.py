from __future__ import print_function
import torch
from setParserArguments import setParserArgumentsMnist
from showExecutionTime import *
from modelSourceCode import EfficientNet

import json
from PIL import Image
from torchvision import transforms


def main():
    startTime = datetime.now()

    # Training settings
    args = setParserArgumentsMnist()

    useCuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if useCuda else "cpu")
    kwargs = {'num_workers': 3, 'pin_memory': True} if useCuda else {}

    #trainLoader = setLoader(kwargs, shouldTrain=True, shouldDownload=False, batchSize=args.batch_size)
    #testLoader = setLoader(kwargs, shouldTrain=False, shouldDownload=False, batchSize=args.test_batch_size)

    # model = NetworkArchitecture().to(device)

    modelName = 'efficientnet-b0'
    imageSize = EfficientNet.get_image_size(modelName)  # 224
    model = EfficientNet.pretrained(modelName)

    #for epoch in range(1, args.epochs + 1):
    for epoch in range(1, 2):
        # Open image
        img1 = Image.open('img.jpg')
        img2 = Image.open('img2.jpg')
        img3 = Image.open('img_200.jpg')
        img4 = Image.open('img2_128.jpg')
        # Preprocess image
        tfms = transforms.Compose([transforms.Resize(imageSize), transforms.CenterCrop(imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
        img1 = tfms(img1).unsqueeze(0)
        img2 = tfms(img2).unsqueeze(0)
        img3 = tfms(img3).unsqueeze(0)
        img4 = tfms(img4).unsqueeze(0)

        # Load class names
        labels_map = json.load(open('labels_map.txt'))
        labels_map = [labels_map[str(i)] for i in range(1000)]

        # Classify with EfficientNet
        model.eval()
        with torch.no_grad():
            logits1 = model(img1)
            logits2 = model(img2)
            logits3 = model(img3)
            logits4 = model(img4)
        preds1 = torch.topk(logits1, k=5).indices.squeeze(0).tolist()
        preds2 = torch.topk(logits2, k=5).indices.squeeze(0).tolist()
        preds3 = torch.topk(logits2, k=5).indices.squeeze(0).tolist()
        preds4 = torch.topk(logits2, k=5).indices.squeeze(0).tolist()

        print('-----')
        for idx in preds1:
            label = labels_map[idx]
            prob = torch.softmax(logits1, dim=1)[0, idx].item()
            print('{:<75} ({:.2f}%)'.format(label, prob * 100))

        print('-----2')
        for idx in preds2:
            label = labels_map[idx]
            prob = torch.softmax(logits2, dim=1)[0, idx].item()
            print('{:<75} ({:.2f}%)'.format(label, prob * 100))

        print('-----3')
        for idx in preds3:
            label = labels_map[idx]
            prob = torch.softmax(logits3, dim=1)[0, idx].item()
            print('{:<75} ({:.2f}%)'.format(label, prob * 100))

        print('-----4')
        for idx in preds4:
            label = labels_map[idx]
            prob = torch.softmax(logits4, dim=1)[0, idx].item()
            print('{:<75} ({:.2f}%)'.format(label, prob * 100))


    showExecutionTime(startTime)



if __name__ == '__main__':
    main()