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
    for epoch in range(1, 2 + 1):
        # Open image
        img = Image.open('img.jpg')
        img2 = Image.open('img2.jpg')
        # Preprocess image
        tfms = transforms.Compose([transforms.Resize(imageSize), transforms.CenterCrop(imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])
        img = tfms(img).unsqueeze(0)
        img2 = tfms(img2).unsqueeze(0)

        # Load class names
        labels_map = json.load(open('labels_map.txt'))
        labels_map = [labels_map[str(i)] for i in range(1000)]

        # Classify with EfficientNet
        model.eval()
        with torch.no_grad():
            logits = model(img)
            logits2 = model(img2)
        preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()
        preds2 = torch.topk(logits2, k=5).indices.squeeze(0).tolist()

        print('-----')
        for idx in preds:
            label = labels_map[idx]
            prob = torch.softmax(logits, dim=1)[0, idx].item()
            print('{:<75} ({:.2f}%)'.format(label, prob * 100))

        print('-----2')
        for idx in preds2:
            label = labels_map[idx]
            prob = torch.softmax(logits2, dim=1)[0, idx].item()
            print('{:<75} ({:.2f}%)'.format(label, prob * 100))

    showExecutionTime(startTime)



if __name__ == '__main__':
    main()
