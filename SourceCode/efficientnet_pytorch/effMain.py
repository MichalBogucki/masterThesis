from __future__ import print_function
import torch

from compareImages import compareImages
from setParserArguments import setParserArgumentsMnist
from showExecutionTime import *
from modelSourceCode import EfficientNet

import json
import PIL
from PIL import Image
from torchvision import transforms

import Augmentor


def main():
    startTime = datetime.now()

    p = Augmentor.Pipeline("aug")

    p.skew(probability=0.5)
    p.shear(probability=0.5, max_shear_left=10, max_shear_right=10)
    p.random_distortion(probability=0.5, grid_width=8, grid_height=8, magnitude=2)
    p.rotate(probability=0.5, max_left_rotation=15, max_right_rotation=15)
    p.flip_left_right(probability=0.5)
    p.zoom_random(probability=0.5, percentage_area=0.8)
    p.sample(10000)

    print("augemented")
    return

    # Training settings
    args = setParserArgumentsMnist()

    useCuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if useCuda else "cpu")
    kwargs = {'num_workers': 3, 'pin_memory': True} if useCuda else {}

    # trainLoader = setLoader(kwargs, shouldTrain=True, shouldDownload=False, batchSize=args.batch_size)
    # testLoader = setLoader(kwargs, shouldTrain=False, shouldDownload=False, batchSize=args.test_batch_size)

    # model = NetworkArchitecture().to(device)

    modelName = 'efficientnet-b2'
    imageSize = EfficientNet.get_image_size(modelName)
    print("imgSize " + str(imageSize))
    model = EfficientNet.pretrained(modelName).cuda()
    model.eval()

    # for epoch in range(1, args.epochs + 1):

    # Preprocess image
    tfms = transforms.Compose([
        transforms.Resize(imageSize, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(imageSize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Open image

    nameViews = [
        'view_d15.jpg',
        'view_g15.jpg',
        'view_m15.jpg',
        'view_p15.jpg',
        'view_rl15.jpg',
        'view_rp15.jpg']

    compareImages(model, nameViews, tfms)

    # ---------------------------------

    nameImg = [
        'building.jpg',
        'dresKolano.jpg',
        'dresTatu.jpg',
        #'dresidzie.jpg',
        'pandaSiedzi.jpg',
        'pandaSiedzi128.jpg',
        'pandaStoi.jpg',
        'pandaStoi200.jpg',
        #'pies.jpg'
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


if __name__ == '__main__':
    main()
