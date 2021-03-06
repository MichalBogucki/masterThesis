from __future__ import print_function
import torch
import time
import copy
import torch.optim as optim
import torch.nn as nn
import os
import csv
import bisect
from operator import itemgetter

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

from performAugmentation import performAugmentation
from UseGluonCv_YOLO3 import UseGluonCv_YOLO3

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

from ImageFolderWithPaths import ImageFolderWithPaths
# from ModelBinaryLayerAfterFC import ModelBinaryLayerAfterFC
from setParserArguments import setParserArgumentsMnist
from visualizeGraphWithOnnxToNetron import visualizeGraphWithOnnxToNetron
from showExecutionTime import *
from modelSourceCode import EfficientNet
import torchvision
import json
import PIL
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np


def main():
    ##---------------Gluon CV----------------
    # UseGluonCv_YOLO3()
    # return
    # return
    ##---------------Gluon CV----------------

    startTime = datetime.now()

    # performAugmentation()

    # Training settings
    device = torch.device("cuda")
    modelName = 'efficientnet-b0tuned_1000_moana_bigger'
    # modelName = 'efficientnet-b0tuned_1000_moana' #ToDo with small samples
    # modelName = 'efficientnet-b0'
    # modelName = 'efficientnet-b0tuned_1000_moana_5epoch'
    # modelName = 'efficientnet-b4tuned'
    # modelName = 'efficientnet-b0tuned_1000_moana_undersampling'

    #imageSize = EfficientNet.get_image_size(modelName)  # Todo for training
    imageSize = 576 # Todo used for ImageLocalization
    print("imgSize " + str(imageSize))

    # Number of classes in the dataset
    num_PreLoad_Classes = 2
    num_tunedClasses = 2
    # Batch size for training (change depending on how much memory you have)
    batch_size = 1
    #batch_size = 60 # ToDo 60 for Classification, 2 for Localization
    # Number of epochs to train for
    num_epochs = 5

    model = EfficientNet.pretrained(modelName, num_classes=num_PreLoad_Classes, tuned_classes=num_tunedClasses).cuda()
    model.eval()

    # ----------
    # for epoch in range(1, args.epochs + 1):

    # Preprocess image
    tfms = transforms.Compose([
        transforms.Resize(size=imageSize, interpolation=PIL.Image.BICUBIC),
        #transforms.CenterCrop(imageSize), #Todo uncomment for training
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # trainDataset = datasets.ImageFolder(root='jpgImages/aug/train',
    #                                     transform=tfms)
    # testDataset = datasets.ImageFolder(root='jpgImages/aug/val',
    #                                    transform=tfms)
    # trainLoader = torch.utils.data.DataLoader(trainDataset,
    #                                           batch_size=4, shuffle=True,
    #                                           num_workers=4)
    # testLoader = torch.utils.data.DataLoader(testDataset,
    #                                          batch_size=8, shuffle=True,
    #                                          num_workers=4, pin_memory=True)

    # data_dir = "jpgImages/aug/val"#/tiny"
    # data_dir = "jpgImages/aug/train"
    # data_dir = "UseMeMaui"  # ToDo use for ImageLocalization
    data_dir = "jpgImages/aug/val" # Todo use for validation
    # data_dir = "jpgImages/aug/train"
    # #$$$$$$$$$$ CSV USING $$$$$$$$$$$$
    # listOfCsvRows = ReadCsvToList(data_dir)
    # fileName ='jpgImages\\video\\frames\\moana_1080p_2min 004.jpg'
    # searchedValue = SearchCollectionForValue(fileName, listOfCsvRows)
    # print(searchedValue)
    # $$$$$$$$$$ CSV USING $$$$$$$$$$$$

    # EXAMPLE USAGE:
    # instantiate the dataset and dataloader

    dataset = ImageFolderWithPaths(data_dir, transform=tfms)  # our custom dataset
    datasetSize = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, #Todo shuffle=True for training
                                             num_workers=0, pin_memory=True)

    # return visualizeGraphWithOnnxToNetron(model) #ToDo ONYXX vizualiser

    # ---Binary---
    startTime = datetime.now()  # Todo deleteME
    print('\n')
    measureDatasetAccuracy(dataloader, device, model)  # Todo UNCOMMENT ME
    return
    return
    validateBinaryTatoo(dataloader, device, model, data_dir, tfms)
    showExecutionTime(startTime)  # Todo deleteME
    if ('tuned' in modelName):
        return
    # return
    # ---Binary---

    # ---labels_map---
    # validateLabelsMap(model, tfms)
    # ---labels_map---

    # +++++ Train Time ++++++++
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True
    # set_parameter_requires_grad(model, feature_extract)
    # Create the Optimizer
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print("\t", name)
    # else:
    #     for name, param in model.named_parameters():
    #         if param.requires_grad == True:
    #             print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # setup DataSet
    data_dir = "jpgImages/aug"
    image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), tfms) for x in ['train', 'val']}
    # Setup the loss fxn

    criterion = nn.CrossEntropyLoss()
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in
        ['train', 'val']}

    # Train and evaluate
    model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer_ft, device, num_epochs=num_epochs)
    # +++++ Train Time ++++++++

    # compareImages(model, nameViews, tfms)
    # compareImages(model_ft, nameImg, tfms)
    # ---------------------------------

    # ---Binary---
    measureDatasetAccuracy(dataloader, device, model)
    # ---Binary---

    # ---labels_map---
    # validateLabelsMap(model_ft, tfms)
    # ---labels_map---

    #saveTrainedModel(model, modelName)  # Todo UNCOMMENT ME
    #saveTrainedModelFT(model_ft, modelName)  # Todo UNCOMMENT ME

    showExecutionTime(startTime)


def SearchCollectionForValue(fileName, listOfCsvRows):
    # searchedValue = 0
    # for row in listOfCsvRows:
    #     if row[0] == fileName:
    #         searchedValue = row[1]
    #         break
    searchedFloatValue = [item[2:6] for item in listOfCsvRows if (fileName in item[0])]
    intCoordinates = [int(float(i)) for i in searchedFloatValue[0]]
    return intCoordinates


def ReadCsvToList(data_dir):
    listOfCsvRows = []
    #with open('{}\detected.csv'.format(data_dir), 'r') as f: #ToDo for Maui
    with open('{}\detected_one_moana.csv'.format(data_dir), 'r') as f:  # ToDo for Moana
        # with open('{}\detected_single.csv'.format(data_dir), 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            listOfCsvRows.append(row)
    return listOfCsvRows


def validateBinaryTatoo(dataloader, device, model, data_dir, tfms):
    data_dir = "UseMeMaui" #ToDo for Lozalization
    labels_map = json.load(open('Binary.txt'))
    labels_map = [labels_map[str(i)] for i in range(2)]
    # print(labels_map) #Todo DELETE ME
    # Classify with EfficientNet
    model.eval()
    with torch.no_grad():
        batchIteration = 1
        listOfCsvRows = ReadCsvToList(data_dir)
        skipped = SkipFirstCsvRow(listOfCsvRows)
        batchesNumber = (len(skipped))
        for row in skipped:
            # for batchData, target, paths in dataloader:
            #    batchData, target = batchData.to(device), target.to(device)
            actFileNameWithPath = row[0]
            onlyFileName = os.path.basename(actFileNameWithPath)
            loadImg = Image.open("{}/output/{}".format(data_dir, onlyFileName))
            # print(onlyFileName)
            batchData = tfms(loadImg).unsqueeze(0).cuda()
            maxName = ''
            maxVal = 0
            xBest = 0
            yBest = 0
            # print('batchData.shape')
            # print(batchData.shape)
            # print(type(batchData))
            # print('target')
            # print(target)
            # print('target.shape: {}'.format(target.shape))
            index = 0
            print('----- BATCH {}/{}-----'.format(batchIteration, batchesNumber))
            batchIteration += 1
            # actFileNameWithPath = paths[index]
            # onlyFileName = os.path.basename(actFileNameWithPath)
            # searchedValue = SearchCollectionForValue(onlyFileName, listOfCsvRows)
            # print('batchData.shape: {}'.format(batchData.shape))
            # imgTemp = batchData[index]
            imgTemp = batchData.squeeze(0)  # ToDo because readImg is 4d not 3d (0th is useless)
            # print('batchData[index]shape: {}'.format(imgTemp.shape))
            # ime = imgTemp.crop((left, top, right, bottom))
            # left, top, right, bottom = 300, 100, 650, 500
            # print(onlyFileName)
            # print(searchedValue)
            #                print('imgTemp: {}'.format(imgTemp.shape))
            left, top, right, bottom = [int(float(i)) for i in row[2:6]]
            #print('left')
            #print(left)
            #print('top')
            #print(top)
            #print('right')
            #print(right)
            #print('bottom')
            #print(bottom)
            imgCropped = imgTemp[:, top:bottom, left:right]
            # print('imgCropped: {}'.format(imgCropped.shape))
            # print('img.size(0)={}, img.size(1)={} img.size(2)={} '.format(imgTemp.size(0), imgTemp.size(1),
            #                                                            imgTemp.size(2)))
            # print('{}/{} "{}"'.format(index + 1, len(logits1), actFileNameWithPath))
            foldSize = 100
            step = 15

            minFromXY = min([imgCropped.size(1), imgCropped.size(2)])
            if (minFromXY < 150):
                foldSize = minFromXY
                step = int(minFromXY / 5)

            torch.cuda.empty_cache() #Todo empty cache
            # startTime = datetime.now()  # todo delete ME --time--
            ############### ---------- folding images into SmallerImages ---------- ##############
            # imgTempPatches = imgTemp.unfold(0, 3, 3).unfold(1, foldSize, step).unfold(2, foldSize, step).squeeze(0)
            imgTempPatches = imgCropped.unfold(0, 3, 3).unfold(1, foldSize, step).unfold(2, foldSize, step).squeeze(0)
            # print('imgTempPatches.shape: {}'.format(imgTempPatches.shape))
            flate2 = torch.flatten(imgTempPatches, 0, 1)
            # print('flate2.shape: {}'.format(flate2.shape))
            # print('flate2[0].shape: {}'.format(flate2[0].shape))
            # 100x100; 200x100; 300x100; 400x100
            # ++++++++++ put flatted smallImages into NN +++++++++++++ #
            nnOutput = model(flate2)
            x2 = foldSize
            y2 = foldSize
            orderedList = []
            #print('---- TOP prediction ----')  # ToDo heatMap2
            for item2 in nnOutput:
                # predictedClasses2 = torch.topk(item2, k=2).indices.squeeze(0).tolist()
                predictedClasses2 = torch.topk(item2, k=2)[1].squeeze(0).tolist()
                if (x2 > imgCropped.size(2)):
                    #print('') #ToDo heatMap2
                    x2 = foldSize
                    y2 = (y2 + step)
                for idx2 in predictedClasses2:
                    if (idx2 == 1):
                        xLeft = x2-foldSize+left
                        yTop = y2-foldSize+top
                        label2 = labels_map[idx2]
                        prob2 = torch.softmax(item2, dim=0)[idx2].item()
                        tempName = '{} ({}) x={} y={} x2={} y2={}'.format(label2, prob2 * 100, (xLeft),
                                                                          (yTop), x2, y2)
                        #print('{} x={} y={} {:<75} ({:.2f}%)'.format(label2,x2, y2, label2, prob2 * 100)) #ToDo heatMap
                        maxName, maxVal, xBest, yBest = checkMax(maxName, maxVal, tempName, prob2, label2, idx2,
                                                                 xBest, yBest, xLeft, yTop)
                        print('')  # ToDo heatMap2
                        print('x{};y{};{:.0f}'.format(xLeft, yTop,prob2 * 100), end=";")  # ToDo heatMap2
                        aTuple = (xLeft, yTop, prob2 * 100)
                        orderedList.append(aTuple)
                        # if (prob2 * 100 > 94):
                        #     print('loop')  # ToDo heatMap2 in new line
                        #     print('x{}_y{}'.format(xLeft,yTop), end=";") #ToDo heatMap2
                        #     print('{:.0f}'.format(prob2 * 100), end=";") #ToDo heatMap2
                x2 = (x2 + step)

                #print('{:.0f}'.format(prob2 * 100), end=";")
            # ++++++++++ put flatted smallImages into NN +++++++++++++ #
            # showExecutionTime(startTime)  # todo delete ME --time--
            # print('maxName: "{}"'.format(maxName))
            # print('maxVal: ({:.2f}%)'.format(maxVal * 100))

            # if (maxVal * 100 > 94):
            #     print('save')  # ToDo heatMap2 in new line
            #     print('x{}_y{}'.format(left + xBest, top + yBest), end=";")  # ToDo heatMap2
            #     print('{:.0f}'.format(maxVal * 100), end=";")  # ToDo heatMap2

            # showImgOnPlot(imgTemp, xBest, yBest, foldSize, foldSize, data_dir, onlyFileName, maxVal, batchIteration) #ToDO saveImageToFile
            # print('savedImage')
            # PrintTopTen(orderedList) #ToDO printTopTenResults


def PrintTopTen(orderedList):
    print('\n---Top 10---')
    orderedList.sort(key=lambda x: x[2], reverse=True)
    topTen = orderedList[:10]
    for listItem in topTen:
        print('\nx{};y{};{:.0f}'.format(listItem[0], listItem[1], listItem[2]), end=";")


def SkipFirstCsvRow(listOfCsvRows):
    return listOfCsvRows[1:]


def showImgOnPlot(imgTemp, left, top, width, height, data_dir, onlyFileName, score, batchIteration):
    if (score * 100 > 94):
        #print('\n\nINSIDE')  # ToDo heatMap2 in new line
        #print('x{}_y{}'.format(left, top), end=";")  # ToDo heatMap2
        #print('{:.0f}'.format(score * 100), end=";")  # ToDo heatMap2
        imgCroppedToCup = np.transpose(UnNormalize(imgTemp).cpu(), (1, 2, 0))
        figure, axis = plt.subplots(1)
        # Display the image
        axis.imshow(imgCroppedToCup)
        axis.text(left, top, '{:.2f}'.format(score * 100), fontsize=10, color='lime')
        # Create a Rectangle patch
        # rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='cyan', facecolor='none')
        rect = patches.Rectangle((left, top), width, height, linewidth=1, edgecolor='lime', facecolor='none')
        # plt.axis('off')
        # Add the patch to the Axes
        axis.add_patch(rect)
        # plt.show()
        #plt.savefig('{}\detected\{}_D.png'.format(data_dir, onlyFileName)) #Todo ForMaui
        #plt.savefig('{}\detected_Moana_bigger\{}_D.png'.format(data_dir, onlyFileName))  # Todo ForMoana
        plt.savefig('{}\prog_92_transposed\{}_{}.png'.format(data_dir, onlyFileName,batchIteration))  # Todo ForMoana
        plt.close()


def checkMax(maxName, maxVal, inputName, inputVal, label, idx2, xBest, yBest, xTemp, yTemp):
    if (maxVal < inputVal) & (idx2 == 1) & (inputVal * 100 > 60):
        maxVal = inputVal
        maxName = inputName
        xBest = xTemp
        yBest = yTemp
    return (maxName, maxVal, xBest, yBest)


def validateLabelsMap(model, tfms):
    labels_map = json.load(open('labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]
    img1 = Image.open('jpgImages/pandaSiedzi.jpg')
    img1 = tfms(img1).unsqueeze(0).cuda()
    # Classify with EfficientNet
    model.eval()
    with torch.no_grad():
        logits1 = model(img1)
        # preds1 = torch.topk(logits1, k=2).indices.squeeze(0).tolist()
        preds1 = torch.topk(logits1, k=2)[1].squeeze(0).tolist()
        print('-----')
        for idx in preds1:
            label = labels_map[idx]
            prob = torch.softmax(logits1, dim=1)[0, idx].item()
            print('{:<75} ({:.2f}%)'.format(label, prob * 100))


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, paths in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def saveTrainedModel(model, name):
    torch.save(model.state_dict(), "savedModel/" + name + "tuned_1000_5epoch.pth")


def saveTrainedModelFT(model, name):
    torch.save(model.state_dict(), "savedModel/" + name + "tuned_1000_FT.pth")

def UnNormalize(tensor):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def measureDatasetAccuracy(dataloader, device, model):
    labels_map = json.load(open('Binary.txt'))
    labels_map = [labels_map[str(i)] for i in range(2)]
    print(labels_map)  # Todo DELETE ME
    # Classify with EfficientNet
    with torch.no_grad():
        for data, target, paths in dataloader:
            data, target = data.to(device), target.to(device)
            print(target)
            logits1 = model(data)
            index = 0
            for item in logits1:
                print(paths[index])
                index += 1
                preds1 = torch.topk(item, k=2).indices.squeeze(0).tolist()
                print(preds1)  # Todo DELETE ME
                for idx in preds1:
                    label = labels_map[idx]
                    prob = torch.softmax(item, dim=0)[idx].item()
                    print('{:<75} ({:.2f}%)'.format(label, prob * 100))
                print('-----')


if __name__ == '__main__':
    main()
