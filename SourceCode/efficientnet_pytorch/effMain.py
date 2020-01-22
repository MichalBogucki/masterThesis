from __future__ import print_function
import torch
import time
import copy
import torch.optim as optim
import torch.nn as nn
import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

from performAugmentation import performAugmentation
from UseGluonCv_YOLO3 import UseGluonCv_YOLO3

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

from ImageFolderWithPaths import ImageFolderWithPaths
# from ModelBinaryLayerAfterFC import ModelBinaryLayerAfterFC
from setParserArguments import setParserArgumentsMnist
from showExecutionTime import *
from modelSourceCode import EfficientNet
import torchvision
import json
import PIL
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np

def main():

    ##---------------Gluon CV----------------
    #UseGluonCv_YOLO3()
    #return
    #return
    ##---------------Gluon CV----------------

    startTime = datetime.now()

    # performAugmentation()

    # Training settings
    device = torch.device("cuda")

    modelName = 'efficientnet-b0tuned'
    # modelName = 'efficientnet-b4tuned'

    #imageSize = EfficientNet.get_image_size(modelName)
    imageSize = 576
    #print("imgSize " + str(imageSize))

    # Number of classes in the dataset
    num_PreLoad_Classes = 2
    num_tunedClasses = 2
    # Batch size for training (change depending on how much memory you have)
    batch_size = 1
    # Number of epochs to train for
    num_epochs = 4

    model = EfficientNet.pretrained(modelName, num_classes=num_PreLoad_Classes, tuned_classes=num_tunedClasses).cuda()
    model.eval()

    # ----------
    # for epoch in range(1, args.epochs + 1):

    # Preprocess image
    tfms = transforms.Compose([
        transforms.Resize(size=imageSize, interpolation=PIL.Image.BICUBIC),
        #transforms.CenterCrop(imageSize),
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

    #data_dir = "jpgImages/aug/val"#/tiny"
    #data_dir = "jpgImages/aug/train"
    data_dir = "UseMeMaui"

    # EXAMPLE USAGE:
    # instantiate the dataset and dataloader

    dataset = ImageFolderWithPaths(data_dir, transform=tfms)  # our custom dataset
    datasetSize = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=0, pin_memory=True)

    # return visualizeGraphWithOnnxToNetron(model) #ToDo ONYXX vizualiser

    # ---Binary---
    startTime = datetime.now()  # Todo deleteME
    validateBinaryTatoo(dataloader, device, model)  # Todo UNCOMMENT ME
    return

    showExecutionTime(startTime)  # Todo deleteME
    if ('tuned' in modelName):
        return
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
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val']}

    # Train and evaluate
    model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer_ft, device, num_epochs=num_epochs)
    # +++++ Train Time ++++++++

    # compareImages(model, nameViews, tfms)
    # compareImages(model_ft, nameImg, tfms)
    # ---------------------------------

    # ---Binary---
    validateBinaryTatoo(dataloader, device, model_ft)  # Todo UNCOMMENT ME
    # ---Binary---

    # ---labels_map---
    # validateLabelsMap(model_ft, tfms)
    # ---labels_map---

    # saveTrainedModel(model, modelName) #Todo UNCOMMENT ME

    showExecutionTime(startTime)


def validateBinaryTatoo(dataloader, device, model):
    labels_map = json.load(open('Binary.txt'))
    labels_map = [labels_map[str(i)] for i in range(2)]
    # print(labels_map) #Todo DELETE ME
    # Classify with EfficientNet
    maxName = ''
    maxVal = 0
    model.eval()
    with torch.no_grad():
        batchIteration = 1
        batchesNumber = (len(dataloader))
        for batchData, target, paths in dataloader:
            batchData, target = batchData.to(device), target.to(device)
            print('batchData.shape')
            print(batchData.shape)
            print(type(batchData))
            print('target')
            print(target)
            print('target.shape: {}'.format(target.shape))
            logits1 = model(batchData)
            index = 0
            print('----- BATCH {}/{}-----'.format(batchIteration, batchesNumber))
            batchIteration += 1
            for item in logits1:
                actFileName = paths[index]
                imgTemp = batchData[index]
                print('batchData[index]shape: {}'.format(imgTemp.shape))
                #ime = imgTemp.crop((left, top, right, bottom))
                left, top, right, bottom = 0,0,224,128
                imgCropped = imgTemp[:, top:bottom, left:right]
                print('imgCropped: {}'.format(imgCropped.shape))
                print('img.size(0)={}, img.size(1)={}img.size(2)={} '.format(imgTemp.size(0),imgTemp.size(1),imgTemp.size(2)))
                print('{}/{} "{}"'.format(index+1, len(logits1), actFileName))
                foldSize = 90
                step = 30
                startTime = datetime.now() #todo delete ME --time--
                ############### ---------- folding images into SmallerImages ---------- ##############
                imgTempPatches2x2x3x50x50 = imgTemp.unfold(0, 3, 3).unfold(1, foldSize, step).unfold(2, foldSize, step).squeeze(0)
                print('imgTempPatches2x2x3x50x50.shape: {}'.format(imgTempPatches2x2x3x50x50.shape))
                flate2 = torch.flatten(imgTempPatches2x2x3x50x50, 0, 1)
                print('flat4x3x25x25.shape: {}'.format(flate2.shape))
                print('flat[0].shape: {}'.format(flate2[0].shape))
                #100x100; 200x100; 300x100; 400x100
                # ++++++++++ put flatted smallImages into NN +++++++++++++ #
                nnOutput = model(flate2)
                x2 = foldSize
                y2 = foldSize
                for item2 in nnOutput:
                    predictedClasses2 = torch.topk(item2, k=2).indices.squeeze(0).tolist()
                    if (x2 > imgTemp.size(2)):
                        x2 = foldSize
                        y2 = (y2 + step)
                    for idx2 in predictedClasses2:
                        label2 = labels_map[idx2]
                        prob2 = torch.softmax(item2, dim=0)[idx2].item()
                        tempName = '{} ({}) x={} y={} x2={} y2={}'.format(label2,prob2*100,(x2-foldSize), (y2-foldSize),x2,y2)
                        print('{} x={} y={} {:<75} ({:.2f}%)'.format(label2,x2, y2, label2, prob2 * 100))
                        maxName, maxVal = checkMax(maxName, maxVal, tempName, prob2, label2, idx2)
                    x2 = (x2 + step)
                # ++++++++++ put flatted smallImages into NN +++++++++++++ #
                showExecutionTime(startTime) #todo delete ME --time--
                print('maxName: "{}"'.format(maxName))
                print('maxVal: ({:.2f}%)'.format(maxVal * 100))
                #imgFromTensor = np.transpose(flate2[0].cpu(), (1,2,0))
                #plt.imshow((imgFromTensor.astype('uint8')), interpolation='bicubic')
                #transforms.ToPILImage()(transforms.ToTensor()(image)), interpolation = "bicubic"
                #plt.show()
                #torchvision.utils.save_image(imgFromTensor, 'flate2.png')

                ############### ---------- folding images into SmallerImages ---------- ##############

                print('savedImage')
                index += 1
                predictedClasses = torch.topk(item, k=2).indices.squeeze(0).tolist()
                #print(predictedClasses)  # Todo DELETE ME
                for idx in predictedClasses:
                    label = labels_map[idx]
                    prob = torch.softmax(item, dim=0)[idx].item()
                    print('{:<75} ({:.2f}%)'.format(label, prob * 100))
                    #maxName, maxVal = checkMax(maxName, maxVal, actFileName, prob, label, idx)
                    return
                print('-----')
            print('maxName: "{}"'.format(maxName))
            print('maxVal: ({:.2f}%)'.format(maxVal * 100))


def checkMax(maxName, maxVal, inputName, inputVal, label, idx2):
    if (maxVal < inputVal) & (idx2 == 1):
        maxVal = inputVal
        maxName = inputName
    return (maxName, maxVal)


def validateLabelsMap(model, tfms):
    labels_map = json.load(open('labels_map.txt'))
    labels_map = [labels_map[str(i)] for i in range(1000)]
    img1 = Image.open('jpgImages/pandaSiedzi.jpg')
    img1 = tfms(img1).unsqueeze(0).cuda()
    # Classify with EfficientNet
    model.eval()
    with torch.no_grad():
        logits1 = model(img1)
        preds1 = torch.topk(logits1, k=2).indices.squeeze(0).tolist()
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
    torch.save(model.state_dict(), "savedModel/" + name + "tuned_1000.pth")


if __name__ == '__main__':
    main()
