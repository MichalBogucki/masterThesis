import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

from datetime import datetime

from matplotlib import pyplot as plt

from showExecutionTime import showExecutionTime
from gluoncv import model_zoo, data, utils
import mxnet as mx
import numpy as np
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
import glob
import torch

def UseGluonCv_YOLO3():
    ##---------------Gluon CV----------------

    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    ctx = mx.gpu(0)
    net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True, ctx=ctx)
    #net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True) #CPU

    #pathDir = 'example'
    pathDir = 'jpgImages\\video'
    startTime = datetime.now()
    result = glob.glob(pathDir+'\\frames' + '\**\*.jpg', recursive=True)
    imageTensor2, imgToDisplay2 = data.transforms.presets.yolo.load_test(result, short=800)
    #imList = mx.nd.array(imageTensor2)
    #print(type(imList))
    print('len: ', len(result))
    f = open('{}\detected.csv'.format(pathDir), "a")
    f.write("fileName;probScore;xTopLeft;yTopLeft;xDownRight;yDownRight;\n")
    f.close()
    for iterIndex, resultItem in enumerate(result):
        processOneImage(ctx, imageTensor2, imgToDisplay2, iterIndex, net, resultItem, pathDir)

    print('memory ', mx.context.gpu_memory_info())
    print('len: ', len(result))
    showExecutionTime(startTime)


def processOneImage(ctx, imageTensor2, imgToDisplay2, iterIndex, net, resultItem, pathDir):
    #if (iterIndex % 10 == 0):
    #    mx.gpu(0).empty_cache()
    print()
    print('memory ', mx.context.gpu_memory_info())
    print('GPU resultItem: ', resultItem)
    # DiskLocation - C:\Users\Michał\.mxnet\models
    threshold = 0.2
    # imageTensor, imgToDisplay = data.transforms.presets.yolo.load_test('Lotr2.jpg', short=1024)
    # imageTensor, imgToDisplay = data.transforms.presets.yolo.load_test(resultItem)#, short=1024)
    # print('imageTensor.shape', imageTensor.shape)
    # print('imgToDisplay.shape', imgToDisplay.shape)
    # print('coco classes: ', net.classes)
    net.reset_class(classes=['person'], reuse_weights=['person'])
    # now net has 2 classes as desired
    # print('new classes: ', net.classes)
    # print('Shape of pre-processed image:', imageTensor.shape)
    imageTensor2[iterIndex] = imageTensor2[iterIndex].as_in_context(ctx)  # Todo MOVE TO GPU
    class_IDs, scores, bounding_boxs = net(imageTensor2[iterIndex])
    thresholdIndex = next(x[0] for x in enumerate(scores[0]) if x[1] < threshold)
    filteredScores = scores[0][:thresholdIndex]
    #print('len maax', len(imageTensor2))
    if (len(filteredScores)==0):
        imageTensor2[iterIndex] = []
        #mx.gpu(0).empty_cache()
        return
    filteredClassIds = class_IDs[0][:thresholdIndex]
    filteredBoundingBoxes = bounding_boxs[0][:thresholdIndex]
    #print('filteredScores: ', filteredScores)
    #print('boundingBoxes: ', filteredBoundingBoxes)
    # # print(net.classes)
    #ax = utils.viz.plot_bbox(imgToDisplay2[iterIndex], filteredBoundingBoxes, filteredScores,
    #           filteredClassIds, class_names=net.classes, thresh=threshold)
    #plt.savefig('{}\detected_frames\plot_{}.jpg'.format(pathDir,iterIndex))
    #plt.close()

    #print('saved')
    imageTensor2[iterIndex] = []
    #mx.gpu(0).empty_cache()
    f = open('{}\detected.csv'.format(pathDir), "a")
    #f.write("fileName;probScore;xTopLeft;yTopLeft;xDownRight;yDownRight;\n")
    for i in range(thresholdIndex):
        f.write('{};{:.2f};{:.2f};{:.2f};{:.2f};{:.2f};\n'.format(str(resultItem) , filteredScores[i].asnumpy().item(0), filteredBoundingBoxes[i][0].asnumpy().item(0), filteredBoundingBoxes[i][1].asnumpy().item(0), filteredBoundingBoxes[i][2].asnumpy().item(0),filteredBoundingBoxes[i][3].asnumpy().item(0)))
    f.close()
    return

    # plt.show()
##---------------Gluon CV----------------
