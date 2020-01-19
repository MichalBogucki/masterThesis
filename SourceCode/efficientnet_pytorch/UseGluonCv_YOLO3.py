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

def UseGluonCv_YOLO3():
    ##---------------Gluon CV----------------
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    ctx = mx.gpu(0)
    net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True, ctx=ctx)
    #net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True) #CPU

    startTime = datetime.now()
    result = glob.glob('example' + '/**/*.jpg', recursive=True)
    iter = 0
    imageTensor2, imgToDisplay2 = data.transforms.presets.yolo.load_test(result)
    #imList = mx.nd.array(imageTensor2)
    #print(type(imList))
    print('len: ', len(result))
    #for item in range(len(result)):
    #    imageTensor2[item] = imageTensor2[item].as_in_context(ctx)  # Todo MOVE TO GPU

    for resultItem in result:
        print('GPU resultItem: ', resultItem)
        # DiskLocation - C:\Users\Michał\.mxnet\models
        threshold = 0.2

        #imageTensor, imgToDisplay = data.transforms.presets.yolo.load_test('Lotr2.jpg', short=1024)
        #imageTensor, imgToDisplay = data.transforms.presets.yolo.load_test(resultItem)#, short=1024)

        #print('imageTensor.shape', imageTensor.shape)
        #print('imgToDisplay.shape', imgToDisplay.shape)
        #print('imageTensor2.shape', imageTensor2[iter].shape)
        #print('imgToDisplay2.shape', imgToDisplay2[iter].shape)
        # print('coco classes: ', net.classes)
        net.reset_class(classes=['person'], reuse_weights=['person'])
        # now net has 2 classes as desired
        # print('new classes: ', net.classes)
        #print('Shape of pre-processed image:', imageTensor.shape)
        imageTensor2[iter] = imageTensor2[iter].as_in_context(ctx) #Todo MOVE TO GPU
        class_IDs, scores, bounding_boxs = net(imageTensor2[iter])

        thresholdIndex = next(x[0] for x in enumerate(scores[0]) if x[1] < threshold)
        filteredScores = scores[0][:thresholdIndex]
        if (len(filteredScores)==0):
            iter += 1
            continue
        filteredClassIds = class_IDs[0][:thresholdIndex]
        filteredBoundingBoxes = bounding_boxs[0][:thresholdIndex]
        #print(filteredScores)
        #print(filteredBoundingBoxes)
        # # print(net.classes)
        ax = utils.viz.plot_bbox(imgToDisplay2[iter], filteredBoundingBoxes, filteredScores,
                    filteredClassIds, class_names=net.classes, thresh=threshold)

        plt.savefig('plot_{}.jpg'.format(iter))
        plt.close()
        print('saved')
        iter += 1
        #plt.show()


    print('len: ', len(result))
    showExecutionTime(startTime)
##---------------Gluon CV----------------
