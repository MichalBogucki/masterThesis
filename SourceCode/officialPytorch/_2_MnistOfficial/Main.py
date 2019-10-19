from __future__ import print_function
import torch.optim as optim
from networkArchitecture import *
from showGraphOnTensorboard import showGraphOnTensorboard
from setLoader import setLoader
from setParserArguments import setParserArgumentsMnist
from testModel import *
from trainModel import *
from useImages import *
from showExecutionTime import *
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_name('efficientnet-b0')

def main():
    startTime = datetime.now()

    # Training settings
    args = setParserArgumentsMnist()

    useCuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if useCuda else "cpu")
    kwargs = {'num_workers': 3, 'pin_memory': True} if useCuda else {}

    trainLoader = setLoader(kwargs, shouldTrain=True, shouldDownload=False, batchSize=args.batch_size)
    testLoader = setLoader(kwargs, shouldTrain=False, shouldDownload=False, batchSize=args.test_batch_size)

    model = NetworkArchitecture().to(device)
    loadTrainedModel(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    #showGraphOnTensorboard(model, trainLoader)

    for epoch in range(1, args.epochs + 1):
        images, labels = next(iter(testLoader)) # ToDo -- na laptopie wyswietla ploty poprawnie
        # showPlotImages(images)  # ToDo -- Showing plot stops code execution until the figure will be closed
        # saveImagesFromTensor(images, labels)
        # trainModel(args, model, device, trainLoader, optimizer, epoch)
        testModel(args, model, device, testLoader, epoch)

    # saveTrainedModel(args, model)

    showExecutionTime(startTime)
    #https://github.com/lukemelas/EfficientNet-PyTorch


if __name__ == '__main__':
    main()
