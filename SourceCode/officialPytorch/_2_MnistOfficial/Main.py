from __future__ import print_function
import torch.optim as optim
from networkArchitecture import *
from useImages import *
from setLoader import setLoader
from setParserArguments import setParserArgumentsMnist
from trainModel import *
from testModel import *
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

    model = NetworkArchitecture()#.to(device)
    loadTrainedModel(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    #----------
    from torch.utils.tensorboard import SummaryWriter
    import matplotlib.pyplot as plt

    # helper function to show an image
    # (used in the `plot_classes_preds` function below)
    def matplotlib_imshow(img, one_channel=False):
        if one_channel:
            img = img.mean(dim=0)
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        if one_channel:
            plt.imshow(npimg, cmap="Greys")
        else:
            plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')
    # get some random training images
    dataiter = iter(trainLoader)
    images, labels = dataiter.next()

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images
    matplotlib_imshow(img_grid, one_channel=True)

    # write to tensorboard
    writer.add_image('four_fashion_mnist_images', img_grid)

    # get some random training images
    dataiter = iter(trainLoader)
    images, labels = dataiter.next()

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images
    matplotlib_imshow(img_grid, one_channel=True)

    writer.add_graph(model, images)
    writer.close()
    # write to tensorboard
    writer.add_image('four_fashion_mnist_images', img_grid)

    # ----------

    #for epoch in range(1, args.epochs + 1):
        # images, labels = next(iter(testLoader)) # ToDo -- na laptopie wyswietla ploty poprawnie
        # showPlotImages(images)  # ToDo -- Showing plot stops code execution until the figure will be closed
        # saveImagesFromTensor(images, labels)
        # trainModel(args, model, device, trainLoader, optimizer, epoch)
        #testModel(args, model, device, testLoader, epoch)

    # saveTrainedModel(args, model)

    showExecutionTime(startTime)
    #https://github.com/lukemelas/EfficientNet-PyTorch



if __name__ == '__main__':
    main()
