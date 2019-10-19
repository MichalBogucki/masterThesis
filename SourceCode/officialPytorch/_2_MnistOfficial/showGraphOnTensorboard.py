import numpy
import torchvision


def showGraphOnTensorboard(model, trainLoader):
    # ----------
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