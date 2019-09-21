import torchvision
import numpy as np
from matplotlib import pyplot as plt


def saveImagesFromTensor(images, labels):
    narrowed_8x2 = images.narrow(0, 0, 16)
    localBatchSize = narrowed_8x2.size()[0]
    torchvision.utils.save_image(narrowed_8x2, 'imagesFromTensor/merged{}.png'.format("_2x8"))  # Todo -- zapisywalo 8x2

    for i in range(localBatchSize):
        torchvision.utils.save_image(images[i, :, :, :], 'imagesFromTensor/{}.png'.format(labels[i].item()))


def showPlotImages(images):
    narrowed_8x2 = images.narrow(0, 0, 16)
    showImagesOnGrid_8xY(torchvision.utils.make_grid(narrowed_8x2))  # ToDo -- dziala wyswietlanie 8x2
    narrowed_1x1 = images.narrow(0, 0, 1)
    showImagesOnGrid_8xY(torchvision.utils.make_grid(narrowed_1x1))  # ToDo -- dziala wyswietlanie 1x1


def showImagesOnGrid_8xY(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # Todo -- uncomment below to get rid of RGB warning
    # plt.imshow((np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8))
    plt.show()
