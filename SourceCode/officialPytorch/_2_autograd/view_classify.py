import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torchvision import datasets, transforms


def view_classify(img, test_loader, epoch):
    ''' Function for viewing an image and it's predicted classes.
    '''
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    print('image')
    # As PIL.Image
    dataset = datasets.MNIST(root='../data/')
    x, _ = dataset[7776 + epoch]
    x.show()  # x is a PIL.Image here ToDo - to dziala

    # As torch.Tensor
    dataset = datasets.MNIST(
        root='../data/',
        transform=transforms.ToTensor()
    )

    def show_batch(batch):
        im = torchvision.utils.make_grid(batch)
        plt.imshow(np.transpose(im.numpy(), (1, 2, 0)).astype(np.uint8))

    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    # print('Labels: ', labels)
    # print('Batch shape: ', images.size())
    show_batch(images)

    x, _ = dataset[7777]  # x is now a torch.Tensor
    print('image2')