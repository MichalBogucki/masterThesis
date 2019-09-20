from __future__ import print_function
import torch.optim as optim

from NetworkArchitecture import NetworkArchitecture
from setParserArguments import setParserArguments
from testModel import *
from timer import *
from view_classify import *

start_time = datetime.now()


def main():
    # Training settings
    args = setParserArguments()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = NetworkArchitecture().to(device)
    model.load_state_dict(torch.load("savedModel/mnist_cnn.pt"))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        images, labels = next(iter(test_loader))
        img = images[0].view(1, 784)
        view_classify(img.view(1, 28, 28), test_loader, epoch)
        #trainModel(args, model, device, train_loader, optimizer, epoch)
        testModel(args, model, device, test_loader, epoch)

    #if (args.save_model):
    #    torch.save(model.state_dict(), "savedModel/mnist_cnn.pt")

    timer()


if __name__ == '__main__':
    main()






