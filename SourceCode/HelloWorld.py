from __future__ import print_function
import torch

from SourceCode.pRo import printLoss

dtype = torch.float
device = torch.device("cuda:0")

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, DIn, H, DOut = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, DIn, device=device, dtype=dtype)
y = torch.randn(N, DOut, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(DIn, H, device=device, dtype=dtype)
w2 = torch.randn(H, DOut, device=device, dtype=dtype)

learningRate = 1e-6

for t in range(500):
    #Forward pass: compute predicted y
    h = x.mm(w1)
    hRelu = h.clamp(min=0)
    yPred = hRelu.mm(w2)

    printLoss(y, yPred, t)

    # BackProp to compute gradients of w1 and w2 with respect to loss
    gradYPred = 2.0 * (yPred - y)
    gradW2 = hRelu.t().mm(gradYPred)
    gradHRelu = gradYPred.mm(w2.t())
    gradH = gradHRelu.clone()
    gradH[h < 0] = 0
    gradW1 = x.t().mm(gradH)

    # Update weights using gradient descent
    w1 -= learningRate * gradW1
    w2 -= learningRate * gradW2
