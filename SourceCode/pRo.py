def printLoss(y, yPred, t):
    # Compute and print loss
    loss = (yPred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)