from collections import deque

import torch
from PIL import Image
from torch.nn import functional as F


def compareImages(model, nameViews, tfms):
    with torch.no_grad():
        views = []
        for view in nameViews:
            views.append(Image.open("jpgImages/"+view))
        featureList = []
        lossList = []
        for item in views:
            item = tfms(item).unsqueeze(0).cuda()
            feature = model.extract_features(item)
            featureList.append(feature)
            del item, feature
            torch.cuda.empty_cache()
        for i in range(len(featureList)):
            for j in range(len(deque(featureList))):
                if (i >= j):
                    continue
                lossFun = F.mse_loss(featureList[i], featureList[j], reduction='mean').item()
                lossList.append(lossFun)
                print(str(nameViews[i]) + " : " + str(nameViews[j]) + " == " + str(lossFun))

    print("max: " + str(max(lossList)))
    print("min: " + str(min(lossList)) + "\n")