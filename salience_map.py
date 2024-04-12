import torch
from torch import nn
import torch.optim as optim
from data_extraction import FakeRealFaceDataset
from torch.utils.data import DataLoader
from model import FaceClassifier
import matplotlib.pyplot as plt
import cv2 as cv
import os



def generateSaliencyMap(idx, test_data, model):
    inputs, labels = test_data.__getitem__(idx)
    labels_idx = labels.argmax()
    origin_inputs = torch.clone(inputs)
    inputs = torch.unsqueeze(inputs, dim=0)
    print(inputs.size())
    

    inputs.requires_grad_()

    scores = model(inputs)

    score_max_index = scores.argmax()
    score_max = scores[0,score_max_index]

    score_max.backward()

    saliency, _ = torch.max(inputs.grad.data.abs(), dim=1)

    #plt.imshow(inputs)
    #plt.imshow(saliency[0], cmap=plt.cm.hot)
    #plt.axis('off')
    #plt.show()

    plt.imsave(fname=f"img/heatmap-{idx}-{labels_idx}-{score_max_index}.png", arr = saliency[0], cmap=plt.cm.hot)
    base_img = cv.imread(os.path.join("real_vs_fake/real-vs-fake", test_data.labels.iloc[idx, 5]))
    heatmap = cv.imread(f"img/heatmap-{idx}-{labels_idx}-{score_max_index}.png")
    blended_img = cv.addWeighted(base_img, 0.4, heatmap, 0.8, 0)
    cv.imwrite(f"img/base_img-{idx}-{labels_idx}-{score_max_index}.png", base_img)
    cv.imwrite(f"img/blended_img-{idx}-{labels_idx}-{score_max_index}.png",blended_img)

if __name__ == "__main__":
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
    )

    device = "cpu"
    print(f"Using {device} device")  
    
    model = FaceClassifier(3,32,0.5,5,5,[2,2,2,2],32).to(device)
    print(model)

    model.load("model/best-model.pt")
    #stripped_model = nn.Sequential(*[model.layers[i] for i in range(13)])

    test_data = FakeRealFaceDataset("test.csv", "real_vs_fake/real-vs-fake", device)
    #test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
    

    #inputs, labels = next(iter(test_dataloader))
    #print(inputs.size())
    idx = 19000
    generateSaliencyMap(idx, test_data, model)