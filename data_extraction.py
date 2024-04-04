import os
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms import v2
import torch
from torch.nn.functional import one_hot

class FakeRealFaceDataset():

    def __init__(self, labelsFile, imgDir, device = "cpu"):
        self.imgDir = imgDir
        self.labels = pd.read_csv(labelsFile)
        self.lenLabels = len(self.labels)
        self.transforms = v2.Compose([v2.ToDtype(torch.float32, scale=True)])
        self.device = device

    def __len__(self):
        return self.lenLabels

    def __getitem__(self, idx):
        img_path = os.path.join(self.imgDir, self.labels.iloc[idx, 5])
        image = self.transforms(read_image(img_path)).to(self.device)
        label = self.labels.iloc[idx, 3]
        label_tensor = torch.zeros(2).to(self.device)
        label_tensor[label] = 1
        return image, label_tensor

if __name__ == "__main__":
    dataset = FakeRealFaceDataset("train.csv", "real_vs_fake/real-vs-fake")
    image, label = dataset.__getitem__(0)
    print(image, label) 
