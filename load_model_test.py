import torch
from torch import nn
import torch.optim as optim
from data_extraction import FakeRealFaceDataset
from torch.utils.data import DataLoader
from model import FaceClassifier

if __name__ == "__main__":
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
    )
    print(f"Using {device} device")
    model = FaceClassifier().to(device)
    print(model)

    model.load("model-parameters.pt")

    test_data = FakeRealFaceDataset("test.csv", "real_vs_fake/real-vs-fake", device)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True)
    model.test(test_dataloader)