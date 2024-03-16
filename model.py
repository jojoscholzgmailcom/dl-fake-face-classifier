import torch
from torch import nn
import torch.optim as optim
from data_extraction import FakeRealFaceDataset
from torch.utils.data import DataLoader

class FaceClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5),
            nn.MaxPool2d(kernel_size = 5, stride = 2),
            nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 5),
            nn.MaxPool2d(kernel_size = 5, stride = 4),
            nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 5),
            nn.MaxPool2d(kernel_size = 5, stride = 4),
            nn.Flatten(),
            nn.Linear(288, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=1)
        )
        self.lossFunction = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        return self.layers(x)

    def train(self, epochs, trainDataLoader):
        for epoch in range(epochs):
            cumulative_loss = 0.0
            for i, data in enumerate(trainDataLoader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.lossFunction(outputs, labels)
                loss.backward()
                self.optimizer.step()
                cumulative_loss += loss.item()
            print(f'{epoch + 1} loss: {cumulative_loss:.3f}')

    def test(self, testDataLoader):
        correct = 0
        total = 0
        for i, data in enumerate(testDataLoader, 0):
            total += 1
            inputs, labels = data

            outputs = self(inputs)
            if torch.equal(torch.argmax(outputs), torch.argmax(labels)):
                correct += 1
        print(f'test accuracy: {(correct / total):.3f}')

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

if __name__ == "__main__":
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
    )
    print(f"Using {device} device")
    model = FaceClassifier().to(device)
    print(model)

    training_data = FakeRealFaceDataset("train.csv", "real_vs_fake/real-vs-fake", device)
    train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)

    model.train(20, train_dataloader)

    model.save("model-parameters.pt")

    test_data = FakeRealFaceDataset("test.csv", "real_vs_fake/real-vs-fake", device)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True)
    model.test(test_dataloader)