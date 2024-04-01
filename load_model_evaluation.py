import torch
from torch import nn
import torch.optim as optim
from data_extraction import FakeRealFaceDataset
from torch.utils.data import DataLoader
from model import FaceClassifier


if __name__ == '__main__':
    
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
    )
    print(f"Using {device} device")  
    
    model = FaceClassifier().to(device)
    model.load("model-parameters.pt")
    
    # model = FaceClassifier(3, 32, 0.5, 3, 3, [2,2,2,2], 32).to(device)
    # training_data = FakeRealFaceDataset("train.csv", "real_vs_fake/real-vs-fake", device)
    # train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
    # model.train(1, train_dataloader)
    
    evaluation_data = FakeRealFaceDataset("valid.csv", "real_vs_fake/real-vs-fake", device)
    evaluation_dataloader = DataLoader(evaluation_data, batch_size=128, shuffle=True)
    accuracy, precision, recall, f1_score, support, matrix_confusion = model.evaluate(evaluation_dataloader)
    
    # print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}, Support: {support}")
    
    model.plot_confusion_matrix(matrix_confusion)