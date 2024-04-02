import itertools
import torch
from torch import nn
import torch.optim as optim
from data_extraction import FakeRealFaceDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
 
class FaceClassifier(nn.Module):

    def __init__(self, conv_layers: int, starting_channels: int, channel_multiplier: float, kernelSize: int, poolKernel: int, poolStrides: list, hiddenNodes: int, image_size = 256):
        super().__init__()
        currentImageSize = image_size
        layersList = []
        last_channels = 3
        for i in range(conv_layers):
            layersList.append(nn.Conv2d(in_channels = last_channels, out_channels = int(starting_channels), kernel_size = kernelSize))
            currentImageSize -= (kernelSize - 1)
            last_channels = starting_channels
            starting_channels = int(starting_channels * channel_multiplier)
            layersList.append(nn.ReLU())
            layersList.append(nn.MaxPool2d(kernel_size = poolKernel, stride = poolStrides[i]))
            currentImageSize = int((currentImageSize - poolKernel ) / poolStrides[i] + 1)
        
        layersList.append(nn.Flatten())
        layersList.append(nn.Linear(((currentImageSize**2) * last_channels), hiddenNodes))
        layersList.append(nn.ReLU())
        layersList.append(nn.Linear(hiddenNodes, 2))
        layersList.append(nn.Softmax(dim=1))
        self.layers = nn.Sequential(*layersList)
        #self.layers = nn.Sequential(
        #    nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size = 5, stride = 2),
        #    nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 5),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size = 5, stride = 4),
        #    nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 5),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size = 5, stride = 4),
        #    nn.Flatten(),
        #    nn.Linear(288, 16),
        #    nn.ReLU(),
        #    nn.Linear(16, 2),
        #    nn.Softmax(dim=1)
        #)
        self.lossFunction = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters())


    def forward(self, x):
        return self.layers(x)


    def train(self, epochs, trainDataLoader, testDataloader = None):
        cummulative_losses_training = []
        cummulative_losses_test = []
        for epoch in range(epochs):
            cumulative_loss_training = 0.0
            for i, data in enumerate(trainDataLoader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.lossFunction(outputs, labels)
                loss.backward()
                self.optimizer.step()
                cumulative_loss_training += loss.item()
            print(f'{epoch + 1} loss: {cumulative_loss_training:.3f}')
            cummulative_losses_training.append(cumulative_loss_training)
            
            if testDataloader is not None:
                cumulative_loss_test = 0.0
                for i, data_test in enumerate(testDataloader, 0):
                    inputs_test, labels_test = data_test
                    outputs_test = self(inputs_test)
                    outputs_test = self(inputs_test)
                    loss_test = self.lossFunction(outputs_test, labels_test)
                    #print(f'test loss: {loss.item():.3f}')
                    #accuracy = self.test(testDataloader)
                    cumulative_loss_test += loss_test.item()
                cummulative_losses_test.append(cumulative_loss_test)
            else:
                continue
            
        return cummulative_losses_training, cummulative_losses_test if testDataloader is not None else cummulative_losses_training


    def test(self, testDataLoader):
        correct = 0
        total = 0
        for i, data in enumerate(testDataLoader, 0):
            
            inputs, labels = data
            total += len(labels)

            outputs = self(inputs)
            equal_tensor = torch.eq(torch.argmax(outputs, dim = 1), torch.argmax(labels, dim = 1))
            for bool_val in equal_tensor:
                if bool_val:
                    correct += 1
        accuracy = (correct / total)
        print(f'test accuracy: {accuracy:.3f}')
        return accuracy
    
    
    def evaluate(self, evaluationDataLoader):
        
        true_labels = []
        predicted_labels = []
        
        for i, data in enumerate(evaluationDataLoader, 0):
            
            inputs, labels = data

            outputs = self(inputs)
            
            predicted_labels.extend(torch.argmax(outputs, dim = 1).tolist())
            true_labels.extend(torch.argmax(labels, dim = 1).tolist())
        
        matrix_confusion = confusion_matrix(true_labels, predicted_labels)
        classification_results = classification_report(true_labels, predicted_labels)
        
        accuracy = classification_results.split("\n")[-3].split()[-2]
        precision, recall, f1_score, support = classification_results.split("\n")[-2].split()[2:]
                
        return accuracy, precision, recall, f1_score, support, matrix_confusion
        
        
    def plot_confusion_matrix(self, matrix_confusion):
        sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', cbar=True, xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion matrix')
        plt.savefig('confusion_matrix.png')
        
        
    def plot_loss(self, cummulative_losses_training, cummulative_losses_test):
        sns.lineplot(x=range(len(cummulative_losses_training)), y=cummulative_losses_training, label='Training loss')
        sns.lineplot(x=range(len(cummulative_losses_test)), y=cummulative_losses_test, label='Test loss')
        plt.title('Training and test loss over ' + str(len(cummulative_losses_training)) + ' epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss.png')


    def save(self, path):
        torch.save(self.state_dict(), path)


    def load(self, path):
        self.load_state_dict(torch.load(path))


def grid_hyperparameter_search(device, train_dataloader, test_dataloader):
    conv_layers = [3, 4]
    starting_channels = [32, 64]
    channel_multiplier = [0.5]
    kernel_size = [3, 5]
    poolKernel = [3, 5]
    poolStrides = [[2,2,2,2], [4,4,4,4]]
    hiddenNodes = [32]
    products = itertools.product(conv_layers, starting_channels, channel_multiplier, kernel_size, poolKernel, poolStrides, hiddenNodes)
    best_accuracy = 0
    for configuration in products:
        print(f"Current: model-parameters-{configuration[0]}-{configuration[1]}-{configuration[2]}-{configuration[3]}-{configuration[4]}-{configuration[5][0]}-{configuration[6]}")

        model = FaceClassifier(configuration[0], configuration[1], configuration[2], configuration[3], configuration[4], configuration[5], configuration[6]).to(device)
        model.train(20, train_dataloader)
        model.save(f"model/model-parameters-{configuration[0]}-{configuration[1]}-{configuration[2]}-{configuration[3]}-{configuration[4]}-{configuration[5][0]}-{configuration[6]}.pt")
        accuracy = model.test(test_dataloader)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"Best: model-parameters-{configuration[0]}-{configuration[1]}-{configuration[2]}-{configuration[3]}-{configuration[4]}-{configuration[5][0]}-{configuration[6]}   acc: {accuracy}")
        break


if __name__ == "__main__":
    
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
    )
    print(f"Using {device} device")
    #model = FaceClassifier().to(device)
    #print(model)

    training_data = FakeRealFaceDataset("train.csv", "real_vs_fake/real-vs-fake", device)
    train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)

    #model.train(20, train_dataloader)

    #model.save("model-parameters.pt")

    test_data = FakeRealFaceDataset("test.csv", "real_vs_fake/real-vs-fake", device)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True)

    grid_hyperparameter_search(device, train_dataloader, test_dataloader)
    
    #model.test(test_dataloader)