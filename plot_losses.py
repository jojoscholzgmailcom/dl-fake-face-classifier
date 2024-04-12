import seaborn as sns
import matplotlib.pyplot as plt
import re

def plot_loss(train_losses, test_losses):
    epochs = len(train_losses)
    sns.lineplot(x=range(epochs), y=train_losses, label='Training loss')
    sns.lineplot(x=range(epochs), y=test_losses, label='Test loss')
    plt.title('Training and test loss over ' + str(epochs) + ' epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('img/loss_plot.png')
    
        
def read_losses(file):
    train_losses = []
    test_losses = []
    with open(file, 'r') as f:
        for line in f:
            match_training = re.findall(r"train mean loss: (\d+\.\d+)", line)
            train_losses += [float(match) for match in match_training]
            match_testing = re.findall(r"test mean loss: (\d+\.\d+)", line)
            test_losses += [float(match) for match in match_testing]
    return train_losses, test_losses


if __name__ == "__main__":
    train_losses, test_losses = read_losses('final-model-logs.txt')
    plot_loss(train_losses, test_losses)