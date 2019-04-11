import torch
import torchvision
import matplotlib.pyplot as plt

from net import Net

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


# show the data
def show_data(test_loader):
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    i = 0
    while True:
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
        plt.show()
        input('Next image: ')
        i += 1


class Trainer:
    def __init__(self):
        self.n_epochs = 7
        self.batch_size_train = 64
        self.batch_size_test = 1000
        self.learning_rate = 0.01
        self.momentum = 0.5
        self.log_interval = 10

        self.train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,))
                                       ])),
            batch_size=self.batch_size_train, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,))
                                       ])),
            batch_size=self.batch_size_test, shuffle=True)

        self.network = Net()
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate,
                                   momentum=self.momentum)
        self.path = './model.pth'

        self.train_losses = []
        self.train_accuracies = []
        self.train_counter = []
        self.test_losses = []
        self.test_accuracies = []
        self.test_counter = [i * len(self.train_loader.dataset) for i in range(self.n_epochs + 1)]

    def train(self, epoch):
        if epoch == 4:
            self.learning_rate = 0.001
        elif epoch == 6:
            self.learning_rate = 0.0001
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
        self.network.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            correct = 0
            self.optimizer.zero_grad()
            output = self.network(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))
                self.train_losses.append(loss.item())
                self.train_accuracies.append(np.array(correct).astype(int) / 64.0)
                self.train_counter.append((batch_idx*64) + ((epoch-1)*len(self.train_loader.dataset)))
                self.save(epoch)

    def test(self, gaussian_noise=False, noise_mean=0, noise_var=1):
        self.network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                if gaussian_noise:
                    data = self.add_gaussian_noise(data, noise_mean, noise_var)
                output = self.network(data)
                test_loss += F.cross_entropy(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(self.test_loader.dataset)
        if not gaussian_noise:
            self.test_losses.append(test_loss)
            self.test_accuracies.append(np.array(correct).astype(int) / len(self.test_loader.dataset))
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(self.test_loader.dataset), 100. * correct / len(self.test_loader.dataset)))
        return test_loss

    @staticmethod
    def add_gaussian_noise(data, mean=0, var=0.1):
        normal = torch.distributions.Normal(loc=mean, scale=var)
        data += normal.sample(data.size())
        return data

    def load(self):
        try:
            checkpoint = torch.load(self.path)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.network.eval()
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint['train_losses']
            self.test_losses = checkpoint['test_losses']
            self.train_accuracies = checkpoint['train_accuracies']
            self.test_accuracies = checkpoint['test_accuracies']
        except FileNotFoundError:
            pass

    def save(self, epoch):
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
        }, self.path)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.load()
    trainer.test()
    for epoch in range(1, trainer.n_epochs + 1):
        trainer.train(epoch)
        trainer.test()

    sigmas = np.linspace(0, 2, 21)
    gaussian_noise_test_losses = []
    for var in sigmas:
        gaussian_noise_test_losses.append(trainer.test(gaussian_noise=True, noise_mean=0, noise_var=var))

    plt.plot(trainer.train_losses, color='blue')
    plt.plot(np.linspace(0, len(trainer.train_losses), trainer.n_epochs), trainer.test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('Cross Entropy loss')
    plt.show()

    plt.plot(trainer.train_accuracies, color='green')
    plt.plot(np.linspace(0, len(trainer.train_accuracies), trainer.n_epochs), trainer.test_accuracies, color='black')
    plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(sigmas, gaussian_noise_test_losses, color='blue')
    plt.legend(['Test loss with gaussian noise'], loc='upper right')
    plt.xlabel('Variance')
    plt.ylabel('Test loss')
    plt.show()
