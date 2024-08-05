import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt


# Define a simple neural network using only nn.Linear and nn.ReLU
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x


def show_predictions(images, labels, predictions):
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for idx in range(10):
        ax = axes[idx]
        ax.imshow(images[idx].view(28, 28), cmap='gray')
        ax.set_title(f'True: {labels[idx]}\nPred: {predictions[idx]}')
        ax.axis('off')
    plt.show()


if __name__ == '__main__':


    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # Split the training set into training and validation sets
    train_idx, val_idx = train_test_split(list(range(len(trainset))), test_size=0.2, random_state=42)
    trainloader = DataLoader(Subset(trainset, train_idx), batch_size=64, shuffle=True)
    valloader = DataLoader(Subset(trainset, val_idx), batch_size=64, shuffle=False)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False)

    # Initialize the neural network, loss function and optimizer
    net = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Early stopping parameters
    patience = 3
    best_val_loss = float('inf')
    patience_counter = 0

    # Train the neural network
    epochs = 1
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # Flatten the inputs
            inputs = inputs.view(-1, 28 * 28)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        # Validate the neural network
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data

                # Flatten the inputs
                inputs = inputs.view(-1, 28 * 28)

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(valloader)
        print(f'Epoch {epoch + 1}, Validation loss: {val_loss:.3f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping')
                break

    # Test the neural network
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data

            # Flatten the inputs
            inputs = inputs.view(-1, 28 * 28)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)



            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

    # Save the model
    torch.save(net.state_dict(), 'mnist_classifier.pth')

    # Load the model and inference
    net.load_state_dict(torch.load('mnist_classifier.pth'))
    net.eval()



    # Display 10 images and their predictions
    images, labels = next(iter(testloader))
    images, labels = images[:10], labels[:10]
    images_flat = images.view(-1, 28 * 28)
    with torch.no_grad():
        outputs = net(images_flat)
        _, predictions = torch.max(outputs, 1)
    show_predictions(images, labels, predictions)



    with torch.no_grad():
        sample_input = torch.zeros(1, 28 * 28)  # Use zeros as input for inference
        output = net(sample_input)
        print(output)

    dummy_input = torch.randn(784).to(torch.device("cpu"))
    PATH = "./models/classifiers_various_depth/"
    torch.onnx.export(net,
                      dummy_input,
                      PATH + "mnist_unscaled20.onnx",
                      export_params=True,
                      verbose=False
                      )
