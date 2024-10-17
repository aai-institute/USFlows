import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

input_size = 196 # img_size = (28,28) ---> 28*28=784 in total
hidden_size = 50 # number of nodes at hidden layer
num_classes = 10 # number of output classes discrete range [0,9]
num_epochs = 7 # number of times which the entire dataset is passed throughout the model
batch_size = 64 # the size of input data took for one iteration
lr = 1e-3 # size of step

train_data = dsets.MNIST(root = './data', train = True,
                        transform = transforms.ToTensor(), download = True)

test_data = dsets.MNIST(root = './data', train = False,
                       transform = transforms.ToTensor())


train_data.data = train_data.data[:, ::2, ::2]
test_data.data = test_data.data[:, ::2, ::2]


train_gen = torch.utils.data.DataLoader(dataset = train_data,
                                             batch_size = batch_size,
                                             shuffle = True)

test_gen = torch.utils.data.DataLoader(dataset = test_data,
                                      batch_size = batch_size,
                                      shuffle = False)


class Net(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.relu2 = nn.ReLU()
    self.fc3 = nn.Linear(hidden_size, num_classes)

  def forward(self,x):
    out = self.fc1(x)
    out = self.relu1(out)
    out = self.fc2(out)
    out = self.relu2(out)
    out = self.fc3(out)
    return out

net = Net(input_size, hidden_size, num_classes)
device = torch.device("cpu")
net.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

for epoch in range(num_epochs):
  for i ,(images,labels) in enumerate(train_gen):
    images = Variable(images.view(-1,14*14)).to(device)
    labels = Variable(labels).to(device)

    optimizer.zero_grad()
    outputs = net(images)
    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()

    if (i+1) % 100 == 0:
      print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                 %(epoch+1, num_epochs, i+1, len(train_data)//batch_size, loss.item()))


correct = 0
total = 0
for images,labels in test_gen:
  images = Variable(images.view(-1,14*14)).to(device)
  labels = labels.to(device)

  output = net(images)
  _, predicted = torch.max(output,1)
  correct += (predicted == labels).sum()
  total += labels.size(0)
dummy_input = torch.randn(196).to(device)
print('Accuracy of the model: %.3f %%' %((100*correct)/(total+1)))
torch.onnx.export(net,
                  dummy_input,
                  "models/MnistClassifier_14_14.onnx",
                  export_params=True,
                  verbose=False
                  )



