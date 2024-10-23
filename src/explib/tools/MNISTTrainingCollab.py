import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


class Net1(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(Net1, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.relu2 = nn.ReLU()
    self.fcfin = nn.Linear(hidden_size, num_classes)

  def forward(self,x):
    out = self.fc1(x)
    out = self.relu1(out)
    out = self.fc2(out)
    out = self.relu2(out)
    out = self.fcfin(out)
    return out

class Net2(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(Net2,self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.relu2 = nn.ReLU()

    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.relu3 = nn.ReLU()
    self.fc4 = nn.Linear(hidden_size, hidden_size)
    self.relu4 = nn.ReLU()
    self.fc5 = nn.Linear(hidden_size, hidden_size)
    self.relu5 = nn.ReLU()

    self.fcfin = nn.Linear(hidden_size, num_classes)

  def forward(self,x):
    out = self.fc1(x)
    out = self.relu1(out)
    out = self.fc2(out)
    out = self.relu2(out)
    out = self.fc3(out)
    out = self.relu3(out)
    out = self.fc4(out)
    out = self.relu4(out)
    out = self.fc5(out)
    out = self.relu5(out)
    out = self.fcfin(out)
    return out

class Net3(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(Net3,self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.relu2 = nn.ReLU()

    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.relu3 = nn.ReLU()
    self.fc4 = nn.Linear(hidden_size, hidden_size)
    self.relu4 = nn.ReLU()
    self.fc5 = nn.Linear(hidden_size, hidden_size)
    self.relu5 = nn.ReLU()

    self.fc6 = nn.Linear(hidden_size, hidden_size)
    self.relu6 = nn.ReLU()
    self.fc7 = nn.Linear(hidden_size, hidden_size)
    self.relu7 = nn.ReLU()
    self.fc8 = nn.Linear(hidden_size, hidden_size)
    self.relu8 = nn.ReLU()

    self.fcfin = nn.Linear(hidden_size, num_classes)

  def forward(self,x):
    out = self.fc1(x)
    out = self.relu1(out)
    out = self.fc2(out)
    out = self.relu2(out)

    out = self.fc3(out)
    out = self.relu3(out)
    out = self.fc4(out)
    out = self.relu4(out)
    out = self.fc5(out)
    out = self.relu5(out)

    out = self.fc6(out)
    out = self.relu6(out)
    out = self.fc7(out)
    out = self.relu7(out)
    out = self.fc8(out)
    out = self.relu8(out)

    out = self.fcfin(out)
    return out


class Net4(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(Net4,self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.relu2 = nn.ReLU()

    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.relu3 = nn.ReLU()
    self.fc4 = nn.Linear(hidden_size, hidden_size)
    self.relu4 = nn.ReLU()
    self.fc5 = nn.Linear(hidden_size, hidden_size)
    self.relu5 = nn.ReLU()

    self.fc6 = nn.Linear(hidden_size, hidden_size)
    self.relu6 = nn.ReLU()
    self.fc7 = nn.Linear(hidden_size, hidden_size)
    self.relu7 = nn.ReLU()
    self.fc8 = nn.Linear(hidden_size, hidden_size)
    self.relu8 = nn.ReLU()

    self.fc9 = nn.Linear(hidden_size, hidden_size)
    self.relu9 = nn.ReLU()
    self.fc10 = nn.Linear(hidden_size, hidden_size)
    self.relu10 = nn.ReLU()
    self.fc11 = nn.Linear(hidden_size, hidden_size)
    self.relu11 = nn.ReLU()

    self.fcfin = nn.Linear(hidden_size, num_classes)

  def forward(self,x):
    out = self.fc1(x)
    out = self.relu1(out)
    out = self.fc2(out)
    out = self.relu2(out)

    out = self.fc3(out)
    out = self.relu3(out)
    out = self.fc4(out)
    out = self.relu4(out)
    out = self.fc5(out)
    out = self.relu5(out)

    out = self.fc6(out)
    out = self.relu6(out)
    out = self.fc7(out)
    out = self.relu7(out)
    out = self.fc8(out)
    out = self.relu8(out)

    out = self.fc9(out)
    out = self.relu9(out)
    out = self.fc10(out)
    out = self.relu10(out)
    out = self.fc11(out)
    out = self.relu11(out)

    out = self.fcfin(out)
    return out



class Net5(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(Net5,self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.relu2 = nn.ReLU()

    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.relu3 = nn.ReLU()
    self.fc4 = nn.Linear(hidden_size, hidden_size)
    self.relu4 = nn.ReLU()
    self.fc5 = nn.Linear(hidden_size, hidden_size)
    self.relu5 = nn.ReLU()

    self.fc6 = nn.Linear(hidden_size, hidden_size)
    self.relu6 = nn.ReLU()
    self.fc7 = nn.Linear(hidden_size, hidden_size)
    self.relu7 = nn.ReLU()
    self.fc8 = nn.Linear(hidden_size, hidden_size)
    self.relu8 = nn.ReLU()

    self.fc9 = nn.Linear(hidden_size, hidden_size)
    self.relu9 = nn.ReLU()
    self.fc10 = nn.Linear(hidden_size, hidden_size)
    self.relu10 = nn.ReLU()
    self.fc11 = nn.Linear(hidden_size, hidden_size)
    self.relu11 = nn.ReLU()

    self.fc12 = nn.Linear(hidden_size, hidden_size)
    self.relu12 = nn.ReLU()
    self.fc13 = nn.Linear(hidden_size, hidden_size)
    self.relu13 = nn.ReLU()
    self.fc14 = nn.Linear(hidden_size, hidden_size)
    self.relu14 = nn.ReLU()

    self.fcfin = nn.Linear(hidden_size, num_classes)

  def forward(self,x):
    out = self.fc1(x)
    out = self.relu1(out)
    out = self.fc2(out)
    out = self.relu2(out)

    out = self.fc3(out)
    out = self.relu3(out)
    out = self.fc4(out)
    out = self.relu4(out)
    out = self.fc5(out)
    out = self.relu5(out)

    out = self.fc6(out)
    out = self.relu6(out)
    out = self.fc7(out)
    out = self.relu7(out)
    out = self.fc8(out)
    out = self.relu8(out)

    out = self.fc9(out)
    out = self.relu9(out)
    out = self.fc10(out)
    out = self.relu10(out)
    out = self.fc11(out)
    out = self.relu11(out)

    out = self.fc12(out)
    out = self.relu12(out)
    out = self.fc13(out)
    out = self.relu13(out)
    out = self.fc14(out)
    out = self.relu14(out)

    out = self.fcfin(out)
    return out





class Net6(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(Net6,self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.relu2 = nn.ReLU()

    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.relu3 = nn.ReLU()
    self.fc4 = nn.Linear(hidden_size, hidden_size)
    self.relu4 = nn.ReLU()
    self.fc5 = nn.Linear(hidden_size, hidden_size)
    self.relu5 = nn.ReLU()

    self.fc6 = nn.Linear(hidden_size, hidden_size)
    self.relu6 = nn.ReLU()
    self.fc7 = nn.Linear(hidden_size, hidden_size)
    self.relu7 = nn.ReLU()
    self.fc8 = nn.Linear(hidden_size, hidden_size)
    self.relu8 = nn.ReLU()

    self.fc9 = nn.Linear(hidden_size, hidden_size)
    self.relu9 = nn.ReLU()
    self.fc10 = nn.Linear(hidden_size, hidden_size)
    self.relu10 = nn.ReLU()
    self.fc11 = nn.Linear(hidden_size, hidden_size)
    self.relu11 = nn.ReLU()

    self.fc12 = nn.Linear(hidden_size, hidden_size)
    self.relu12 = nn.ReLU()
    self.fc13 = nn.Linear(hidden_size, hidden_size)
    self.relu13 = nn.ReLU()
    self.fc14 = nn.Linear(hidden_size, hidden_size)
    self.relu14 = nn.ReLU()

    self.fc15 = nn.Linear(hidden_size, hidden_size)
    self.relu15 = nn.ReLU()
    self.fc16 = nn.Linear(hidden_size, hidden_size)
    self.relu16 = nn.ReLU()
    self.fc17 = nn.Linear(hidden_size, hidden_size)
    self.relu17 = nn.ReLU()

    self.fcfin = nn.Linear(hidden_size, num_classes)

  def forward(self,x):
    out = self.fc1(x)
    out = self.relu1(out)
    out = self.fc2(out)
    out = self.relu2(out)

    out = self.fc3(out)
    out = self.relu3(out)
    out = self.fc4(out)
    out = self.relu4(out)
    out = self.fc5(out)
    out = self.relu5(out)

    out = self.fc6(out)
    out = self.relu6(out)
    out = self.fc7(out)
    out = self.relu7(out)
    out = self.fc8(out)
    out = self.relu8(out)

    out = self.fc9(out)
    out = self.relu9(out)
    out = self.fc10(out)
    out = self.relu10(out)
    out = self.fc11(out)
    out = self.relu11(out)

    out = self.fc12(out)
    out = self.relu12(out)
    out = self.fc13(out)
    out = self.relu13(out)
    out = self.fc14(out)
    out = self.relu14(out)

    out = self.fc15(out)
    out = self.relu15(out)
    out = self.fc16(out)
    out = self.relu16(out)
    out = self.fc17(out)
    out = self.relu17(out)

    out = self.fcfin(out)
    return out


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


nets = [Net6(input_size, hidden_size, num_classes),
        Net5(input_size, hidden_size, num_classes),
        Net4(input_size, hidden_size, num_classes),
        Net3(input_size, hidden_size, num_classes),
        Net2(input_size, hidden_size, num_classes),
        Net1(input_size, hidden_size, num_classes)]
for net_index, net in enumerate(nets):
  #net = Net(input_size, hidden_size, num_classes)
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
                    f'models/classifier_14_px/classifier_mnist_14_px_{net_index}.onnx',
                    export_params=True,
                    verbose=False
                    )



