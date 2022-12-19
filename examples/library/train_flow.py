import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

class Flow(nn.Module):

    def __init__(self, data_dim, hidden_dim=412, layer_count=4, conditioner_layer_count=2):
        super().__init__()

        conditioner_hidden_layer = []
        for i in range(conditioner_layer_count):
          conditioner_hidden_layer.append(nn.Linear(hidden_dim, hidden_dim))
          conditioner_hidden_layer.append(nn.ReLU())
        self.model = nn.ModuleList([nn.Sequential(
            nn.Linear(data_dim // 2, hidden_dim), nn.ReLU(),
            *conditioner_hidden_layer,
            nn.Linear(hidden_dim, data_dim // 2), ) for i in range(layer_count)])
        self.s = nn.Parameter(torch.randn(data_dim))

    def forward(self, x):
      x = x.clone()
      for i in range(len(self.model)):
        x = self.__single_forward_step(x, i)
      z = torch.exp(self.s) * x
      log_jacobian = torch.sum(self.s)
      return z, log_jacobian
    
    def forward_step_by_step(self, x):
      out = []
      x = x.clone()
      for i in range(len(self.model)):
        x = self.__single_forward_step(x, i)
        out.append(x)
      z = torch.exp(self.s) * x
      log_jacobian = torch.sum(self.s)
      return out, log_jacobian

    def __single_forward_step(self, x, i):
      x_i1 = x[:, ::2] if (i % 2) == 0 else x[:, 1::2]
      x_i2 = x[:, 1::2] if (i % 2) == 0 else x[:, ::2]
      h_i1 = x_i1
      h_i2 = x_i2 + self.model[i](x_i1)
      x = torch.empty(x.shape, device=x.device)
      x[:, ::2] = h_i1
      x[:, 1::2] = h_i2
      return x

    def invert(self, z):
      x = z.clone() / torch.exp(self.s)
      for i in range(len(self.model) - 1, -1, -1):
        x = self.__single_invert_step(x, i)
      return x
    
    def invert_step_by_step(self, z):
      out = []
      x = z.clone() / torch.exp(self.s)
      out.append(x)
      for i in range(len(self.model) - 1, -1, -1):
        x = self.__single_invert_step(x, i)
        out.append(x)
      return out
    
    def __single_invert_step(self, x, i):
      h_i1 = x[:, ::2]
      h_i2 = x[:, 1::2]
      x_i1 = h_i1
      x_i2 = h_i2 - self.model[i](x_i1)
      x = torch.empty(x.shape, device=x.device)
      x[:, ::2] = x_i1 if (i % 2) == 0 else x_i2
      x[:, 1::2] = x_i2 if (i % 2) == 0 else x_i1
      return x

def train_flow(normalizing_flow, optimizer, dataloader, distribution, nb_epochs=1500, device='cpu'):
    training_loss = []
    for _ in tqdm(range(nb_epochs)):

        for batch in dataloader:
            z, log_jacobian = normalizing_flow(batch.to(device))
            log_likelihood = distribution.log_prob(z).sum(dim=1) + log_jacobian
            loss = -log_likelihood.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())

    return training_loss

LAYER_COUNT = 2
CONDITIONER = 1

def training(training_data, device, data_dim, y_distribution):
  normalizing_flow = Flow(data_dim = data_dim, hidden_dim=512, layer_count=LAYER_COUNT, conditioner_layer_count=CONDITIONER).to(device=device)

  x = torch.randn(10, data_dim, device=device)
  assert torch.allclose(normalizing_flow.invert(normalizing_flow(x)[0]), x, rtol=1e-04, atol=1e-06)

  print(normalizing_flow)

  optimizer = torch.optim.Adam(normalizing_flow.parameters(), lr=0.0002, weight_decay=0.9)
  dataloader = DataLoader(np.float32(training_data), batch_size=32, shuffle=True)
  training_loss = train_flow(normalizing_flow, optimizer, dataloader, y_distribution, nb_epochs=60,
                            device=device)
  return normalizing_flow