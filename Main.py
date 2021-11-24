from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

from Stock_data import *

import torch
import torch.cuda as cu
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from pytorch_forecasting.metrics import MAPE
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, seq_length):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.input_size = input_size
        self.seq_length = seq_length

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, input, future=0):
        outputs = []
        # h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        # c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        # h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        # c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        outputs, (h_n, c_n) = self.lstm1(input)
        outputs = self.linear(outputs)



        # for input_t in input.split(1, dim=1):
        #     h_t, c_t = self.lstm1(input_t, (h_t, c_t))
        #     h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        #     output = self.linear(h_t2)
        #     outputs += [output]
        # for i in range(future):  # if we should predict the future
        #     h_t, c_t = self.lstm1(output, (h_t, c_t))
        #     h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        #     output = self.linear(h_t2)
        #     outputs += [output]
        # outputs = torch.cat(outputs, dim=1)
        return outputs


def train(model, train_data, target_data, optimizer, epoch, loss_func):
    model.train(True)
    # for batch_idx, (data, target) in enumerate(train_loader):
    #     data, target = data.to(device), target.to(device)
    #     output = model(data)
    #     loss = loss_func.loss(output, target)
    #     loss.backward()
    #     optimizer.step()

    out = model(train_data)

    loss = loss_func(out[0], target_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch: {:d}  Loss: {:.6f}'.format(epoch, loss.item()))
    return loss

def test(model, test_data, target_data, epoch, loss_func):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        out = model(test_data)
        test_loss += loss_func(out[0], target_data)
        pred = out.argmax(dim=1, keepdim=True)
        correct += pred.eq(target_data.view_as(pred)).sum().item()

    test_loss /= len(test_data.dataset)

    if epoch % 10 == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_data.dataset),
            100. * correct / len(test_data.dataset)))

    return test_loss




x_stock_data1 = torch.tensor([get_index_data("03/05/2018", "29/12/2019", "S&P 500")[:-1]]).to(device)
x_stock_data2 = torch.tensor([get_index_data("13/03/2020", "29/05/2020", "S&P 500")[:-1]]).to(device)

y_stock_data1 = torch.tensor(get_index_data("03/05/2018", "29/12/2019", "S&P 500")[1:]).to(device)
y_stock_data2 = torch.tensor(get_index_data("13/03/2020", "29/05/2020", "S&P 500")[1:]).to(device)

# stock_data2 = get_index_data("03/05/2018", "29/12/2019", "Nasdaq").tolist()

# dataset_train = TensorDataset(torch.tensor(stock_data1[:-1]), torch.tensor(stock_data1[1:]))
# dataset_test = TensorDataset(torch.tensor(stock_data2[:-1]), torch.tensor(stock_data2[1:]))

# train_loader = DataLoader(dataset_train, batch_size=1)
# test_loader = DataLoader(dataset_test, batch_size=1)


model = Net(input_size=5, hidden_size=50, num_classes=5, seq_length=len(x_stock_data1)-1).to(device)
model.double()
learning_rate = 0.9
num_epochs = 200

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
loss = 1000000000
epoch = 0
while loss > 0.001:
    epoch += 1
    train(model, x_stock_data1, y_stock_data1, optimizer, epoch, nn.L1Loss()) # Mean absolute percentage error for regression
    pred = test(model, x_stock_data2, y_stock_data2, epoch, nn.L1Loss())

    y = pred.detach().numpy()

    plt.figure(figsize=(30, 10))
    plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)


    def draw(yi, color):
        plt.plot(np.arange(x_stock_data1.size(1)), yi, color, linewidth=2.0)

    draw(y, 'r')
    plt.savefig('predict%d.pdf' % epoch)
    plt.close()

