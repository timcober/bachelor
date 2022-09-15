import warnings
import torch
import numpy as np
import random

warnings.filterwarnings("ignore")

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}.')


from sklearn import datasets
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
n_samples, n_features = data.shape

# Split data into 80% train and 20% test subsets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2, shuffle=False, random_state=seed)


#appending 10 outlier to the data for detection testing
#0
X_train = np.append(X_train, [np.full(shape=64, fill_value=0, dtype=np.float64)], axis=0)
y_train = np.append(y_train, np.full(shape=1, fill_value=0, dtype=np.int64), axis=0)
#1
X_train = np.append(X_train, [np.full(shape=64, fill_value=1, dtype=np.float64)], axis=0)
y_train = np.append(y_train, np.full(shape=1, fill_value=1, dtype=np.int64), axis=0)
#2
X_train = np.append(X_train, [np.full(shape=64, fill_value=2, dtype=np.float64)], axis=0)
y_train = np.append(y_train, np.full(shape=1, fill_value=2, dtype=np.int64), axis=0)
#3
X_train = np.append(X_train, [np.full(shape=64, fill_value=3, dtype=np.float64)], axis=0)
y_train = np.append(y_train, np.full(shape=1, fill_value=3, dtype=np.int64), axis=0)
#4
X_train = np.append(X_train, [np.full(shape=64, fill_value=4, dtype=np.float64)], axis=0)
y_train = np.append(y_train, np.full(shape=1, fill_value=4, dtype=np.int64), axis=0)
#5
X_train = np.append(X_train, [np.full(shape=64, fill_value=5, dtype=np.float64)], axis=0)
y_train = np.append(y_train, np.full(shape=1, fill_value=5, dtype=np.int64), axis=0)
#6
X_train = np.append(X_train, [np.full(shape=64, fill_value=6, dtype=np.float64)], axis=0)
y_train = np.append(y_train, np.full(shape=1, fill_value=6, dtype=np.int64), axis=0)
#7
X_train = np.append(X_train, [np.full(shape=64, fill_value=7, dtype=np.float64)], axis=0)
y_train = np.append(y_train, np.full(shape=1, fill_value=7, dtype=np.int64), axis=0)
#8
X_train = np.append(X_train, [np.full(shape=64, fill_value=8, dtype=np.float64)], axis=0)
y_train = np.append(y_train, np.full(shape=1, fill_value=8, dtype=np.int64), axis=0)
#9
X_train = np.append(X_train, [np.full(shape=64, fill_value=16, dtype=np.float64)], axis=0)
y_train = np.append(y_train, np.full(shape=1, fill_value=9, dtype=np.int64), axis=0)


X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.int64))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.int64))

def train(model, criterion, optimizer, dataloader, epoch, print_every, device=device):
    losses = []
    model.train()
    for X, Y in dataloader:
        x = X.to(device)
        y = Y.to(device)

        # zero grad before new step
        optimizer.zero_grad()

        # Forward pass and loss
        y_pred = model(x)
        loss = criterion(y_pred, y)

        # Backward pass and update
        loss.backward()    
        optimizer.step()

        losses.append(loss.item())

    if(print_every):
        print(
            f"\tTrain Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
        )

def test(model, X_test, y_test, device=device):
    model.eval()
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    test_predictions = torch.argmax(torch.softmax(model(X_test), 1), axis=1)
    test_accuracy = float(sum(test_predictions == y_test)) / y_test.shape[0]
    return test_accuracy 


import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# Training hyper-parameters
n_samples, n_features = X_train.shape
input_dim = n_features
output_dim = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 50
EPOCHS = 100

# model and optimizer
model = LogisticRegression(input_dim,output_dim).to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

import copy
from torch.utils.data import Dataset, DataLoader

class DataSet(Dataset):
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    if len(self.X) != len(self.Y):
      raise Exception("The length of X does not match the length of Y")

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    _x = self.X[index]
    _y = self.Y[index]

    return _x, _y

trainloader = DataLoader(DataSet(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)

# Save a clone of initial model to restore later
initial_model = copy.deepcopy(model)
initial_optimizer = copy.deepcopy(optimizer)
initial_trainloader = copy.deepcopy(trainloader)

#5) Only training the model
def training_only(model, criterion, optimizer, dataloader, epochs):
    print("Train stats: \n")
    for epoch in range(epochs):
        PRINT_EVERY = epoch % 99 == 0
        train(model, criterion, optimizer, dataloader, epoch, PRINT_EVERY)

import time
from torch.utils import data

# timing
tic = time.perf_counter()

training_only(model, criterion, optimizer, trainloader, EPOCHS)
intitial_performance = test(model, X_test, y_test)
print("Intitial model performance: ", intitial_performance)


# LOOCV
accuracy = []
for idx in range(len(X_train)):
 
    print('\nLeaving sample {} out of training'.format(idx))
 
    print('Resetting model')
    model_copy = LogisticRegression(input_dim,output_dim).to(device)
    
    optimizer_copy = torch.optim.Adam(model_copy.parameters(), lr=LEARNING_RATE)
 
    # Get all indices and remove train sample
    indices = list(range(len(X_train)))
    del indices[idx]
 
    # Create new sampler
    sampler = data.SubsetRandomSampler(indices)
    dataloader = DataLoader(
        DataSet(X_train, y_train),
        batch_size=BATCH_SIZE,
        sampler=sampler
    )
 
    training_only(model = model_copy, criterion = criterion, optimizer = optimizer_copy, dataloader = dataloader, epochs = EPOCHS)
 
    # Test the model without one sample
    testing = test(model_copy, X_test, y_test)
    print(testing)
    accuracy.append(testing)
    
performances={}
for idx in range(len(accuracy)):
    performances[idx] = intitial_performance-accuracy[idx]
    print(f"Performance score of sample {idx}: ", performances[idx])

print(f"Overall Total Time for calculation: {time.perf_counter()-tic:0.4f} seconds")


import matplotlib.pyplot as plt

#removing low valued data points
performanceOrder = {k: v for k, v in sorted(performances.items(), key=lambda item: item[1])}
print(performanceOrder)

# removing 1 percent of training data
percentile = int(len(performanceOrder)*0.01)
indices = list(performanceOrder.keys())
print(indices)
modelperformance = {}
modelperformance[0] = intitial_performance

for i in range(1,50):
  print('\nLeaving the next 1% of data with lowest value out of training')

  print('Resetting model')
  model_copy = LogisticRegression(input_dim,output_dim).to(device)
    
  optimizer_copy = torch.optim.Adam(model_copy.parameters(), lr=LEARNING_RATE)

  for _ in range(percentile):
    del indices[0]

  print("Training with ", len(indices), "samples")
  sampler = data.SubsetRandomSampler(indices)
  dataloader = DataLoader(
        DataSet(X_train, y_train),
        batch_size=BATCH_SIZE,
        sampler=sampler
  )
  training_only(model = model_copy, criterion = criterion, optimizer = optimizer_copy, dataloader = dataloader, epochs = EPOCHS)
 
  # Test the model without one sample
  testing = test(model_copy, X_test, y_test)
  modelperformance[i] = testing

print(modelperformance)

plt.plot(list(modelperformance.keys()),list(modelperformance.values()))


#removing high valued data points
performanceOrder = {k: v for k, v in sorted(performances.items(), key=lambda item: item[1], reverse=True)}
print(performanceOrder)

# removing 1 percent of training data
percentile = int(len(performanceOrder)*0.01)
indices = list(performanceOrder.keys())
print(indices)
modelperformance = {}
modelperformance[0] = intitial_performance

for i in range(1,50):
  print('\nLeaving the next 1% of data with highest value out of training')

  print('Resetting model')
  model_copy = LogisticRegression(input_dim,output_dim).to(device)
    
  optimizer_copy = torch.optim.Adam(model_copy.parameters(), lr=LEARNING_RATE)

  for _ in range(percentile):
    del indices[0]

  print("Training with ", len(indices), "samples")
  sampler = data.SubsetRandomSampler(indices)
  dataloader = DataLoader(
        DataSet(X_train, y_train),
        batch_size=BATCH_SIZE,
        sampler=sampler
  )
  model_ft = training_only(model = model_copy, criterion = criterion, optimizer = optimizer_copy, dataloader = dataloader, epochs = EPOCHS)
 
  # Test the model without one sample
  testing = test(model_copy, X_test, y_test)
  modelperformance[i] = testing

print(modelperformance)

plt.plot(list(modelperformance.keys()),list(modelperformance.values()))