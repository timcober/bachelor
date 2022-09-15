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


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split data into 80% train and 20% test subsets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2, shuffle=False, random_state=seed)

#appending outlier to the data for detection testing
#0
X_train = np.append(X_train, [np.full(shape=64, fill_value=0, dtype=np.float64)], axis=0)
y_train = np.append(y_train, np.full(shape=1, fill_value=0, dtype=np.int64), axis=0)
#9
X_train = np.append(X_train, [np.full(shape=64, fill_value=16, dtype=np.float64)], axis=0)
y_train = np.append(y_train, np.full(shape=1, fill_value=9, dtype=np.int64), axis=0)
#5
X_train = np.append(X_train, [np.full(shape=64, fill_value=8, dtype=np.float64)], axis=0)
y_train = np.append(y_train, np.full(shape=1, fill_value=5, dtype=np.int64), axis=0)

'''
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
'''

X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.int64))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.int64))


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
BATCH_SIZE = 48
assert len(X_train)%BATCH_SIZE==0, "The training set size must be evenly divisible by the batch size"
EPOCHS = 1000
CLIP = 1.0
DELTA = 1e-3
EPSILON = 2.0
SECURE_MODE = False

model = LogisticRegression(input_dim, output_dim).to(device)
criterion = nn.CrossEntropyLoss() 

trainloader = DataLoader(DataSet(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)


from functorch import make_functional, vmap, grad
from opacus.optimizers import DPOptimizer
from opacus.accountants.utils import get_noise_multiplier

fmodel, params = make_functional(model)

def compute_loss(params, sample, target, fmodel, loss_fn):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)
    predictions = fmodel(params, batch)
    loss = loss_fn(predictions, targets)
    return loss

compute_per_sample_grads = vmap(
    grad(compute_loss), in_dims=(None, 0, 0, None, None)
)
#DPOptimizer
optimizer = DPOptimizer(torch.optim.Adam(params, lr=LEARNING_RATE), noise_multiplier=get_noise_multiplier(target_epsilon=EPSILON, target_delta=1/len(X_train), sample_rate=1/len(trainloader), epochs=EPOCHS), max_grad_norm=CLIP, expected_batch_size=BATCH_SIZE)


from statistics import mean
from collections import defaultdict
import time

torch.set_grad_enabled(False)
epoch_accs = defaultdict(list)
epoch_losses = defaultdict(list)

gnorms = dict()
start_time = time.time()

for epoch in range(EPOCHS):
  batch_accs = []
  batch_losses = []
  batch_grads = []
  
  for batch, target in trainloader:
    batch = batch.to(device)
    target = target.to(device)

    y_pred = fmodel(params, batch)
    loss = criterion(y_pred, target)
    batch_losses.append(loss.item())
    preds = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
    labels = target.detach().cpu().numpy()
    batch_acc = (preds == labels).mean()
    batch_accs.append(batch_acc.item())

    per_sample_grads = compute_per_sample_grads(params, batch, target, fmodel, criterion)

    for param, grad_sample in zip(params, per_sample_grads):
        param.grad_sample = grad_sample
        param.grad = grad_sample.mean(0)

    clipped_norms = torch.clamp_max(torch.cat([elem.flatten(1) for elem in per_sample_grads], 1).norm(2, 1), CLIP)
    batch_grads.extend(clipped_norms.cpu().numpy().tolist())

    optimizer.step()
    optimizer.zero_grad(True)

  gnorms[epoch+1] = batch_grads
  
  epoch_accs[epoch+1].append(mean(batch_accs))
  epoch_losses[epoch+1].append(mean(batch_losses))
  if epoch!=0 and epoch%5==0:
    print(f"Epoch {epoch+1}: Loss {epoch_losses[epoch+1][0]:.2f}, Train accuracy {epoch_accs[epoch+1][0]:.2f}")

test_predictions = torch.argmax(torch.softmax(fmodel(params, X_test), 1), axis=1)
test_accuracy = float(sum(test_predictions == y_test)) / y_test.shape[0]

#print(f"Test accuracy: {mean(test_batch_accs):.2f}")
print(f"Test accuracy: {test_accuracy:.2f}")
print("--- %s seconds ---" % ((time.time() - start_time)))


performances = dict.fromkeys(range(len(gnorms[1])),0)
for i in range(len(gnorms)):
  for j in range(len(gnorms[i+1])):
    performances[j] += gnorms[i+1][j]
print(performances)
for k in range(len(performances)):
  performances[k] /= len(gnorms)
print(performances)


def training(model, train_dataloader, lr=LEARNING_RATE, loss_criteria=criterion, device=device):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(EPOCHS):
            # TRAINING
            model.train()
            # run batches
            for i, (X, Y) in enumerate(train_dataloader):
                
                # move images, labels to device (GPU)
                X = X.to(device)
                Y = Y.to(device)

                # clear previous gradient
                optimizer.zero_grad()

                # feed forward the model
                output = model(X)
                train_batch_loss = loss_criteria(output, Y)
                
                # back propagation
                train_batch_loss.backward()
                
                # update parameters
                optimizer.step()


def test(model, X_test, y_test, device=device):
    model.eval()
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    test_predictions = torch.argmax(torch.softmax(model(X_test), 1), axis=1)
    test_accuracy = float(sum(test_predictions == y_test)) / y_test.shape[0]
    return test_accuracy


import matplotlib.pyplot as plt
from torch.utils import data

#Leaving out high valued data
performanceOrder = {k: v for k, v in sorted(performances.items(), key=lambda item: item[1])}
print(performanceOrder)

torch.set_grad_enabled(True)

# removing 5 percent of training data
percentile = int(len(performanceOrder)*0.05)
indices = list(performanceOrder.keys())
print(indices)
modelperformance = {}

print("Training initial model")
model = LogisticRegression(input_dim,output_dim).to(device)
train_loader = DataLoader(DataSet(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
training(model, train_loader)
modelperformance[0] = test(model, X_test, y_test)

for i in range(1,10):
  print('\nLeaving the next 5% of data with lowest value out of training')

  print('Resetting model')
  model_copy = LogisticRegression(input_dim,output_dim).to(device)

  for _ in range(percentile):
    del indices[0]

  print("Training with ", len(indices), "samples")
  sampler = data.SubsetRandomSampler(indices)
  dataloader = DataLoader(
        DataSet(X_train, y_train),
        batch_size=BATCH_SIZE,
        sampler=sampler
  )
  training(model_copy, dataloader)
 
  # Test the model without one sample
  testing = test(model_copy, X_test, y_test)
  modelperformance[i*5] = testing

print(modelperformance)

plt.plot(list(modelperformance.keys()),list(modelperformance.values()))


#Leaving out high valued data
performanceOrder = {k: v for k, v in sorted(performances.items(), key=lambda item: item[1], reverse=True)}
print(performanceOrder)

# removing 5 percent of training data
percentile = int(len(performanceOrder)*0.05)
indices = list(performanceOrder.keys())
print(indices)
modelperformance = {}

print("Training initial model")
model = LogisticRegression(input_dim,output_dim).to(device)
train_loader = DataLoader(DataSet(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False)
training(model, train_loader)
modelperformance[0] = test(model, X_test, y_test)

for i in range(1,10):
  print('\nLeaving the next 5% of data with lowest value out of training')

  print('Resetting model')
  model_copy = LogisticRegression(input_dim,output_dim).to(device)

  for _ in range(percentile):
    del indices[0]

  print("Training with ", len(indices), "samples")
  sampler = data.SubsetRandomSampler(indices)
  dataloader = DataLoader(
        DataSet(X_train, y_train),
        batch_size=BATCH_SIZE,
        sampler=sampler
  )
  training(model_copy, dataloader)
 
  # Test the model without one sample
  testing = test(model_copy, X_test, y_test)
  modelperformance[i*5] = testing

print(modelperformance)

plt.plot(list(modelperformance.keys()),list(modelperformance.values()))