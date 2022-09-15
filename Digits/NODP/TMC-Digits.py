import torch
import random
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# Seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Set the device
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Running on {DEVICE}.')


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


# Split data into 80% train and 20% test subsets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2, shuffle=False, random_state=seed)


#appending 2 outlier to the data for detection testing
X_train = np.append(X_train, [np.full(shape=64, fill_value=0, dtype=np.float64)], axis=0)
y_train = np.append(y_train, np.full(shape=1, fill_value=0, dtype=np.int64), axis=0)
X_train = np.append(X_train, [np.full(shape=64, fill_value=16, dtype=np.float64)], axis=0)
y_train = np.append(y_train, np.full(shape=1, fill_value=9, dtype=np.int64), axis=0)
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
from torch.utils.data import Dataset

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


import torch
import numpy as np
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

BATCH_SIZE = 50
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
DEFAULT_LOSS = CrossEntropyLoss() 


class FitModule(Module):
    def fit(self,
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            loss_criteria=DEFAULT_LOSS,
            device=DEVICE,
            tmc=False):
        """fits the model with the train data, evaluates model during training on validation data
        # Arguments
            X_train: training data array of images
            y_train: training data array of labels
            lr: learning rate
            batch_size: number of samples per gradient update
            loss_criteria: training loss
            device: device used, GPU or CPU
            tmc: if fit is called within TMC 
        """
        X_train = torch.from_numpy(X_train.astype(np.float32))
        y_train = torch.from_numpy(y_train.astype(np.int64))
        # create dataloader
        train_dataloader = DataLoader(DataSet(X_train, y_train), batch_size, shuffle=False)

      
        optimizer = Adam(self.parameters(), lr=lr)

       
        accs = []
        losses = []
        for epoch in range(MAX_EPOCHS):
            # TRAINING
            self.train()
            train_epoch_loss = 0
            # run batches
            for i, (X, Y) in enumerate(train_dataloader):
                
                # move images, labels to device (GPU)
                X = X.to(device)
                Y = Y.to(device)

                # clear previous gradient
                optimizer.zero_grad()

                # feed forward the model
                output = self(X)
                train_batch_loss = loss_criteria(output, Y)
                
                # back propagation
                train_batch_loss.backward()
                
                # update parameters
                optimizer.step()
                

                # update training loss after each batch
                train_epoch_loss += train_batch_loss.item()

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = Y.detach().cpu().numpy()

                # measure accuracy and record loss
                acc = (preds == labels).mean()
                losses.append(train_batch_loss.item())
                accs.append(acc)
           
            if not tmc: 
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Accuracy: {np.mean(accs):.6f} "
                    # f"(ε = {epsilon:.2f}, δ = {DELTA})"
                )
            else:
                if epoch == MAX_EPOCHS-1:
                    print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Accuracy: {np.mean(accs):.6f} "
                    # f"(ε = {epsilon:.2f}, δ = {DELTA})"
                    )
                else:
                    continue

    def fitDL(self,
            dataloader,
            lr=LEARNING_RATE,
            loss_criteria=DEFAULT_LOSS,
            device=DEVICE
            ):
        """fits the model with the train data, evaluates model during training on validation data
        # Arguments
            dataloader: training data
            lr: learning rate
            batch_size: number of samples per gradient update
            loss_criteria: training loss
            device: device used, GPU or CPU
            tmc: if fit is called within TMC 
        """

        # create dataloader
        train_dataloader = dataloader

        optimizer = Adam(self.parameters(), lr=lr)

        accs = []
        losses = []
        for epoch in range(MAX_EPOCHS):
            # TRAINING
            self.train()
            train_epoch_loss = 0
            # run batches
            for i, (X, Y) in enumerate(train_dataloader):
                
                # move images, labels to device (GPU)
                X = X.to(device)
                Y = Y.to(device)

                # clear previous gradient
                optimizer.zero_grad()

                # feed forward the model
                output = self(X)
                train_batch_loss = loss_criteria(output, Y)
                
                # back propagation
                train_batch_loss.backward()
                
                # update parameters
                optimizer.step()
                
                # update training loss after each batch
                train_epoch_loss += train_batch_loss.item()

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = Y.detach().cpu().numpy()

                # measure accuracy and record loss
                acc = (preds == labels).mean()
                losses.append(train_batch_loss.item())
                accs.append(acc)
           
            # calculate training loss
            if epoch == MAX_EPOCHS-1:
                print(
                f"\tTrain Epoch: {epoch} \t"
                f"Loss: {np.mean(losses):.6f} "
                f"Accuracy: {np.mean(accs):.6f} "
                )
            else:
                continue

    def evaluate(self,
                 X_test,
                 y_test,
                 device=DEVICE):
        """evaluates performance of ML predictor on test data
        # Arguments
            X_test: test data array of images
            y_test: test data array of labels
            batch_size: number of samples per gradient update
            device: device used, GPU or CPU
        # Returns
            performane score"""
        self.eval()
        X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)
        y_test = torch.from_numpy(y_test.astype(np.int64)).to(device)
        
        # TESTING
        self.eval()
        test_predictions = torch.argmax(torch.softmax(self(X_test), 1), axis=1)
        test_accuracy = float(sum(test_predictions == y_test)) / y_test.shape[0]

        return test_accuracy


import torch.nn as nn

class LogisticRegression(FitModule):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, y):
        return self.linear(y)
        
    def score(self, X, y):
        return self.evaluate(X, y)
        
def get_model(input_dim, output_dim):
  return LogisticRegression(input_dim, output_dim).to(DEVICE)


import matplotlib
matplotlib.use('Agg')
import scipy

class DShap(object):

    def __init__(self, X_train, y_train, X_test, y_test, metric, sources=None, seed=None, input_dim=1, output_dim=1):
        """
        Args:
            X_train: Train covariates
            y_train: Train labels
            X_test: Test covariates
            y_test: Test labels
            model_family: The model family used for learning algorithm
            metric: Evaluation metric
            X_val: Validation covariates
            y_val: Validation labels
            X_train_deep: Train deep features
            X_test_deep: Test deep features
            sources: An array or dictionary assigning each point to its group.
                If None, evey points gets its individual value.
            directory: Directory to save results and figures.
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting
                same permutations.
            **kwargs: Arguments of the model
        """

        if seed is not None:
            np.random.seed(seed)
        self.metric = metric
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._initialize_instance(X_train, y_train, X_test, y_test, sources)
        self.model = get_model(self.input_dim, self.output_dim)
        self.random_score = self.init_score(self.metric)
        self.initial_state_dict = self.model.state_dict()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def _initialize_instance(self, X_train, y_train,X_test, y_test, sources=None):
        """loads or creates data"""

        sources = {i: np.array([i]) for i in range(X_train.shape[0])}
        self.sources = sources
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.mem_tmc = np.zeros((0, self.X_train.shape[0]))
        idxs_shape = (0, self.X_train.shape[0] if self.sources is None else len(self.sources.keys()))
        self.idxs_tmc = np.zeros(idxs_shape).astype(int)
        self.vals_tmc = np.zeros((self.X_train.shape[0],))

    def init_score(self, metric):
        """ gives the value of an initial untrained model"""
        if metric == 'auc':
            return 0.5
        if metric == 'accuracy':
            return np.max(np.bincount(self.y_test).astype(float) / len(self.y_test))
        else:
            print('Invalid metric!')


    def run(self, iterations, tolerance=0.1, tmc_run=False):
        """calculates data sources(points) values
        Args:
            iterations: Number of iterations to run.
            err: stopping criteria for each of TMC-Shapley or G-Shapley algorithm.
            tolerance: Truncation tolerance. If None, the instance computes its own.
        """

        self.restart_model()
        self.model.fit(self.X_train, self.y_train)
        
        print('-----Starting TMC-Shapley calculations:')
        self._tol_mean_score()
        marginals, idxs = [], []
        
        for iteration in range(iterations):
            if 100 * (iteration + 1) / iterations % 1 == 0:
                print('{} out of {} TMC_Shapley iterations.'.format(iteration + 1, iterations))
            marginals, idxs = self.one_iteration(tolerance=tolerance)
            self.mem_tmc = np.concatenate([self.mem_tmc, np.reshape(marginals, (1, -1))])
            self.idxs_tmc = np.concatenate([self.idxs_tmc, np.reshape(idxs, (1, -1))])
        self.vals_tmc = np.mean(self.mem_tmc, 0)
        print('-----TMC-Shapley values calculated!')

    def _tol_mean_score(self):
        """computes the average performance and its error using bagging"""
        scores = []
        self.restart_model()
        for _ in range(1):
            self.model.fit(self.X_train, self.y_train)
            for i in range(100):
                bag_idxs = np.random.choice(len(self.y_test), len(self.y_test))
                scores.append(self.model.score(self.X_test[bag_idxs], self.y_test[bag_idxs]))          
        self.mean_score = np.mean(scores)
        print("Average performance: ", self.mean_score)

    def one_iteration(self, tolerance):
        """runs one iteration of TMC-Shapley algorithm"""
        sources= self.sources
        idxs, marginal_contribs = np.random.permutation(len(sources.keys())), np.zeros(self.X_train.shape[0])
        new_score = self.random_score
        X_batch, y_batch = (np.zeros((0,) + tuple(self.X_train.shape[1:])), np.zeros((0,) + tuple(self.y_train.shape[1:])))
        
        truncation_counter = 0
        
        for n, idx in enumerate(idxs):
            old_score = new_score
            if isinstance(self.X_train, scipy.sparse.csr_matrix):
                X_batch = scipy.sparse.vstack([X_batch, self.X_train[sources[idx]]])
            else:
                X_batch = np.concatenate((X_batch, self.X_train[sources[idx]]))
            y_batch = np.concatenate([y_batch, self.y_train[sources[idx]]])               
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bool = True  # self.is_regression or len(set(y_batch)) == len(set(self.y_test))
                if bool:
                    self.restart_model()
                    self.model.fit(X_batch, y_batch, tmc=True)
                    new_score = self.model.score(self.X_test, self.y_test)
            marginal_contribs[sources[idx]] = (new_score - old_score) / len(sources[idx])
            
            if np.abs(new_score - self.mean_score) <= tolerance * self.mean_score:
                truncation_counter += 1
                if truncation_counter > 5:
                    break
            else:
                truncation_counter = 0
        return marginal_contribs, idxs

    def restart_model(self):
        self.model.load_state_dict(self.initial_state_dict)


import time

n_samples, n_features = X_train.shape

# parameters
metric = 'accuracy'
input_dim = n_features
iterations = 1000
tolerance = 0.1

print('--- calculate tmc run time')
start_time = time.time()
dshap = DShap(X_train, y_train, X_test, y_test, metric, seed=seed, input_dim=input_dim, output_dim=10)
dshap.run(iterations, tolerance, tmc_run=True)


performances={}
values = dshap.vals_tmc
for idx in range(len(values)):
    performances[idx] = values[idx]
    print(f"Performance score of sample {idx}: ", performances[idx])

print("# Shapley Values: ", len(values), "\n Values: ", values)

print("--- %s seconds ---" % ((time.time() - start_time)))
from torch.utils import data
import matplotlib.pyplot as plt

#Eliminate Low value scores 
performanceOrder = {k: v for k, v in sorted(performances.items(), key=lambda item: item[1])}
print(performanceOrder)
# removing 5 percent of training data
percentile = int(len(performanceOrder)*0.01)
indices = list(performanceOrder.keys())
print(indices)

#initial performance
modelperformance = {}
model = get_model(input_dim, 10)
model.fitDL(DataLoader(DataSet(X_train.astype(np.float32), y_train.astype(np.int64)), batch_size=BATCH_SIZE, shuffle=False))
modelperformance[0] = model.evaluate(X_test, y_test)

for i in range(1,50):
  print('\nLeaving the next 1% of data with lowest value out of training')

  print('Resetting model')
  model = get_model(input_dim, 10)

  for _ in range(percentile):
    del indices[0]

  print("Training with ", len(indices), "samples")
  sampler = data.SubsetRandomSampler(indices)
  dataloader = DataLoader(
        DataSet(X_train.astype(np.float32), y_train.astype(np.int64)),
        batch_size=BATCH_SIZE,
        sampler=sampler
  )
  model.fitDL(dataloader)
 
  # Test the model without one sample
  testing = model.evaluate(X_test, y_test)
  modelperformance[i] = testing

print(modelperformance)
plt.plot(list(modelperformance.keys()),list(modelperformance.values()))


# Eliminating High Value Scores 
performanceOrder = {k: v for k, v in sorted(performances.items(), key=lambda item: item[1], reverse=True)}
print(performanceOrder)

# removing 5 percent of training data
percentile = int(len(performanceOrder)*0.01)
indices = list(performanceOrder.keys())
print(indices)

#initial performance
modelperformance = {}
model = get_model(input_dim, 10)
model.fitDL(DataLoader(DataSet(X_train.astype(np.float32), y_train.astype(np.int64)), batch_size=BATCH_SIZE, shuffle=False))
modelperformance[0] = model.evaluate(X_test, y_test)

for i in range(1,50):
  print('\nLeaving the next 1% of data with lowest value out of training')

  print('Resetting model')
  model = get_model(input_dim, 10)

  for _ in range(percentile):
    del indices[0]

  print("Training with ", len(indices), "samples")
  sampler = data.SubsetRandomSampler(indices)
  dataloader = DataLoader(
        DataSet(X_train.astype(np.float32), y_train.astype(np.int64)),
        batch_size=BATCH_SIZE,
        sampler=sampler
  )
  model.fitDL(dataloader)
 
  # Test the model without one sample
  testing = model.evaluate(X_test, y_test)
  modelperformance[i] = testing

print(modelperformance)
plt.plot(list(modelperformance.keys()),list(modelperformance.values()))