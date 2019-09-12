import torch
import numpy as np
from model import resnet_tiny, resnet18, resnet34, resnet101
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
import matplotlib.pyplot as plt

from train import *

# create pytorch models
parallel_data = True
prefix = 'output/models/random5_batch16_0_'
if '_0_' in prefix:
    model_type = resnet_tiny
elif '_18_' in prefix:
    model_type = resnet18
elif '_34_' in prefix:
    model_type = resnet34
elif '_101_' in prefix:
    model_type = resnet101
    
    
# load pytorch models
print('making models')
gate_set = ['I', 'RX', 'RZ', 'CZ']
circuit_len = 4

checkpoint = torch.load(prefix + 'final_model.mdl')
model = model_type(n_channels=2*circuit_len, concat_features=False)
model.cuda()
if parallel_data:
    model = torch.nn.DataParallel(model)
model.load_state_dict(checkpoint['state_dict'], strict=False)

checkpoint2 = torch.load(prefix + 'early_stop.mdl')
model2 = model_type(n_channels=2*circuit_len, concat_features=False)
model2.cuda()
if parallel_data:
    model2 = torch.nn.DataParallel(model2)
model2.load_state_dict(checkpoint2['state_dict'], strict=False)


# load numpy data
print('loading data')
families_file = 'output/circuit-families-random-5.npz'
families = np.load(families_file)['arr_0']

np_data = []
np_target = []
files = np.load(prefix + 'dataset_files.npy')
    
for i in range(len(files)):
    arr = np.load(files[i])
    ind = int(files[i].split('.')[0].split('-')[-1])
    np_target.append(np.sum(arr, axis=(1, 2, 3))/arr.shape[1])
    np_data.append(family_to_images(families[ind]))

np_data = np.array(np_data)
np_target = np.array(np_target)


# normalize and split numpy data
print('preparing data')
train_fraction = 0.8
validate_fraction = 0.1
train_len = int(train_fraction*len(np_data))
valid_len = int(validate_fraction*len(np_data))

channel_mean = np.mean(np_data, axis=(0, 1, 3, 4))
channel_stdev = np.std(np_data, axis=(0, 1, 3, 4))


# make into pytorch data
import torchvision.transforms as transforms
from model import CircuitPairDataset
from torch.utils.data import DataLoader

train_circuits = np_data[:train_len]
train_target = np_target[:train_len]
valid_circuits = np_data[train_len + valid_len:]
valid_target = np_target[train_len + valid_len:]
test_circuits = np_data[train_len:train_len + valid_len]
test_target = np_target[train_len:train_len + valid_len]

transform = transforms.Normalize(channel_mean, channel_stdev)
train_dataset = CircuitPairDataset(train_circuits, train_target, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
valid_dataset = CircuitPairDataset(valid_circuits, valid_target, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
test_dataset = CircuitPairDataset(test_circuits, test_target, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())


# evaluate on valid and test datasets
print('creating predictions')
labels = np.zeros(0)
model2.eval()
predictions_early = np.zeros(0)
for i, data in enumerate(test_loader, 0):
    inputs, circuit_lengths, target = data
    inputs = inputs.cuda()
    circuit_lengths = circuit_lengths.cuda()
    outputs = model2(inputs)
    outputs = outputs.cpu().detach().numpy()
    outputs = np.reshape(outputs, len(outputs))
    
    labels = np.append(labels, target)
    predictions_early = np.append(predictions_early, outputs)

valid_labels = np.zeros(0)
model2.eval()
valid_predictions_early = np.zeros(0)
for i, data in enumerate(valid_loader, 0):
    inputs, circuit_lengths, target = data
    inputs = inputs.cuda()
    circuit_lengths = circuit_lengths.cuda()
    outputs = model2(inputs)
    outputs = outputs.cpu().detach().numpy()
    outputs = np.reshape(outputs, len(outputs))
    
    valid_labels = np.append(valid_labels, target)
    valid_predictions_early = np.append(valid_predictions_early, outputs)
    
    
# two-way prediction (average [c1, c2] with -[c2, c1])
def predict(circuit_pairs, target, asymmetric_predictions):
    pairs = target[np.unravel_index(circuit_pairs, target.shape)]
    shaped_target = pairs[:, 1] - pairs[:, 0]
    reverse_inds = np.zeros(len(pairs)).astype(int)
    for i in range(len(pairs)):
        b, a = circuit_pairs[i]
        ind = (a - a//21)*20 + (b - b//21)
        if a < b:
            ind -= 1
        reverse_inds[i] = ind
    predictions_ensemble = (asymmetric_predictions - asymmetric_predictions[reverse_inds])/2
    return predictions_ensemble, shaped_target
    
# data analysis
def plot(predicted, target):
    max_lim = np.percentile(target, 95)
    min_lim = np.percentile(target, 5)
    plt.scatter(target, predicted, alpha=0.01)
    x = np.linspace(min_lim, max_lim)
    plt.plot(x, x, color='orange')
    plt.xlim((min_lim, max_lim))
    plt.ylim((min_lim, max_lim))
    plt.xlabel('True ' + 'noise')
    plt.ylabel('Predicted ' + 'noise')

def show_results(predicted, target):
    print('MSE:', mean_squared_error(target, predicted))
    print('MAE:', mean_absolute_error(target, predicted))
    print('R^2:', r2_score(target, predicted))
#     plot(predicted, target)

# run analysis
print('analyzing')
valid_flat_preds, valid_flat_target = predict(valid_dataset.circuit_pairs, valid_target, valid_predictions_early)
test_flat_preds, test_flat_target = predict(test_dataset.circuit_pairs, test_target, predictions_early)
show_results(test_flat_preds, test_flat_target)