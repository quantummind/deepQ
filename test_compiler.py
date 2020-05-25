import numpy as np
from qiskit import *
import torch
from model import SmallNet
from train_5qubit import circuit_to_image, family_to_images
import torchvision.transforms as transforms
from model import CircuitPairDataset
from torch.utils.data import DataLoader
import glob
from generate_circuits import pad_circuit
from multiprocessing import Pool
import random

parallel_data = True
prefixes = ['models/burlington_batch4_scheduledLR2_']
n_files = -1
model_type = SmallNet
    
worst = False

# tries = 20
# tournament_size = 10

tries = 1000
tournament_size = 20


pool_size = 25
thread_size = tries // pool_size
test_circuits = np.load('test_circuits_5.npy')

group_size = 4
n_channels = 4

models = []
for prefix in prefixes:
#     checkpoint2 = torch.load(prefix + 'final_model.mdl')
    checkpoint2 = torch.load(prefix + 'early_stop.mdl')
    model2 = model_type(n_channels=2*n_channels, concat_features=False)
    model2.cuda()
    if parallel_data:
        model2 = torch.nn.DataParallel(model2)
    model2.load_state_dict(checkpoint2['state_dict'], strict=False)
    model2.eval()
    models.append(model2)
    

noise_file = 'supremacy_all_5_unique/burlington_noise.npy'
circuits_file = 'supremacy_all_5_unique/circuits.npy'

np_target = np.load(noise_file)
np_data = []
raw_data = np.load(circuits_file)
for f in raw_data:
    np_data.append(family_to_images(f))

# get data normalization
np_data = np.array(np_data)
channel_mean = np.mean(np_data, axis=(0, 1, 3, 4))
channel_stdev = np.std(np_data, axis=(0, 1, 3, 4))
transform = transforms.Normalize(channel_mean, channel_stdev)
np_data = None # clear variable

def predict(circuit_data):
    fake_target = np.zeros((circuit_data.shape[0], circuit_data.shape[1]))
    dataset = CircuitPairDataset(circuit_data, fake_target, transform=transform, reflect=False)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    
    predictions = np.zeros(0)
    for i, d in enumerate(loader, 0):
        inputs, circuit_lengths, target = d
        inputs = inputs.cuda()
        circuit_lengths = circuit_lengths.cuda()
        target = np.reshape(target.numpy(), len(target))
        all_outputs = []
        for model2 in models:
            outputs = model2(inputs)
            outputs = outputs.cpu().detach().numpy()
            outputs = np.reshape(outputs, len(outputs))
            all_outputs.append(outputs)
        outputs = np.mean(all_outputs, axis=0)

        predictions = np.append(predictions, outputs)
    return predictions

def find_best(circuit_data, predictions):
    if worst:
        predictions = -predictions
    best_circuits = []
    for i in range(circuit_data.shape[0]):
        best_j = 0
        best_delta = 0
        pads = circuit_data.shape[1]
        for j in range(pads):
            avg_delta = np.average(predictions[(i*pads+j)*(pads-1):(i*pads+j+1)*(pads-1)])
            if avg_delta > best_delta:
                best_j = j
                best_delta = avg_delta
        best_circuits.append(best_j)
    best_circuits = np.array(best_circuits)
    return best_circuits

def pad_family(s, length=thread_size):
    family = []
    for i in range(length):
        if random.uniform(0, 1) > 1/16:
            family.append(pad_circuit(s, backend, n=5))
        else:
            family.append(s)
    random.shuffle(family)
    return family, family_to_images(family)

if __name__ == '__main__':
    IBMQ.load_account()
    provider = IBMQ.get_provider(group='open')
    backend = provider.get_backend('ibmq_burlington')

    # TODO make more intelligent: try swapping padding zones of best ones
    compiled_circuits = []
    valid_circuits = []
    
    for ind in range(len(test_circuits)):
        s = test_circuits[ind]
        print(ind, 'running first tournament')
        data = []
        all_families = []
        
        args = [s]*(pool_size)
        with Pool(pool_size) as p:
            out = p.map(pad_family, args)
        
        out_family = []
        out_data = []
        for i in range(len(out)):
            entry = out[i]
            if entry[1] is not None:
                out_family.extend(entry[0])
                out_data.extend(entry[1])
        
        count = 0
        family = []
        data_entry = []
        for i in range(len(out_family)):
            family.append(out_family[i])
            data_entry.append(out_data[i])
            count += 1
            if count % tournament_size == 0:
                all_families.append(family)
                data.append(data_entry)
                family = []
                data_entry = []
                
        data = np.array(data)
        print(data.shape)
        predictions = predict(data)
        best_circuits = find_best(data, predictions)

        print('running final tournament')
        family = []
        for i in range(len(data)):
            family.append(all_families[i][best_circuits[i]])
        data = np.expand_dims(np.array(family_to_images(family)), axis=0)
        predictions = predict(data)
        best_circuit = find_best(data, predictions)[0]
        compiled_circuits.append(family[best_circuit])
        valid_circuits.append(s)

        np.save('test_5_burlington_compiled.npy', compiled_circuits)
        np.save('test_5_burlington_free.npy', valid_circuits)