import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CircuitPairDataset
import numpy as np
import torchvision.transforms as transforms
from os import listdir
import torch.optim as optim
from torch.optim.lr_scheduler import *
import torch.backends.cudnn as cudnn
from model import SmallNet
import sys
import glob

# shape: (# channels, # layers, # qubits)
# only put u3 and cx channels; u1 and u2 are encoded in u3 channel since they are rare
def circuit_to_image(s3, n=5, n_channels=4, max_size=44):
    # create array to determine where gaps are, assuming latest-as-possible scheduling
    trimmed = s3[s3.find('creg'):s3.find('barrier')].splitlines()[1:]
    trimmed.reverse()
    circuit_arr = np.zeros((n, max_size), dtype=np.int64)
    circuit_img = np.zeros((n, max_size, n_channels))
    for ind in range(len(trimmed)):
        line = trimmed[ind]
        offset = 0
        qubits = []
        for i in range(2):
            q1_start = line.find('q', offset)
            if q1_start == -1:
                break
            q1_end = line.find(']', q1_start)
            qubit = line[q1_start+2:q1_end]
            qubits.append(int(qubit))
            offset = q1_end
        col_num = []
        for q in qubits:
            if len(np.nonzero(circuit_arr[q])[0]) != 0:
                col_num.append(np.max(np.nonzero(circuit_arr[q])))
        if len(col_num) == 0:
            col_num = 0
        else:
            col_num = np.max(col_num) + 1
        if col_num >= max_size:
            break
        for i in range(len(qubits)):
            q = qubits[i]
            circuit_arr[q][col_num] = 1 + ind
            if 'u1' in line:
                args = line[line.find('(') + 1 : line.find(')')].split(',')
                circuit_img[q][col_num][0] = eval(args[0])
                circuit_img[q][col_num][1] = -1
                circuit_img[q][col_num][2] = -1
            elif 'u2' in line:
                args = line[line.find('(') + 1 : line.find(')')].split(',')
                circuit_img[q][col_num][0] = eval(args[0])
                circuit_img[q][col_num][1] = eval(args[1])
                circuit_img[q][col_num][2] = -1
            elif 'u3' in line:
                args = line[line.find('(') + 1 : line.find(')')].split(',')
                circuit_img[q][col_num][0] = eval(args[0])
                circuit_img[q][col_num][1] = eval(args[1])
                circuit_img[q][col_num][2] = eval(args[2])
            elif 'cx' in line:
                circuit_img[q][col_num][3] = i*2 - 1
    
    image = np.swapaxes(circuit_img, 0, 2)
    return image
    
def family_to_images(family):
    images = []
    for f in family:
        images.append(circuit_to_image(f))
    return images

if __name__ == '__main__':
    noise_file = 'supremacy_all_5_unique/burlington_noise.npy'
    circuits_file = 'supremacy_all_5_unique/circuits.npy'
    model_dir = 'models/'
    evaluate_validation = True

    dropout = False
    parallel_data = True
    scheduled = True
    train_batch_size = 4
    concat_data = False
    epochs = 2000
    
    small_LR = False
    prefix = 'burlington'
    
    data_files_filename = model_dir + prefix + '_files.npy'
    
    prefix += '_batch' + str(train_batch_size)

    if dropout:
        prefix += '_drop'
    if scheduled:
        prefix += '_scheduledLR2'
    if concat_data:
        prefix += '_concatdata'
    if not evaluate_validation:
        prefix += '_alltrain'

    early_stop_filename = model_dir + prefix + '_early_stop.mdl'
    final_model_filename = model_dir + prefix + '_final_model.mdl'
    predictions_filename = model_dir + prefix + '_predictions.npy'
    labels_filename = model_dir + prefix + '_labels.npy'
    data_files_filename = model_dir + prefix + '_dataset_files.npy'
    train_fraction = 0.8
    validate_fraction = 0.1

    def make_model(n_channels, concat_features=False):
        if 'super' in prefix:
            return SuperSmallNet(n_channels=n_channels, concat_features=concat_features)
        elif 'small2' in prefix:
            return SmallNet2(n_channels=n_channels, concat_features=concat_features)
        elif 'small' in prefix:
            return SmallNet(n_channels=n_channels, concat_features=concat_features)
        else:
            return Net(n_channels=n_channels, concat_features=concat_features)
    
    
    np_target = np.load(noise_file)
    np_data = []
    raw_data = np.load(circuits_file)
    for f in raw_data:
        np_data.append(family_to_images(f))
    filenames = [noise_file, circuits_file]
    

    np_data = np.array(np_data)
    filenames = np.array(filenames)
    np.save(data_files_filename, np.array(filenames))
    print('data shape', np_data.shape)
    print('target shape', np_target.shape)
    
    # shuffle to remove time ordering of runs
    np.random.seed(0)
    order = np.arange(np_data.shape[0])
    np.random.shuffle(order)
    np_data = np_data[order]
    np_target = np_target[order]


    # normalize data
    channel_mean = np.mean(np_data, axis=(0, 1, 3, 4))
    channel_stdev = np.std(np_data, axis=(0, 1, 3, 4))
    transform = transforms.Normalize(channel_mean, channel_stdev)

    if evaluate_validation:
        train_len = int(train_fraction*len(np_data))
        valid_len = int(validate_fraction*len(np_data))
        
        test_dataset = CircuitPairDataset(np_data[train_len:train_len + valid_len], np_target[train_len:train_len + valid_len], transform=transform, reflect=False)
        valid_dataset = CircuitPairDataset(np_data[train_len + valid_len:], np_target[train_len + valid_len:], transform=transform, reflect=False)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
        valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    else:
        train_len = len(np_data)

    train_dataset = CircuitPairDataset(np_data[:train_len], np_target[:train_len], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    # create model
    model = make_model(2*len(np_data[0][0]))
    model.cuda()
    if parallel_data:
        model = nn.DataParallel(model)

    lr = 1.0e-2
    if scheduled:
        lr = 2.0e-2
        if train_batch_size > 32:
            lr = 0.2
    if train_batch_size <= 4 and small_LR:
        lr = 0.1e-2
    momentum = 0.9
    wd = 0
    if 'reg' in prefix:
        wd = 0.0001

    criterion = nn.MSELoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    cudnn.benchmark = True

    best_loss = -1
    losses = np.zeros((2, epochs))

    scheduler = None
    if scheduled:
        ss = 15
        g = 0.2
        if train_batch_size >= 64:
            ss = 30
            g = 0.5
        scheduler = StepLR(optimizer, step_size=ss, gamma=0.5)

    for epoch in range(epochs):
        # train model
        model.train()
        running_loss = 0.0
        count = 0
        for i, data in enumerate(train_loader, 0):
            inputs, circuit_lengths, labels = data
            optimizer.zero_grad()

            inputs = inputs.cuda()
            circuit_lengths = circuit_lengths.cuda()
            labels = labels.cuda()
            if concat_data:
                outputs = model(inputs, circuit_lengths)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1
        if scheduled:
            scheduler.step()
        train_loss = running_loss/count
        
        
        # evaluate model on validation set
        if evaluate_validation:
            model.eval()
            running_loss = 0.0
            count = 0
            for i, data in enumerate(valid_loader, 0):
                inputs, circuit_lengths, labels = data
                inputs = inputs.cuda()
                circuit_lengths = circuit_lengths.cuda()
                labels = labels.cuda()
                if concat_data:
                    outputs = model(inputs, circuit_lengths)
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                count += 1
            valid_loss = running_loss / count
            if (valid_loss < best_loss) or (best_loss == -1):
                best_loss = valid_loss
                torch.save({'epoch':epoch+1, 'state_dict':model.state_dict(), 'best_loss':best_loss}, early_stop_filename)
            losses[0][epoch] = train_loss
            losses[1][epoch] = valid_loss
            if ((epoch + 1) % 10 == 0) or ((epoch + 1) == epochs):
                torch.save({'epoch':epoch+1, 'state_dict':model.state_dict(), 'loss_history':losses}, final_model_filename)
            print(epoch + 1, train_loss, valid_loss, sep='\t')
        else:
            losses[0][epoch] = train_loss
            torch.save({'epoch':epoch+1, 'state_dict':model.state_dict(), 'loss_history':losses}, final_model_filename)
            print(epoch + 1, train_loss, sep='\t')

            
    if evaluate_validation:
        checkpoint = torch.load(early_stop_filename)
        model = make_model(2*len(np_data[0]), concat_data)
        model.cuda()
        if parallel_data:
            model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['state_dict'])

        # evaluate early stopping on test set
        model.eval()
        all_labels = np.zeros(0)
        all_predictions = np.zeros(0)
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data

            inputs = inputs.cuda()
            outputs = model(inputs)
            labels = np.reshape(labels.numpy(), len(labels))
            outputs = outputs.cpu().detach().numpy()
            outputs = np.reshape(outputs, len(outputs))

            all_labels = np.append(all_labels, labels)
            all_predictions = np.append(all_predictions, outputs)


        np.save(predictions_filename, all_predictions)
        np.save(labels_filename, all_labels)