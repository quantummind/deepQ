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
from model import resnet_tiny, resnet18, resnet34, resnet50, resnet101, resnet152
import sys
import glob

# converts Rigetti I, RX, RZ, CZ set to 4-channel image
def circuit_to_image(c):
    image = np.zeros((4, c.shape[0], c.shape[1]))
    pi = np.pi
    for i in range(len(c)):
        for j in range(len(c[i])):
            if c[i][j] == 'I':
                image[0][i][j] = 1
            elif 'RX' in c[i][j]:
                image[1][i][j] = eval(c[i][j][3:-1])
            elif 'RZ' in c[i][j]:
                image[2][i][j] = eval(c[i][j][3:-1])
            elif 'CZ' in c[i][j]:
                image[3][i][j] = 1.0
            elif c[i][j] == '':
                image[3][i][j] = -1.0
    return image
    
def family_to_images(family):
    images = []
    for f in family:
        images.append(circuit_to_image(f))
    return images

if __name__ == '__main__':
    data_prefix = 'output/simulations-random-5/readouts1000-noisy-'
    families_file = 'output/circuit-families-random-5.npz'
    model_dir = 'output/models/'


    resnet_depth = 0
    dropout = False
    parallel_data = True
    scheduled = False
    train_batch_size = 16
    concat_data = False
    epochs = 2000
    prefix = 'random5'

    prefix += '_batch' + str(train_batch_size)
    prefix += '_' + str(resnet_depth)

    if dropout:
        prefix += '_drop'
    if scheduled:
        prefix += '_scheduledLR2'
    if concat_data:
        prefix += '_concatdata'

    early_stop_filename = model_dir + prefix + '_early_stop.mdl'
    final_model_filename = model_dir + prefix + '_final_model.mdl'
    predictions_filename = model_dir + prefix + '_predictions.npy'
    labels_filename = model_dir + prefix + '_labels.npy'
    data_files_filename = model_dir + prefix + '_dataset_files.npy'
    train_fraction = 0.8
    validate_fraction = 0.1

    def make_model(n_channels, concat_features=False):
        if resnet_depth == 18:
            return resnet18(n_channels=n_channels, use_dropout=dropout, concat_features=concat_features)
        elif resnet_depth == 34:
            return resnet34(n_channels=n_channels, use_dropout=dropout, concat_features=concat_features)
        elif resnet_depth == 50:
            return resnet50(n_channels=n_channels, use_dropout=dropout, concat_features=concat_features)
        elif resnet_depth == 101:
            return resnet101(n_channels=n_channels, use_dropout=dropout, concat_features=concat_features)
        elif resnet_depth == 152:
            return resnet152(n_channels=n_channels, use_dropout=dropout, concat_features=concat_features)
        elif resnet_depth == 0:
            return resnet_tiny(n_channels=n_channels, use_dropout=dropout, concat_features=concat_features)

    families = np.load(families_file)['arr_0']
    np_data = []
    np_target = []
    filenames = []

    # load data
    files = sorted(glob.glob(data_prefix + '*'))
    if len(files) == 0:
        print('NO DATA')

    family_size = len(np.load(files[0]))
    for i in range(len(files)):
        arr = np.load(files[i])
        if len(arr) == family_size:
            filenames.append(files[i])
            ind = int(files[i].split('.')[0].split('-')[-1])
            np_target.append(np.sum(arr, axis=(1, 2, 3))/arr.shape[1])
            np_data.append(family_to_images(families[ind]))

    np_data = np.array(np_data)
    np_target = np.array(np_target)
    filenames = np.array(filenames)
    np.save(data_files_filename, np.array(filenames))
    print('data shape', np_data.shape)
    print('target shape', np_target.shape)


    # normalize data
    channel_mean = np.mean(np_data, axis=(0, 1, 3, 4))
    channel_stdev = np.std(np_data, axis=(0, 1, 3, 4))
    transform = transforms.Normalize(channel_mean, channel_stdev)

    train_len = int(train_fraction*len(np_data))
    valid_len = int(validate_fraction*len(np_data))

    train_dataset = CircuitPairDataset(np_data[:train_len], np_target[:train_len], transform=transform)
    test_dataset = CircuitPairDataset(np_data[train_len:train_len + valid_len], np_target[train_len:train_len + valid_len], transform=transform)
    valid_dataset = CircuitPairDataset(np_data[train_len + valid_len:], np_target[train_len + valid_len:], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(valid_dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    # create model
    model = make_model(2*len(np_data[0][0]))
    model.cuda()
    if parallel_data:
        model = nn.DataParallel(model)

    lr = 1.0e-2
    if scheduled:
        lr = 5.0e-2
    momentum = 0.9

    criterion = nn.L1Loss().cuda()
    # optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    cudnn.benchmark = True

    best_loss = -1
    losses = np.zeros((2, epochs))

    scheduler = None
    if scheduled:
        scheduler = StepLR(optimizer, step_size=15, gamma=0.2)

    # TODO make into cross-validation with nfolds
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