# Data generation and training/test functions

import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from source.state_prep import InputDataset
from tqdm.notebook import tqdm

def generate_data(model, n_points = 400, train_split = 0.7, save = True):
    """ 
    Generate training and testing data.
        Args: 
        n_points (int): the number of circuits for both training/testing.
            Default is 200
        train_split (float): the proportion of circuits that will be used 
            for testing. Default is 0.7
        save (bool): whether to save the generated data
        Returns:
        train_loader, train_labels, test_loader, test_labels
    """
    datapoints = []
    labels = torch.empty([n_points,1])
    for i in range(n_points): # produce
        # generates random values of the gate angles theta and phi
        rng_theta = np.random.uniform(0,np.pi/2)
        rng_phi = np.random.uniform(0,np.pi/2)
        x = [rng_theta, rng_phi]

        datapoints.append(x)
        if model.circuit_builder.generate_label(rng_theta, rng_phi) == 1:
            labels[i] = 1
        else:
            labels[i] = 0

    # partitions circuit list into test and train sets
    split_ind = int(len(datapoints) * train_split)
    train_data = datapoints[:split_ind]
    test_data = datapoints[split_ind:]

    train_labels = labels[:split_ind]
    test_labels = labels[split_ind:]

    if save:
        # saving data
        with open('./data/train_data.pkl', 'wb') as writer:
            pickle.dump(train_data, writer)
        with open('./data/test_data.pkl', 'wb') as writer:
            pickle.dump(test_data, writer)

    # loading data
    test_dataset = InputDataset(fname='./data/test_data.pkl')
    test_loader = DataLoader(test_dataset, batch_size=1,
                    shuffle=False, num_workers=0)
    
    train_dataset = InputDataset(fname='./data/train_data.pkl')
    train_loader = DataLoader(train_dataset, batch_size=1,
                    shuffle=False, num_workers=0)

    return train_loader, train_labels, test_loader, test_labels


def get_accuracy(preds, labels, cutoff = 0.2):
    score = 0
    count = 0
    for i, pred in enumerate(preds):
        if 1-cutoff < pred and labels[i] == 1:
            score += 1
        if pred < cutoff and labels[i] == 0:
            score += 1
        count += 1
    return score/count
   


# Testing and Training functions
    
def train(model, trainloader, train_labels, epochs = 10, lr = 5e-3, device = 'cpu'):
    
    # calling model, loss, optimizer
    lossfn = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training loop
    preds = []
    for epoch in tqdm(range(epochs)):
        for i, data in enumerate(trainloader):
            # loading data
            input = data #.to(device)
            target = train_labels[i]

            # running QCNN on input state
            pred = model(input)[0]
            if epoch == epochs-1:
                preds.append(pred)

            # calculating loss and backpropogating
            loss = lossfn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # print(f"epoch: {epoch}, loss: {loss.item()}", end="\r")

        # changed to print each line
        print("epoch: " + str(epoch) + ", loss: " + str(loss.item()))
    
    return model, preds

def test(model, testloader, test_labels):
    model.eval()
    preds = []
    for i, data in enumerate(testloader):
        # loading data
        input = data
        pred = model(input)[0]
        preds.append(pred)
    accuracy = get_accuracy(preds, test_labels)
    return preds, accuracy
