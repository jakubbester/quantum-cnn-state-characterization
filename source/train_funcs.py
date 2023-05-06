# Data generation and training/test functions

import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from source.state_prep import InputDataset
from tqdm.notebook import tqdm

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

# testing and training functions
def train(model, trainloader, train_labels, epochs=10, lr=5e-3, device = 'cpu'):
    
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
        print(f"epoch: {epoch}, loss: {loss.item()}") # changed to print each line
    accuracy = get_accuracy(preds, train_labels)
    return model, preds, accuracy

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
