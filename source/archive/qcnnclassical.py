# QCNN source code

import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import torchquantum.measurement as tqm

class QCNN(nn.Module):
    def __init__(self, n_qubits = 8, n_cycles = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_cycles = n_cycles

        self.meas_basis = tq.PauliZ

        # first convolutional layer
        self.crx0 = tq.CRX(has_params=True, trainable=True)
        self.crx1 = tq.CRX(has_params=True, trainable=True)
        self.crx2 = tq.CRX(has_params=True, trainable=True)
        self.crx3 = tq.CRX(has_params=True, trainable=True)
        self.crx4 = tq.CRX(has_params=True, trainable=True)
        self.crx5 = tq.CRX(has_params=True, trainable=True)
        self.crx6 = tq.CRX(has_params=True, trainable=True)

        # first pooling layer
        self.u3_0 = tq.U3(has_params=True, trainable=True)
        self.u3_1 = tq.U3(has_params=True, trainable=True)
        self.u3_2 = tq.U3(has_params=True, trainable=True)
        self.u3_3 = tq.U3(has_params=True, trainable=True)

        # second convolutional layer
        self.crx7 = tq.CRX(has_params=True, trainable=True)
        self.crx8 = tq.CRX(has_params=True, trainable=True)
        self.crx9 = tq.CRX(has_params=True, trainable=True)

        # second pooling layer
        self.u3_4 = tq.U3(has_params=True, trainable=True)
        self.u3_5 = tq.U3(has_params=True, trainable=True)

        #multilevel perceptron layer
        self.mlp_class = nn.Sequential(nn.Linear(2, 10), nn.Tanh(), nn.Linear(10, 1))
        self.mlp_classical = nn.Sequential(nn.Linear(8,16), nn.Tanh(),nn.Linear(16,1))
    def forward(self, x):
        """x is a list with [theta, phi]"""
        theta = x[0]
        phi = x[1]

        # create a quantum device to run the gates
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, device = 'cpu')

        # encode the circuit from the parameters
        for qub in range(self.n_qubits):
            tqf.hadamard(qdev,wires= qub)

        for _ in range(self.n_cycles):
            # for qubit in range(self.n_qubits): #DO Z GATE ON EVERY QUBIT
            for qub in range(self.n_qubits):
                tqf.rz(qdev, wires = qub, params = 2*phi)
            for num in range(self.n_qubits): #DO A XX GATE FOR EVERY NEIGHBORING PAIR OF QUBITS
                if num%2 ==1:
                    if num-1 < 0:
                        pass
                    else:
                        tqf.rxx(qdev, wires =[num-1,num], params = 2*theta)
            for num in range(self.n_qubits): 
                if num%2 ==0:
                    if num-1 < 0:
                        pass
                    else:
                        tqf.rxx(qdev, wires =[num-1,num], params = 2*theta)

        # STEP 1: add trainable gates for QCNN circuit
        # base case comparison to classical neural network classification
        # # first convolutional layer
        # self.crx0(qdev, wires=[0, 1])
        # self.crx1(qdev, wires=[2, 3])
        # self.crx2(qdev, wires=[4, 5])
        # self.crx3(qdev, wires=[6, 7])
        # self.crx4(qdev, wires=[1, 2])
        # self.crx5(qdev, wires=[3, 4])
        # self.crx6(qdev, wires=[5, 6])

        # # first pooling layer
        # meas_qubits = [0,2,4,6]
        # _  = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        # self.u3_0(qdev, wires = 1)
        # self.u3_1(qdev, wires = 3)
        # self.u3_2(qdev, wires = 5)
        # self.u3_3(qdev, wires = 7)

        # # second convolutional layer
        # self.crx7(qdev, wires=[1, 3])
        # self.crx8(qdev, wires=[5, 7])
        # self.crx9(qdev, wires=[3, 5])

        # # second pooling layer
        # meas_qubits = [1,5]
        # _ = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))
        # self.u3_4(qdev, wires = 3)
        # self.u3_5(qdev, wires = 7)

        # perform measurement to get expectations (back to classical domain)
        meas_qubits = [0,1,2,3,4,5,6,7]
        x = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        # classification
        x = self.mlp_classical(x)
        x = torch.sigmoid(x)
        return x
    
    def generate_data(self, n_points = 500, train_split = 0.8, save = True):
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
            if QCNN.topological_classifier(rng_theta, rng_phi) == 1:
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
        test_dataset = MajoranaDataset(fname='./data/test_data.pkl')
        test_loader = DataLoader(test_dataset, batch_size=1,
                        shuffle=False, num_workers=0)
        
        train_dataset = MajoranaDataset(fname='./data/train_data.pkl')
        train_loader = DataLoader(train_dataset, batch_size=1,
                        shuffle=False, num_workers=0)

        return train_loader, train_labels, test_loader, test_labels
    
    @staticmethod
    def topological_classifier(theta, phi):
        """Given some theta and phi, classifies whether a state is in topological or trivial regime.
            Args:
            theta (float): angle of two-qubit RXX gates
            phi (float): angle of single-qubit RZ gates
            Returns:
            label (int): 1 if state is in topological (MZM, MPM, MZM+MPM) regime; 
                         0 if state is in trivial regime 
            """
        if theta <= np.pi/4:
            if phi <= np.pi/4 and theta <= phi:
                label = 0
            elif np.pi/4 < phi and theta < (np.pi/2-phi):
                label = 0
            else:
                label = 1
        else:
            label = 1
        return label
    
    @staticmethod
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
    
# Custom dataset class for Majorana modes

class MajoranaDataset(Dataset):
    """Dataset of (non)topological Majorana states. 
    I need to make this in order to pass to Pytorch's `DataLoader` class.
    """

    def __init__(self, fname):
        """
        Arguments:
        """
        with open(fname, 'rb') as brick:
            self.angles_array = pickle.load(brick)

    def __len__(self):
        return len(self.angles_array)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.angles_array[idx]
        return sample


# Testing and Training functions
    
def train(trainloader, train_labels, epochs = 10, n_qubits = 8, n_cycles = 4, lr = 5e-2, device = 'cpu'):
    # calling model, loss, optimizer
    model = QCNN(n_qubits, n_cycles)
    lossfn = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # training loop
    preds = []
    for epoch in range(epochs):
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
            print(f"epoch: {epoch}, loss: {loss.item()}")
    return model, preds

def test(model, testloader, test_labels):
    model.eval()
    preds = []
    for i, data in enumerate(testloader):
        # loading data
        input = data
        pred = model(input)[0]
        preds.append(pred)
    accuracy = QCNN.get_accuracy(preds, test_labels)
    return preds, accuracy