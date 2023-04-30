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
import copy
from shared import sharedWeights

class QCNN(nn.Module):
    def __init__(self, n_qubits = 8, n_cycles = 4, weights = torch.randn(10)):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_cycles = n_cycles
        self.weights = weights

        self.meas_basis = tq.PauliZ

        # first convolutional layer
        self.cr0 = sharedWeights(self.weights[0])
        self.cr1 = sharedWeights(self.weights[1])
        self.cr2 = sharedWeights(self.weights[2])
        self.cr3 = sharedWeights(self.weights[3])
        self.cr4 = sharedWeights(self.weights[4])
        self.cr5 = sharedWeights(self.weights[5])
        self.cr6 = sharedWeights(self.weights[6])

        # first pooling layer
        self.u3_0 = tq.U3(has_params=True, trainable=True)
        self.u3_1 = tq.U3(has_params=True, trainable=True)
        self.u3_2 = tq.U3(has_params=True, trainable=True)
        self.u3_3 = tq.U3(has_params=True, trainable=True)

        # second convolutional layer
        self.cr7 = sharedWeights(self.weights[7])
        self.cr8 = sharedWeights(self.weights[8])
        self.cr9 = sharedWeights(self.weights[9])

        # second pooling layer
        self.u3_4 = tq.U3(has_params=True, trainable=True)
        self.u3_5 = tq.U3(has_params=True, trainable=True)

        # Second Filter
        # first pooling layer
        self.u3_10 = tq.U3(has_params=True, trainable=True)
        self.u3_11 = tq.U3(has_params=True, trainable=True)
        self.u3_12 = tq.U3(has_params=True, trainable=True)
        self.u3_13 = tq.U3(has_params=True, trainable=True)

        # second pooling layer
        self.u3_14 = tq.U3(has_params=True, trainable=True)
        self.u3_15 = tq.U3(has_params=True, trainable=True)

        # Third Filter
        # first pooling layer
        self.u3_20 = tq.U3(has_params=True, trainable=True)
        self.u3_21 = tq.U3(has_params=True, trainable=True)
        self.u3_22 = tq.U3(has_params=True, trainable=True)
        self.u3_23 = tq.U3(has_params=True, trainable=True)

        # second pooling layer
        self.u3_24 = tq.U3(has_params=True, trainable=True)
        self.u3_25 = tq.U3(has_params=True, trainable=True)        

        #multilevel perceptron layer
        self.mlp_class = nn.Sequential(nn.Linear(6, 12), nn.Tanh(), nn.Linear(12, 1))

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

        # Create copies of circuit for parallel feature maps
        qdev1 = copy.deepcopy(qdev)
        qdev2 = copy.deepcopy(qdev)

        # STEP 1: add trainable gates for QCNN circuit
        
        # first convolutional layer
        self.cr0(qdev, "CRX", 0, 1)
        self.cr1(qdev, "CRX", 2, 3)
        self.cr2(qdev, "CRX", 4, 5)
        self.cr3(qdev, "CRX", 6, 7)
        self.cr4(qdev, "CRX", 1, 2)
        self.cr5(qdev, "CRX", 3, 4)
        self.cr6(qdev, "CRX", 5, 6)

        # first pooling layer
        meas_qubits = [0,2,4,6]
        _  = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        self.u3_0(qdev, wires = 1)
        self.u3_1(qdev, wires = 3)
        self.u3_2(qdev, wires = 5)
        self.u3_3(qdev, wires = 7)

        # second convolutional layer
        self.cr7(qdev, "CRX", 1, 3)
        self.cr8(qdev, "CRX", 5, 7)
        self.cr9(qdev, "CRX", 3, 5)

        # second pooling layer
        meas_qubits = [1,5]
        _ = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))
        self.u3_4(qdev, wires = 3)
        self.u3_5(qdev, wires = 7)

        # perform measurement to get expectations (back to classical domain)
        meas_qubits = [3,7]
        x = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        # SAME OPERATIONS BUT FOR THE OTHER FEATURE MAPS

        # first convolutional layer
        self.cr0(qdev1, "CRY", 0, 1)
        self.cr1(qdev1, "CRY", 2, 3)
        self.cr2(qdev1, "CRY", 4, 5)
        self.cr3(qdev1, "CRY", 6, 7)
        self.cr4(qdev1, "CRY", 1, 2)
        self.cr5(qdev1, "CRY", 3, 4)
        self.cr6(qdev1, "CRY", 5, 6)

        # first pooling layer
        meas_qubits = [0,2,4,6]
        _  = tqm.expval(qdev1, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        self.u3_10(qdev1, wires = 1)
        self.u3_11(qdev1, wires = 3)
        self.u3_12(qdev1, wires = 5)
        self.u3_13(qdev1, wires = 7)

        # second convolutional layer
        self.cr7(qdev1, "CRY", 1, 3)
        self.cr8(qdev1, "CRY", 5, 7)
        self.cr9(qdev1, "CRY", 3, 5)

        # second pooling layer
        meas_qubits = [1,5]
        _ = tqm.expval(qdev1, meas_qubits, [self.meas_basis()] * len(meas_qubits))
        self.u3_14(qdev1, wires = 3)
        self.u3_15(qdev1, wires = 7)

        # perform measurement to get expectations (back to classical domain)
        meas_qubits = [3,7]
        y = tqm.expval(qdev1, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        # first convolutional layer
        self.cr0(qdev2, "CRZ", 0, 1)
        self.cr1(qdev2, "CRZ", 2, 3)
        self.cr2(qdev2, "CRZ", 4, 5)
        self.cr3(qdev2, "CRZ", 6, 7)
        self.cr4(qdev2, "CRZ", 1, 2)
        self.cr5(qdev2, "CRZ", 3, 4)
        self.cr6(qdev2, "CRZ", 5, 6)

        # first pooling layer
        meas_qubits = [0,2,4,6]
        _  = tqm.expval(qdev2, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        self.u3_20(qdev2, wires = 1)
        self.u3_21(qdev2, wires = 3)
        self.u3_22(qdev2, wires = 5)
        self.u3_23(qdev2, wires = 7)

        # second convolutional layer
        self.cr7(qdev2, "CRZ", 1, 3)
        self.cr8(qdev2, "CRZ", 5, 7)
        self.cr9(qdev2, "CRZ", 3, 5)

        # second pooling layer
        meas_qubits = [1,5]
        _ = tqm.expval(qdev2, meas_qubits, [self.meas_basis()] * len(meas_qubits))
        self.u3_24(qdev2, wires = 3)
        self.u3_25(qdev2, wires = 7)

        # perform measurement to get expectations (back to classical domain)
        meas_qubits = [3,7]
        z = tqm.expval(qdev2, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        # classification
        result = self.mlp_class(torch.cat((x,y,z), 1))
        result = torch.sigmoid(result)
        return result
    
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
    
def train(trainloader, train_labels, epochs = 10, n_qubits = 8, n_cycles = 4, lr = 9e-3, device = 'cpu'):
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