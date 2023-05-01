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
        
        self.rz1 = tq.RZ(has_params=True, trainable=True)
        self.rz2 = tq.RZ(has_params=True, trainable=True)
        self.rz3 = tq.RZ(has_params=True, trainable=True)
        self.cnot1 = tq.CNOT(has_params=True, trainable=True)
        self.cnot2 = tq.CNOT(has_params=True, trainable=True)
        self.cnot3 = tq.CNOT(has_params=True, trainable=True)
        self.ry1 = tq.RY(has_params=True, trainable=True)
        self.ry2 = tq.RY(has_params=True, trainable=True)

        self.rz4 = tq.RZ(has_params=True, trainable=True)
        self.rz5 = tq.RZ(has_params=True, trainable=True)
        self.rz6 = tq.RZ(has_params=True, trainable=True)
        self.cnot4 = tq.CNOT(has_params=True, trainable=True)
        self.cnot5 = tq.CNOT(has_params=True, trainable=True)
        self.cnot6 = tq.CNOT(has_params=True, trainable=True)
        self.ry3 = tq.RY(has_params=True, trainable=True)
        self.ry4 = tq.RY(has_params=True, trainable=True)

        self.rz7 = tq.RZ(has_params=True, trainable=True)
        self.rz8 = tq.RZ(has_params=True, trainable=True)
        self.rz9 = tq.RZ(has_params=True, trainable=True)
        self.cnot7 = tq.CNOT(has_params=True, trainable=True)
        self.cnot8 = tq.CNOT(has_params=True, trainable=True)
        self.cnot9 = tq.CNOT(has_params=True, trainable=True)
        self.ry5 = tq.RY(has_params=True, trainable=True)
        self.ry6 = tq.RY(has_params=True, trainable=True)

        self.rz10 = tq.RZ(has_params=True, trainable=True)
        self.rz11 = tq.RZ(has_params=True, trainable=True)
        self.rz12 = tq.RZ(has_params=True, trainable=True)
        self.cnot10 = tq.CNOT(has_params=True, trainable=True)
        self.cnot11 = tq.CNOT(has_params=True, trainable=True)
        self.cnot12 = tq.CNOT(has_params=True, trainable=True)
        self.ry7 = tq.RY(has_params=True, trainable=True)
        self.ry8 = tq.RY(has_params=True, trainable=True)

        # first pooling layer
        self.u3_0 = tq.U3(has_params=True, trainable=True)
        self.u3_1 = tq.U3(has_params=True, trainable=True)
        self.u3_2 = tq.U3(has_params=True, trainable=True)
        self.u3_3 = tq.U3(has_params=True, trainable=True)

        # second convolutional layer
       
        self.rz13 = tq.RZ(has_params=True, trainable=True)
        self.rz14 = tq.RZ(has_params=True, trainable=True)
        self.rz15 = tq.RZ(has_params=True, trainable=True)
        self.cnot13 = tq.CNOT(has_params=True, trainable=True)
        self.cnot14 = tq.CNOT(has_params=True, trainable=True)
        self.cnot15 = tq.CNOT(has_params=True, trainable=True)
        self.ry9 = tq.RY(has_params=True, trainable=True)
        self.ry10 = tq.RY(has_params=True, trainable=True)

        self.rz16 = tq.RZ(has_params=True, trainable=True)
        self.rz17 = tq.RZ(has_params=True, trainable=True)
        self.rz18 = tq.RZ(has_params=True, trainable=True)
        self.cnot16 = tq.CNOT(has_params=True, trainable=True)
        self.cnot17 = tq.CNOT(has_params=True, trainable=True)
        self.cnot18 = tq.CNOT(has_params=True, trainable=True)
        self.ry11 = tq.RY(has_params=True, trainable=True)
        self.ry12 = tq.RY(has_params=True, trainable=True)

        self.rz19 = tq.RZ(has_params=True, trainable=True)
        self.rz20 = tq.RZ(has_params=True, trainable=True)
        self.rz21 = tq.RZ(has_params=True, trainable=True)
        self.cnot19 = tq.CNOT(has_params=True, trainable=True)
        self.cnot20 = tq.CNOT(has_params=True, trainable=True)
        self.cnot21 = tq.CNOT(has_params=True, trainable=True)
        self.ry13 = tq.RY(has_params=True, trainable=True)
        self.ry14 = tq.RY(has_params=True, trainable=True)

        # second pooling layer
        self.u3_4 = tq.U3(has_params=True, trainable=True)
        self.u3_5 = tq.U3(has_params=True, trainable=True)

        #multilevel perceptron layer
        self.mlp_class = nn.Sequential(nn.Linear(2, 10), nn.Tanh(), nn.Linear(10, 1))

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
        
        # first convolutional layer
       
        self.rz1(qdev, wires = 1)
        self.cnot1(qdev, wires = [1,0])
        self.rz2(qdev, wires = 0)
        self.ry1(qdev, wires = 1)
        self.cnot2(qdev, wires = [0,1])
        self.ry2(qdev, wires = 1)
        self.cnot3(qdev, wires = [1,0])
        self.rz3(qdev, wires = 0)

        self.rz4(qdev, wires = 3)
        self.cnot4(qdev, wires = [3,2])
        self.rz5(qdev, wires = 2)
        self.ry3(qdev, wires = 3)
        self.cnot5(qdev, wires = [2,3])
        self.ry4(qdev, wires = 3)
        self.cnot6(qdev, wires = [3,2])
        self.rz6(qdev, wires = 2)

        self.rz7(qdev, wires = 5)
        self.cnot7(qdev, wires = [5,4])
        self.rz8(qdev, wires = 4)
        self.ry5(qdev, wires = 5)
        self.cnot8(qdev, wires = [4,5])
        self.ry6(qdev, wires = 5)
        self.cnot9(qdev, wires = [5,4])
        self.rz9(qdev, wires = 4)

        self.rz10(qdev, wires = 7)
        self.cnot10(qdev, wires = [7,6])
        self.rz11(qdev, wires = 6)
        self.ry7(qdev, wires = 7)
        self.cnot11(qdev, wires = [6,7])
        self.ry8(qdev, wires = 7)
        self.cnot12(qdev, wires = [7,6])
        self.rz12(qdev, wires = 6)


        # first pooling layer
        meas_qubits = [0,2,4,6]
        _  = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        self.u3_0(qdev, wires = 1)
        self.u3_1(qdev, wires = 3)
        self.u3_2(qdev, wires = 5)
        self.u3_3(qdev, wires = 7)

        # second convolutional layer
        
        self.rz13(qdev, wires = 3)
        self.cnot13(qdev, wires = [3,1])
        self.rz14(qdev, wires = 1)
        self.ry9(qdev, wires = 3)
        self.cnot14(qdev, wires = [1,3])
        self.ry10(qdev, wires = 3)
        self.cnot15(qdev, wires = [3,1])
        self.rz15(qdev, wires = 1)

        self.rz16(qdev, wires = 7)
        self.cnot16(qdev, wires = [7,5])
        self.rz17(qdev, wires = 5)
        self.ry11(qdev, wires = 7)
        self.cnot17(qdev, wires = [5,7])
        self.ry12(qdev, wires = 7)
        self.cnot18(qdev, wires = [7,5])
        self.rz18(qdev, wires = 5)

        self.rz19(qdev, wires = 5)
        self.cnot19(qdev, wires = [5,3])
        self.rz20(qdev, wires = 3)
        self.ry13(qdev, wires = 5)
        self.cnot20(qdev, wires = [3,5])
        self.ry14(qdev, wires = 5)
        self.cnot21(qdev, wires = [5,3])
        self.rz21(qdev, wires = 3)

        # second pooling layer
        meas_qubits = [1,5]
        _ = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))
        self.u3_4(qdev, wires = 3)
        self.u3_5(qdev, wires = 7)

        # perform measurement to get expectations (back to classical domain)
        meas_qubits = [3,7]
        x = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        # classification
        x = self.mlp_class(x)
        x = torch.sigmoid(x)
        return x
    

    def generate_data(self, n_points = 400, train_split = 0.7, save = True):
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
    
def train(trainloader, train_labels, epochs = 10, n_qubits = 8, n_cycles = 4, lr = 5e-3, device = 'cpu'):
    
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
            print(f"epoch: {epoch}, loss: {loss.item()}", end="\r")
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
