# Preparing states to classify

import torch
import pickle
import numpy as np
import torchquantum.functional as tqf
from torch.utils.data import Dataset, DataLoader

class MajoranaCircuit:
    def __init__(self, n_qubits = 8, n_cycles = 4):
        self.n_qubits = n_qubits
        self.n_cycles = n_cycles
        
    def generate_circuit(self, qdev, theta, phi):
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
        return qdev

    def generate_label(self, theta, phi):
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

# Custom dataset class for Majorana modes

class InputDataset(Dataset):
    """Dataset of (non)topological Majorana states. 
    I need to make this in order to pass to Pytorch's `DataLoader` class.
    """

    def __init__(self, fname):
        """
        Arguments:
        """
        with open(fname, 'rb') as brick:
            self.input = pickle.load(brick)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.input[idx]
        return sample