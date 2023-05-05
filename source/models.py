# QCNN Source Code

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
from source.state_prep import *
import copy
from .shared import SharedWeights

##############################################
## Naive Implementations (for benchmarking) ##
##############################################

class Classical_NN(nn.Module):
    """ This is a naive classical NN implementation, 
        where we instantly measure all of the qubits
        of the input quantum state $\rho$ and feed
        them into a MLP layer.
    """
    def __init__(self, circuit_builder, n_qubits = 8, n_cycles = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_cycles = n_cycles
        self.circuit_builder = circuit_builder(n_qubits, n_cycles)

        self.meas_basis = tq.PauliZ

        self.mlp_classical = nn.Sequential(nn.Linear(n_qubits, n_qubits*2), nn.Tanh(),nn.Linear(n_qubits*2, 1))

    def forward(self, x):
        """x is a list with [theta, phi]"""
        theta = x[0]
        phi = x[1]

        # create a quantum device to run the gates
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, device = 'cpu')

        # prepare majorana circuit
        qdev = self.circuit_builder.generate_circuit(qdev, theta, phi)

        meas_qubits = range(n_qubits)
        x = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        # classification
        x = self.mlp_classical(x)
        x = torch.sigmoid(x)
        return x

class QCNN_Pure(nn.Module):
    """ This is a naive purely quantum 
        implementation, where we implement
        only the QCNN without the fully 
        connected MLP layer.
    """
    def __init__(self, circuit_builder, n_qubits = 8, n_cycles = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_cycles = n_cycles
        self.circuit_builder = circuit_builder(n_qubits, n_cycles)

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

        # third convolutional layer
        self.crx10 = tq.CRX(has_params=True, trainable=True)

        # third pooling layer
        self.u3_6 = tq.U3(has_params=True, trainable=True)

    def forward(self, x):
        """x is a list with [theta, phi]"""
        theta = x[0]
        phi = x[1]

        # create a quantum device to run the gates
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, device = 'cpu')

        # prepare majorana circuit
        qdev = self.circuit_builder.generate_circuit(qdev, theta, phi)

        # STEP 1: add trainable gates for QCNN circuit
        
        # first convolutional layer
        self.crx0(qdev, wires=[0, 1])
        self.crx1(qdev, wires=[2, 3])
        self.crx2(qdev, wires=[4, 5])
        self.crx3(qdev, wires=[6, 7])
        self.crx4(qdev, wires=[1, 2])
        self.crx5(qdev, wires=[3, 4])
        self.crx6(qdev, wires=[5, 6])

        # first pooling layer
        meas_qubits = [0,2,4,6]
        _  = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        self.u3_0(qdev, wires = 1)
        self.u3_1(qdev, wires = 3)
        self.u3_2(qdev, wires = 5)
        self.u3_3(qdev, wires = 7)

        # second convolutional layer
        self.crx7(qdev, wires=[1, 3])
        self.crx8(qdev, wires=[5, 7])
        self.crx9(qdev, wires=[3, 5])

        # second pooling layer
        meas_qubits = [1,5]
        _ = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))
        self.u3_4(qdev, wires = 3)
        self.u3_5(qdev, wires = 7)

        # third convolutional layer
        self.crx10(qdev, wires = [3,7])

        # third pooling layer
        meas_qubits = [3]
        _ = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))
        self.u3_6(qdev, wires = 7)

        # perform measurement to get expectations (back to classical domain)
        meas_qubits = [7]
        x = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        # classification
        return torch.relu(x)
    
#############################
## Base Model + Variations ##
##     (1 feature map)     ##
#############################

# QCNN Base Model
class QCNN_Base(nn.Module):
    """ This is the base implementation of our QCNN model. 
        It has log (N) convolution layers and log(N) 
        pooling layers (for an N qubit system), and after 
        applying the layers, passes the output to a fully 
        connected MLP network, which outputs a binary 
        classification value y $\in$ [0,1].
    """
    
    def __init__(self, circuit_builder, n_qubits = 8, n_cycles = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_cycles = n_cycles
        self.circuit_builder = circuit_builder(n_qubits, n_cycles)

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

        # multilevel perceptron layer
        self.mlp_class = nn.Sequential(nn.Linear(2, 10), nn.Tanh(), nn.Linear(10, 1))

    def forward(self, x):
        """x is a list with [theta, phi]"""
        theta = x[0]
        phi = x[1]

        # create a quantum device to run the gates
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, device = 'cpu')

        # prepare majorana circuit
        qdev = self.circuit_builder.generate_circuit(qdev, theta, phi)

        # STEP 1: add trainable gates for QCNN circuit
        
        # first convolutional layer
        self.crx0(qdev, wires=[0, 1])
        self.crx1(qdev, wires=[2, 3])
        self.crx2(qdev, wires=[4, 5])
        self.crx3(qdev, wires=[6, 7])
        self.crx4(qdev, wires=[1, 2])
        self.crx5(qdev, wires=[3, 4])
        self.crx6(qdev, wires=[5, 6])

        # first pooling layer
        meas_qubits = [0,2,4,6]
        _  = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        self.u3_0(qdev, wires = 1)
        self.u3_1(qdev, wires = 3)
        self.u3_2(qdev, wires = 5)
        self.u3_3(qdev, wires = 7)

        # second convolutional layer
        self.crx7(qdev, wires=[1, 3])
        self.crx8(qdev, wires=[5, 7])
        self.crx9(qdev, wires=[3, 5])

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


# Two-Qubit Unitary of RZ, CNOT, and RY for convolution
class QCNN_ZNOTY(nn.Module):
    """ This is a variation of the QCNN_BASE class, 
        where the only change is that we implement 
        a more general unitary in place of CRX in 
        the convolution layer.
    """
    
    # we added circuit_builder (which comes from state_prep.py file), this is the class that makes the input circuits that we want to extract information from (the Majorana circuits)
    def __init__(self, circuit_builder, n_qubits = 8, n_cycles = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_cycles = n_cycles
        self.circuit_builder = circuit_builder(n_qubits, n_cycles)

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

        # multilevel perceptron layer
        self.mlp_class = nn.Sequential(nn.Linear(2, 10), nn.Tanh(), nn.Linear(10, 1))

    def forward(self, x):
        """x is a list with [theta, phi]"""
        theta = x[0]
        phi = x[1]

        # create a quantum device to run the gates
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, device = 'cpu')

        # prepare majorana circuit
        qdev = self.circuit_builder.generate_circuit(qdev, theta, phi)

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
    
#############################
## Base Model + Variations ##
##     (n feature maps)    ##
#############################

# Three Feature Maps of CRX, CRY, CRZ for Convolutional Layers with Shared Weights
class QCNN_Shared(nn.Module):
    """ This is a variation of QCNN_BASE where
        we implement 3 feature maps for the first 
        convolutional layer, each with a different
        unitary (CRX, CRY, CRZ). We feed the
        measurement results into a fully connected
        MLP at the end. All of the feature maps
        have shared weights.
    """
    def __init__(self, circuit_builder, n_qubits = 8, n_cycles = 4, weights = torch.randn(10)):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_cycles = n_cycles
        self.weights = weights
        self.circuit_builder = circuit_builder(n_qubits, n_cycles)

        self.meas_basis = tq.PauliZ

        # first convolutional layer
        self.cr0 = SharedWeights(self.weights[0])
        self.cr1 = SharedWeights(self.weights[1])
        self.cr2 = SharedWeights(self.weights[2])
        self.cr3 = SharedWeights(self.weights[3])
        self.cr4 = SharedWeights(self.weights[4])
        self.cr5 = SharedWeights(self.weights[5])
        self.cr6 = SharedWeights(self.weights[6])

        # first pooling layer
        self.u3_0 = tq.U3(has_params=True, trainable=True)
        self.u3_1 = tq.U3(has_params=True, trainable=True)
        self.u3_2 = tq.U3(has_params=True, trainable=True)
        self.u3_3 = tq.U3(has_params=True, trainable=True)

        # second convolutional layer
        self.cr7 = SharedWeights(self.weights[7])
        self.cr8 = SharedWeights(self.weights[8])
        self.cr9 = SharedWeights(self.weights[9])

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
        self.mlp_class = nn.Sequential(nn.Linear(6, 15), nn.Tanh(), nn.Linear(15, 1))

    def forward(self, x):
        """x is a list with [theta, phi]"""
        theta = x[0]
        phi = x[1]

        # create a quantum device to run the gates
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, device = 'cpu')

        # prepare majorana circuit
        qdev = self.circuit_builder.generate_circuit(qdev, theta, phi)

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

# QCNN three feature maps CRX, CRY, CRZ with different weights
class QCNN_Diff(nn.Module):
    """ This is a variation of QCNN_BASE where
        we implement 3 feature maps for the first 
        convolutional layer, each with a different
        unitary (CRX, CRY, CRZ). We feed the
        measurement results into a fully connected
        MLP at the end. All of the feature maps
        have unique weights.
    """
    def __init__(self, circuit_builder, n_qubits = 8, n_cycles = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_cycles = n_cycles
        self.circuit_builder = circuit_builder(n_qubits, n_cycles)

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

        # Second Filter
        # first convolutional layer
        self.crx10 = tq.CRY(has_params=True, trainable=True)
        self.crx11 = tq.CRY(has_params=True, trainable=True)
        self.crx12 = tq.CRY(has_params=True, trainable=True)
        self.crx13 = tq.CRY(has_params=True, trainable=True)
        self.crx14 = tq.CRY(has_params=True, trainable=True)
        self.crx15 = tq.CRY(has_params=True, trainable=True)
        self.crx16 = tq.CRY(has_params=True, trainable=True)

        # first pooling layer
        self.u3_10 = tq.U3(has_params=True, trainable=True)
        self.u3_11 = tq.U3(has_params=True, trainable=True)
        self.u3_12 = tq.U3(has_params=True, trainable=True)
        self.u3_13 = tq.U3(has_params=True, trainable=True)

        # second convolutional layer
        self.crx17 = tq.CRY(has_params=True, trainable=True)
        self.crx18 = tq.CRY(has_params=True, trainable=True)
        self.crx19 = tq.CRY(has_params=True, trainable=True)

        # second pooling layer
        self.u3_14 = tq.U3(has_params=True, trainable=True)
        self.u3_15 = tq.U3(has_params=True, trainable=True)

        # Third Layer
        # first convolutional layer
        self.crx20 = tq.CRZ(has_params=True, trainable=True)
        self.crx21 = tq.CRZ(has_params=True, trainable=True)
        self.crx22 = tq.CRZ(has_params=True, trainable=True)
        self.crx23 = tq.CRZ(has_params=True, trainable=True)
        self.crx24 = tq.CRZ(has_params=True, trainable=True)
        self.crx25 = tq.CRZ(has_params=True, trainable=True)
        self.crx26 = tq.CRZ(has_params=True, trainable=True)

        # first pooling layer
        self.u3_20 = tq.U3(has_params=True, trainable=True)
        self.u3_21 = tq.U3(has_params=True, trainable=True)
        self.u3_22 = tq.U3(has_params=True, trainable=True)
        self.u3_23 = tq.U3(has_params=True, trainable=True)

        # second convolutional layer
        self.crx27 = tq.CRZ(has_params=True, trainable=True)
        self.crx28 = tq.CRZ(has_params=True, trainable=True)
        self.crx29 = tq.CRZ(has_params=True, trainable=True)

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

        # prepare majorana circuit
        qdev = self.circuit_builder.generate_circuit(qdev, theta, phi)

        # Create copies of circuit for parallel feature maps
        qdev1 = copy.deepcopy(qdev)
        qdev2 = copy.deepcopy(qdev)

        # STEP 1: add trainable gates for QCNN circuit
        
        # first convolutional layer
        self.crx0(qdev, wires=[0, 1])
        self.crx1(qdev, wires=[2, 3])
        self.crx2(qdev, wires=[4, 5])
        self.crx3(qdev, wires=[6, 7])
        self.crx4(qdev, wires=[1, 2])
        self.crx5(qdev, wires=[3, 4])
        self.crx6(qdev, wires=[5, 6])

        # first pooling layer
        meas_qubits = [0,2,4,6]
        _  = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        self.u3_0(qdev, wires = 1)
        self.u3_1(qdev, wires = 3)
        self.u3_2(qdev, wires = 5)
        self.u3_3(qdev, wires = 7)

        # second convolutional layer
        self.crx7(qdev, wires=[1, 3])
        self.crx8(qdev, wires=[5, 7])
        self.crx9(qdev, wires=[3, 5])

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
        self.crx10(qdev1, wires=[0, 1])
        self.crx11(qdev1, wires=[2, 3])
        self.crx12(qdev1, wires=[4, 5])
        self.crx13(qdev1, wires=[6, 7])
        self.crx14(qdev1, wires=[1, 2])
        self.crx15(qdev1, wires=[3, 4])
        self.crx16(qdev1, wires=[5, 6])

        # first pooling layer
        meas_qubits = [0,2,4,6]
        _  = tqm.expval(qdev1, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        self.u3_10(qdev1, wires = 1)
        self.u3_11(qdev1, wires = 3)
        self.u3_12(qdev1, wires = 5)
        self.u3_13(qdev1, wires = 7)

        # second convolutional layer
        self.crx17(qdev1, wires=[1, 3])
        self.crx18(qdev1, wires=[5, 7])
        self.crx19(qdev1, wires=[3, 5])

        # second pooling layer
        meas_qubits = [1,5]
        _ = tqm.expval(qdev1, meas_qubits, [self.meas_basis()] * len(meas_qubits))
        self.u3_14(qdev1, wires = 3)
        self.u3_15(qdev1, wires = 7)

        # perform measurement to get expectations (back to classical domain)
        meas_qubits = [3,7]
        y = tqm.expval(qdev1, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        # first convolutional layer
        self.crx20(qdev2, wires=[0, 1])
        self.crx21(qdev2, wires=[2, 3])
        self.crx22(qdev2, wires=[4, 5])
        self.crx23(qdev2, wires=[6, 7])
        self.crx24(qdev2, wires=[1, 2])
        self.crx25(qdev2, wires=[3, 4])
        self.crx26(qdev2, wires=[5, 6])

        # first pooling layer
        meas_qubits = [0,2,4,6]
        _  = tqm.expval(qdev2, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        self.u3_20(qdev2, wires = 1)
        self.u3_21(qdev2, wires = 3)
        self.u3_22(qdev2, wires = 5)
        self.u3_23(qdev2, wires = 7)

        # second convolutional layer
        self.crx27(qdev2, wires=[1, 3])
        self.crx28(qdev2, wires=[5, 7])
        self.crx29(qdev2, wires=[3, 5])

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

# QCNN with Feature Maps of CRX and the Unitary CRZ, NOT, Y gates with different weights
class QCNN_ZNOTY_Diff(nn.Module):
    """ This is a variation of QCNN_BASE where
        we implement 2 feature maps for the first 
        convolutional layer, each with a different
        unitary (general ZNOTY, CRX). We feed the
        measurement results into a fully connected
        MLP at the end. The feature maps have
        unique weights.
    """
    # we added circuit_builder (which comes from state_prep.py file), this is the class that makes the input circuits that we want to extract information from (the Majorana circuits)
    def __init__(self, circuit_builder, n_qubits = 8, n_cycles = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_cycles = n_cycles
        self.circuit_builder = circuit_builder(n_qubits, n_cycles)

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

        # Second Filter
        # first convolutional layer
        self.crx10 = tq.CRX(has_params=True, trainable=True)
        self.crx11 = tq.CRX(has_params=True, trainable=True)
        self.crx12 = tq.CRX(has_params=True, trainable=True)
        self.crx13 = tq.CRX(has_params=True, trainable=True)
        self.crx14 = tq.CRX(has_params=True, trainable=True)
        self.crx15 = tq.CRX(has_params=True, trainable=True)
        self.crx16 = tq.CRX(has_params=True, trainable=True)

        # first pooling layer
        self.u3_10 = tq.U3(has_params=True, trainable=True)
        self.u3_11 = tq.U3(has_params=True, trainable=True)
        self.u3_12 = tq.U3(has_params=True, trainable=True)
        self.u3_13 = tq.U3(has_params=True, trainable=True)

        # second convolutional layer
        self.crx17 = tq.CRX(has_params=True, trainable=True)
        self.crx18 = tq.CRX(has_params=True, trainable=True)
        self.crx19 = tq.CRX(has_params=True, trainable=True)

        # second pooling layer
        self.u3_14 = tq.U3(has_params=True, trainable=True)
        self.u3_15 = tq.U3(has_params=True, trainable=True)

        # multilevel perceptron layer
        self.mlp_class = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, x):
        """x is a list with [theta, phi]"""
        theta = x[0]
        phi = x[1]

        # create a quantum device to run the gates
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, device = 'cpu')

        # prepare majorana circuit
        qdev = self.circuit_builder.generate_circuit(qdev, theta, phi)

        qdev1 = copy.deepcopy(qdev)

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

        # Same Operations for the second feature map
        # first convolutional layer
        self.crx10(qdev1, wires=[0, 1])
        self.crx11(qdev1, wires=[2, 3])
        self.crx12(qdev1, wires=[4, 5])
        self.crx13(qdev1, wires=[6, 7])
        self.crx14(qdev1, wires=[1, 2])
        self.crx15(qdev1, wires=[3, 4])
        self.crx16(qdev1, wires=[5, 6])

        # first pooling layer
        meas_qubits = [0,2,4,6]
        _  = tqm.expval(qdev1, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        self.u3_10(qdev1, wires = 1)
        self.u3_11(qdev1, wires = 3)
        self.u3_12(qdev1, wires = 5)
        self.u3_13(qdev1, wires = 7)

        # second convolutional layer
        self.crx17(qdev1, wires=[1, 3])
        self.crx18(qdev1, wires=[5, 7])
        self.crx19(qdev1, wires=[3, 5])

        # second pooling layer
        meas_qubits = [1,5]
        _ = tqm.expval(qdev1, meas_qubits, [self.meas_basis()] * len(meas_qubits))
        self.u3_14(qdev1, wires = 3)
        self.u3_15(qdev1, wires = 7)

        # perform measurement to get expectations (back to classical domain)
        meas_qubits = [3,7]
        y = tqm.expval(qdev1, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        # classification
        result = self.mlp_class(torch.cat((x,y), 1))
        result = torch.sigmoid(result)
        return result
    

#######################
## UNDER DEVELOPMENT ##
#######################

class QRNN_Base(nn.Module):
    """ 

    """
    def __init__(self, circuit_builder, n_qubits=8, n_cycles=4):
        super().__init__()
        
        # setting some initial variables
        self.n_qubits, self.n_cycles = n_qubits, n_cycles
        self.circuit_builder = circuit_builder(n_qubits, n_cycles)
        self.meas_basis = tq.PauliZ # setting the measurement basis

        # TODO: finish the __init__ function


    def forward(self, x):
        theta, phi = x[0], x[1] # x is the following list [theta, phi]

        # creating a quantum device to run the gates on
        qdev = tq.QuantumDevice(self.n_qubits, device='cpu') # create quantum device
        qdev = self.circuit_builder.generate_circuit(qdev, theta, phi) # prepare majorana circuit

        # TODO : implement the forward function


        return x

class VQCNN_Base(nn.Module):
    def __init__(self, circuit_builder, n_qubits=8, n_cycles=4, trainable_params=10):
        super().__init__()
        
        # setting some initial variables
        self.n_qubits, self.n_cycles = n_qubits, n_cycles
        self.circuit_builder = circuit_builder(n_qubits, n_cycles)
        self.meas_basis = tq.PauliZ # setting the measurement basis

        # TODO: finish the __init__ function

        # creating trainable parameters for the variational QCNN
        self.trainable_params = trainable_params

        # setting up the quantum gates for the variational QCNN
        self.gates = nn.ModuleList([
            tq.CRX(has_params=True, trainable=True) for _ in range(trainable_params)
        ])

        # creating the classical MLP
        self.mlp_class = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )
    
    def forward(self, x):
        theta, phi = x[0], x[1] # x is the following list [theta, phi]

        # creating a quantum device to run the gates on
        qdev = tq.QuantumDevice(self.n_qubits, device='cpu') # create quantum device
        qdev = self.circuit_builder.generate_circuit(qdev, theta, phi) # prepare majorana circuit

        # TODO : implement the forward function

        # add trainable gates for the QCNN circuit
        for i in range(self.trainable_params):
            self.gates[i](qdev, wires=[i % 8, (i + 1) % 8])

        # perform measurement to get expectations (back to classical domain)
        meas_qubits = [3, 7]
        x = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        # classification
        x = self.mlp_class(x)
        x = torch.sigmoid(x)

        return x

class VQCNN_Parameterized(nn.Module):
    def __init__(self, circuit_builder, n_qubits=8, n_cycles=4, trainable_params=10):
        super().__init__()
        
        # setting some initial variables
        self.n_qubits, self.n_cycles = n_qubits, n_cycles
        self.circuit_builder = circuit_builder(n_qubits, n_cycles)
        self.meas_basis = tq.PauliZ # setting the measurement basis

        # TODO: finish the __init__ function

        # creating trainable parameters for the variational QCNN
        self.trainable_params = nn.ParameterList([
            nn.Parameter(torch.randn(1, 4), requires_grad=True) for _ in range(trainable_params)
        ])

        # setting up the quantum gates for the variational QCNN
        self.gates = nn.ModuleList([
            tq.CRX(has_params=True, trainable=True) for _ in range(trainable_params)
        ])

        # creating the classical MLP
        self.mlp_class = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )
    
    def forward(self, x):
        theta, phi = x[0], x[1] # x is the following list [theta, phi]

        # creating a quantum device to run the gates on
        qdev = tq.QuantumDevice(self.n_qubits, device='cpu') # create quantum device
        qdev = self.circuit_builder.generate_circuit(qdev, theta, phi) # prepare majorana circuit

        # TODO : implement the forward function

        # add trainable gates for the QCNN circuit
        for i in range(len(self.trainable_params)):
            self.gates[i](qdev, wires=[i % 8, (i + 1) % 8], params=self.trainable_params[i])

        # perform measurement to get expectations (back to classical domain)
        meas_qubits = [3, 7]
        x = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        # classification
        x = self.mlp_class(x)
        x = torch.sigmoid(x)

        return x

class VQCNN_Rotation_RY_Single(nn.Module):
    def __init__(self, circuit_builder, n_qubits=8, n_cycles=4, trainable_params=10):
        super().__init__()
        
        # setting some initial variables
        self.n_qubits, self.n_cycles = n_qubits, n_cycles
        self.circuit_builder = circuit_builder(n_qubits, n_cycles)
        self.meas_basis = tq.PauliZ # setting the measurement basis

        # TODO: finish the __init__ function

        # creating trainable parameters for the variational QCNN
        self.trainable_params = nn.ParameterList([
            nn.Parameter(torch.randn(1, 4), requires_grad=True) for _ in range(trainable_params)
        ])

        # setting up the quantum gates for the variational QCNN
        self.gates = nn.ModuleList([
            tq.CRX(has_params=True, trainable=True) for _ in range(trainable_params)
        ])

        # creating the classical MLP
        self.mlp_class = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )
    
    def forward(self, x):
        theta, phi = x[0], x[1] # x is the following list [theta, phi]

        # creating a quantum device to run the gates on
        qdev = tq.QuantumDevice(self.n_qubits, device='cpu') # create quantum device
        qdev = self.circuit_builder.generate_circuit(qdev, theta, phi) # prepare majorana circuit

        # TODO : implement the forward function

        # add trainable gates for the QCNN circuit
        for i in range(len(self.trainable_params)):
            self.gates[i](qdev, wires=[i % 8, (i + 1) % 8], params=self.trainable_params[i])

        # perform measurement to get expectations (back to classical domain)
        meas_qubits = [3, 7]
        x = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        # classification
        x = self.mlp_class(x)
        x = torch.sigmoid(x)

        return x

class VQCNN_Rotation_RY_On_Top(nn.Module):
    def __init__(self, circuit_builder, n_qubits=8, n_cycles=4, trainable_params=10):
        super().__init__()
        
        # setting some initial variables
        self.n_qubits, self.n_cycles = n_qubits, n_cycles
        self.circuit_builder = circuit_builder(n_qubits, n_cycles)
        self.meas_basis = tq.PauliZ # setting the measurement basis

        # TODO: finish the __init__ function

        # creating trainable parameters for the variational QCNN
        self.trainable_params = nn.ParameterList([
            nn.Parameter(torch.randn(1, 4), requires_grad=True) for _ in range(trainable_params)
        ])

        # setting up the quantum gates for the variational QCNN
        self.gates = nn.ModuleList([
            tq.CRX(has_params=True, trainable=True) for _ in range(trainable_params)
        ])

        # creating the classical MLP
        self.mlp_class = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )
    
    def forward(self, x):
        theta, phi = x[0], x[1] # x is the following list [theta, phi]

        # creating a quantum device to run the gates on
        qdev = tq.QuantumDevice(self.n_qubits, device='cpu') # create quantum device
        qdev = self.circuit_builder.generate_circuit(qdev, theta, phi) # prepare majorana circuit

        # TODO : implement the forward function

        # add trainable gates for the QCNN circuit
        for i in range(len(self.trainable_params)):
            self.gates[i](qdev, wires=[i % 8, (i + 1) % 8], params=self.trainable_params[i])

        # perform measurement to get expectations (back to classical domain)
        meas_qubits = [3, 7]
        x = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        # classification
        x = self.mlp_class(x)
        x = torch.sigmoid(x)

        return x

class VQCNN_Controlled(nn.Module):
    def __init__(self, circuit_builder, n_qubits=8, n_cycles=4, trainable_params=10):
        super().__init__()
        
        # setting some initial variables
        self.n_qubits, self.n_cycles = n_qubits, n_cycles
        self.circuit_builder = circuit_builder(n_qubits, n_cycles)
        self.meas_basis = tq.PauliZ # setting the measurement basis

        # TODO: finish the __init__ function

        # creating trainable parameters for the variational QCNN
        self.trainable_params = nn.ParameterList([
            nn.Parameter(torch.randn(1, 4), requires_grad=True) for _ in range(trainable_params)
        ])

        # setting up the quantum gates for the variational QCNN
        self.gates = nn.ModuleList([
            tq.CRX(has_params=True, trainable=True) for _ in range(trainable_params)
        ])

        # creating the classical MLP
        self.mlp_class = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )
    
    def forward(self, x):
        theta, phi = x[0], x[1] # x is the following list [theta, phi]

        # creating a quantum device to run the gates on
        qdev = tq.QuantumDevice(self.n_qubits, device='cpu') # create quantum device
        qdev = self.circuit_builder.generate_circuit(qdev, theta, phi) # prepare majorana circuit

        # TODO : implement the forward function

        # add trainable gates for the QCNN circuit
        for i in range(len(self.trainable_params)):
            self.gates[i](qdev, wires=[i % 8, (i + 1) % 8], params=self.trainable_params[i])

        # perform measurement to get expectations (back to classical domain)
        meas_qubits = [3, 7]
        x = tqm.expval(qdev, meas_qubits, [self.meas_basis()] * len(meas_qubits))

        # classification
        x = self.mlp_class(x)
        x = torch.sigmoid(x)

        return x

class QGCNN_Base(nn.Module):
    def __init__(self, circuit_builder, n_qubits=8, n_cycles=4, trainable_params=10):
        super().__init__()
        
        # setting some initial variables
        self.n_qubits, self.n_cycles = n_qubits, n_cycles
        self.circuit_builder = circuit_builder(n_qubits, n_cycles)
        self.meas_basis = tq.PauliZ # setting the measurement basis

        # TODO: finish the __init__ function

    
    def forward(self, x):
        theta, phi = x[0], x[1] # x is the following list [theta, phi]

        # creating a quantum device to run the gates on
        qdev = tq.QuantumDevice(self.n_qubits, device='cpu') # create quantum device
        qdev = self.circuit_builder.generate_circuit(qdev, theta, phi) # prepare majorana circuit

        # TODO : implement the forward function

        return x
