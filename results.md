 - Benchmarking for different project directions can go here.
1. **Classical Neural Network**: 
   - *Structure*:
    - 8 classical measurements, mlp from 8 -> 16 -> 1, 
   - *Training and testing summary*:
    - training on 400, testing on 100; loss of 0.26, fluctuating often between 0.2 and 1.6, very low accuracy of 0.05, unable to detect Majorana states;
    - CHANGED OPTIMIZER LEARNING RATE to 5e-2, loss of 0.08, still fluctuating often between 0.02 and 1.86, relatively good accuracy of 0.59 now
2. **Pure Quantum Convolutional Neural Network**: 
   - *Structure*:
    - three layers of convolution and pooling, measurement on 1 qubit
   - *Training and testing summary*:
    - training of 400, testing on 100; learning rate of 5e-2; loss of 0.10, fluctuating often between 0.0 and 0.6, maybe more, relatively good accuracy of 0.64
3. **Quantum Hybrid Convolutional Neural Network v1**: 
   - *Structure*:
    - with three feature maps of same unitaries for convolutions with different weights that is inputted into a fully connected layer: 
   - *Training and testing summary*:
    - training on 400, testing on 100; loss of 5.8e-05, very stable with fluctuations all between 0 and 1e-4, very high accuracy of 0.97; optimizer learning rate of 9e-3  **NKH: why do we do three different feature maps if they're implementing the same unitaries? Even if they have different weights, not sure if this is useful... we should compare to baseline case with only one feature map.**
4. **Quantum Hybrid Convolutional Neural Network v2**
   - *Structure*:
    - with three feature maps of different unitaries (CRX, CRY, CRZ) for convolutions with different weights that is inputted into a fully connected layer
   - *Training and testing summary*: 
    - training on 400, testing on 100; loss of 0.0004, very stable with fluctuations all between 0 and 0.01, very high accuracy of 0.96; optimizer learning rate of 9e-3
5. **Quantum Hybrid Convolutional Neural Network v3**
   - *Structure*:
    - with three feature maps of different unitaries (CRX, CRY, CRZ) for convolutions with shared weights that is inputted into a fully connected layer, pooling layer does not use shared weight, changed mlp to 6 -> 15 -> 1, ran with different parameters to get optimal accuracy
   - *Training and testing summary*: 
    - training on 400, testing on 100; loss of 0.013, not as stable with fluctuations between 0 and 0.01, high accuracy of 0.79 to 0.84; optimizer learning rate of 2e-3 **Frank: I think we can do improvements for this, open to ideas for updates**

(Nikhil) results:
400 datapoints (70% train, 30% test) 10 epochs

trainable two qubit CZ gate, acc: 0.833
trainable two qubit CX gate, acc: 0.866
trainable two qubit CY gate, acc: 0.9
trainable two qubit CNOT gate, acc: 0.75
trainable two qubit SWAP gate, acc: 0.80
Qiskit 2- qubit unitary gate (convolution layer) , acc: 0.96

**Important Summary Information!**

All QCNN Models (that we built)
- QCNN_Base - this is the base model/QCNN structure
- QCNN_ZNOTY - two-qubit unitary of RZ, CNOT, and RY for convolution
- QCNN_Classical - classical neural network from measuring each qubit without quantum layers
- QCNN_Pure - purely quantum neural network with no fully connected classical neural network layer
- QCNN_Shared - three feature maps of CRX, CRY, CRZ for convolutional layers with shared weights
- QCNN_Diff - three feature maps CRX, CRY, CRZ with different weights
- QCNN_ZNOTY_Diff - feature maps of CRX and the unitary CRZ, NOT, Y with different weights

Added on the following modules and trained them! (Jakub Bester)

- Quantum Residual Neural Network (QRNN)
    - QRNN_Base - TODO (working on getting this to work from within the quantum and classical part of the model)
- Variational Quantum Convolutional Neural Network (VCNN)
    - VQCNN_Base - regular quantum convolutional (not really) neural network meant as a comparison (acc: 0.69 : 500 points, 0.8 train split, 10 epochs) (acc: 0.885 : 1000 points, 0.8 train split, 20 epochs)
    - VQCNN_Parameterized - Variational Quantum Convolutional (not really) Neural Network (VQCNN) (acc: 0.64 : 500 points, 0.8 train split, 10 epochs) (acc: 0.665: 1000 points, 0.8 train split, 20 epochs)
    - VQCNN_Rotation_RY_Single - uses solely rotation gates
    - VQCNN_Rotation_RY_On_Top - uses rotation gates in order to pseudo-parameterize the quantum circuit
    - VQCNN_Controlled - this uses controlled gates
There is some notable difference to adding parameterization! We would have to modify the model to not use the trivial
qubit selection that it is using right now, but this is a good start. There is also a lot of variation in the accuracies I noticed.
We can also try combining the different ones to see what happens, to generate a very comprehensive suite of things to test.
- Quantum Graph Convolutional Neural Network (QGCNN)
    - QGCNN_Base - TODO (not sure if this is a good idea)

Thoughts on paper : simply create a table of all possible combinations and run tests on them, and have different team members explain them accordingly
