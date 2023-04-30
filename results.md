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
