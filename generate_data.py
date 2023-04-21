from source.ExperimentCode import MajoranaCircuit

import tensorflow as tf
import tensorflow_quantum as tfq
from qusetta import Qiskit

import cirq
import sympy
import numpy as np

# visualization tools
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

class QCNN:
    def __init__(self,n_qubits=8, n_cycles_total=10):
        self.n_qubits = n_qubits
        self.n_cycles_total = n_cycles_total

    def generate_data(self, n_points = 200, train_split = 0.7, use_qiskit = 0):
        """ 
        Generate training and testing data.
            Args: 
            n_points (int): the number of circuits for both training/testing.
                Default is 200
            train_split (float): the proportion of circuits that will be used 
                for testing. Default is 0.7
        """
        circuits = []
        labels = []
        for _ in range(n_points): # produce
            # generates random values of the gate angles theta and phi
            rng_theta = np.random.uniform(0,np.pi/2)
            rng_phi = np.random.uniform(0,np.pi/2)

            # calls Nikhil's code to make a circuit from gate parameters
            constructor = MajoranaCircuit(i_theta = rng_theta, i_phi = rng_phi, n_qubits = self.n_qubits, n_cycles_total=self.n_cycles_total)
            qiskit_cir = constructor.build_circuit()

            # converts Qiskit circuit to Circ and appends to our list of circuits with a label
            if use_qiskit:
                circuits.append(qiskit_cir)
            else:
                cirq_cir = Qiskit.to_cirq(qiskit_cir)
                circuits.append(cirq_cir)
            
            labels.append(QCNN.topological_classifier(rng_theta, rng_phi))

        # partitions circuit list into test and train sets
        split_ind = int(len(circuits) * train_split)
        train_circuits = circuits[:split_ind]
        test_circuits = circuits[split_ind:]

        train_labels = labels[:split_ind]
        test_labels = labels[split_ind:]

        return tfq.convert_to_tensor(train_circuits), np.array(train_labels), \
            tfq.convert_to_tensor(test_circuits), np.array(test_labels)
    
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
            elif np.pi/4 < phi and theta <= (phi-np.pi/4):
                label = 0
            else:
                label = 1
        else:
            label = 1
        return label

## The following code is copied/adapted from https://www.tensorflow.org/quantum/tutorials/qcnn
    def one_qubit_unitary(self, bit, symbols):
        """Make a Cirq circuit enacting a rotation of the bloch sphere about the X,
        Y and Z axis, that depends on the values in `symbols`.
        """
        return cirq.Circuit(
            cirq.X(bit)**symbols[0],
            cirq.Y(bit)**symbols[1],
            cirq.Z(bit)**symbols[2])

    def two_qubit_unitary(self, bits, symbols):
        """Make a Cirq circuit that creates an arbitrary two qubit unitary."""
        circuit = cirq.Circuit()
        circuit += self.one_qubit_unitary(bits[0], symbols[0:3])
        circuit += self.one_qubit_unitary(bits[1], symbols[3:6])
        circuit += [cirq.ZZ(*bits)**symbols[6]]
        circuit += [cirq.YY(*bits)**symbols[7]]
        circuit += [cirq.XX(*bits)**symbols[8]]
        circuit += self.one_qubit_unitary(bits[0], symbols[9:12])
        circuit += self.one_qubit_unitary(bits[1], symbols[12:])
        return circuit

    def two_qubit_pool(self,source_qubit, sink_qubit, symbols):
        """Make a Cirq circuit to do a parameterized 'pooling' operation, which
        attempts to reduce entanglement down from two qubits to just one."""
        pool_circuit = cirq.Circuit()
        sink_basis_selector = self.one_qubit_unitary(sink_qubit, symbols[0:3])
        source_basis_selector = self.one_qubit_unitary(source_qubit, symbols[3:6])
        pool_circuit.append(sink_basis_selector)
        pool_circuit.append(source_basis_selector)
        pool_circuit.append(cirq.CNOT(control=source_qubit, target=sink_qubit))
        pool_circuit.append(sink_basis_selector**-1)
        return pool_circuit
    
    def quantum_conv_circuit(self, bits, symbols):
        """Quantum Convolution Layer following the above diagram.
        Return a Cirq circuit with the cascade of `two_qubit_unitary` applied
        to all pairs of qubits in `bits` as in the diagram above.
        """
        circuit = cirq.Circuit()
        for first, second in zip(bits[0::2], bits[1::2]):
            circuit += self.two_qubit_unitary([first, second], symbols)
        for first, second in zip(bits[1::2], bits[2::2] + [bits[0]]):
            circuit += self.two_qubit_unitary([first, second], symbols)
        return circuit
    
    def quantum_pool_circuit(self, source_bits, sink_bits, symbols):
        """A layer that specifies a quantum pooling operation.
        A Quantum pool tries to learn to pool the relevant information from two
        qubits onto 1.
        """
        circuit = cirq.Circuit()
        for source, sink in zip(source_bits, sink_bits):
            circuit += self.two_qubit_pool(source, sink, symbols)
        return circuit
    

    def create_model_circuit(self, qubits):
        """Create sequence of alternating convolution and pooling operators 
        which gradually shrink over time."""
        # TODO: generalize this to n qubits, not just 8.
        model_circuit = cirq.Circuit()
        symbols = sympy.symbols('qconv0:63')
        # Cirq uses sympy.Symbols to map learnable variables. TensorFlow Quantum
        # scans incoming circuits and replaces these with TensorFlow variables.
        model_circuit += self.quantum_conv_circuit(qubits, symbols[0:15])
        model_circuit += self.quantum_pool_circuit(qubits[:4], qubits[4:],
                                            symbols[15:21])
        model_circuit += self.quantum_conv_circuit(qubits[4:], symbols[21:36])
        model_circuit += self.quantum_pool_circuit(qubits[4:6], qubits[6:],
                                            symbols[36:42])
        model_circuit += self.quantum_conv_circuit(qubits[6:], symbols[42:57])
        model_circuit += self.quantum_pool_circuit([qubits[6]], [qubits[7]],
                                            symbols[57:63])
        return model_circuit
    
    def init_model(self):
        # Create our qubits and readout operators in Cirq.
        cluster_state_bits = cirq.GridQubit.rect(1, self.n_qubits)
        readout_operators = cirq.Z(cluster_state_bits[-1])

        # Build a sequential model enacting the logic in 1.3 of this notebook.
        # Here you are making the static cluster state prep as a part of the AddCircuit and the
        # "quantum datapoints" are coming in the form of excitation
        circ_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        cluster_state = tfq.layers.AddCircuit()(
            circ_input) #, prepend=cluster_state_circuit(cluster_state_bits))

        quantum_model = tfq.layers.PQC(self.create_model_circuit(cluster_state_bits),
                                    readout_operators)(cluster_state)

        qcnn_model = tf.keras.Model(inputs=[circ_input], outputs=[quantum_model])

        # Show the keras plot of the model
        # tf.keras.utils.plot_model(qcnn_model,
        #                         show_shapes=True,
        #                         show_layer_names=False,
        #                         dpi=70)
        return qcnn_model
        
