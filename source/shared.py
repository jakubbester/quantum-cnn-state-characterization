import torch
import torch.nn as nn
import torchquantum as tq

class SharedWeights(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, input, gate_type, wire1, wire2):
        if gate_type == "CRX":
            gate = tq.CRX(self.weight, trainable = True)
        elif gate_type == "CRY":
            gate = tq.CRY(self.weight, trainable=True)
        elif gate_type == "CRZ":
            gate = tq.CRZ(self.weight, trainable=True)
        else:
            raise ValueError("Invalid gate type")

        output = gate(input, wires = [wire1, wire2])
        return output