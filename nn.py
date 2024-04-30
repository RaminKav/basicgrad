import random
from typing import Callable

from engine import Value

class Module:
    def zero_grad(self):
        """zero out all parameter gradient values"""
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]  # randomize weights at init
        self.b = Value(random.uniform(-1,1))
        self.nonlin = nonlin

    def __call__(self, x):
        # weights * x + bias, x is the inputs
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'} Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def train(self, data: [[float]], targets: [float], loss_fn: Callable[[[float],[float]], Value], n=200, learn=0.01, debug=False):
        for i in range(n):
            preds = [self(x) for x in data]
            loss = loss_fn(targets, preds)
            self.zero_grad()
            loss.backward()

            for p in self.parameters():
                p.data += -learn * p.grad
        if debug:
            print(f"Pass {i}: Loss: {loss} Preds: {[p.data for p in preds]}")

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


n = MLP(3, [4, 4, 1])
data = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0]
]
targets = [1.0, -1.0, -1.0, 1.0]
n.train(data, targets, lambda t, p: sum((yout - ygt)**2 for ygt, yout in zip(t, p)))





