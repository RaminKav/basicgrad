from trace_viz import draw_dot

class Value:
    """Stores a single Scalar value and its gradient"""

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0

        # internal variables for autograd graph construction
        self._backward = lambda: None    # each operation supplies its backprop function
        self._prev = set(_children)
        self._op = _op  # operation that produced this node, for debug graph visualisations

    def backward(self):
        # topological ordering of all children in graph
        topo = []
        visited = set()

        def build_topo(parent: Value):
            if parent not in visited:
                visited.add(parent)
                for child in parent._prev:
                    build_topo(child)
                topo.append(parent)

        build_topo(self)

        # apply chain rule to each node in topological graph
        self.grad = 1
        v: Value
        for v in reversed(topo):
            v._backward()

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward

        return out

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only supports int and float powers"
        out = Value(self.data ** power, (self,), f'**{power}')

        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def draw_graph(self, format='png', name='gout'):
        dot = draw_dot(self, 'png')
        dot.render(name)


# look back into the pwr backward fn logic