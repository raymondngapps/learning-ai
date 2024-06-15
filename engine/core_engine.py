import numpy as np
from graphviz import Digraph

class Value:

    def __init__(self, value, children=(), ops=None, label=''):
        self.value = value
        self.grad = 0
        self._children = children
        self._ops = ops
        self._label = label
        
    def __repr__(self) -> str:
        return f"Value(data={self.value})"

    def __add__(self, other):
        return Value(self.value + other.value, children=(self, other), ops='+', label="(%s + %s)" % (self._label, other._label))
    
    def __sub__(self, other):
        return Value(self.value - other.value, children=(self, other), ops='-',  label="(%s - %s)" % (self._label, other._label))
    
    def __mul__(self, other):
        return Value(self.value * other.value, children=(self, other), ops='*', label="%s * %s" % (self._label, other._label))
    
    def __truediv__(self, other):
        return Value(self.value / other.value, children=(self, other), ops='/', label="%s / %s" % (self._label, other._label))
    
    def tanh(self):
        return Value(np.tanh(self.value), children=(self,), ops='tanh', label="tanh(%s)" % self._label)
    
    # Draw the graph using graphviz that represents the computation
    def draw_graph(self):
        dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
        dot.node(name='A', label="%s | Data: %.4f | Grad: %.4f"  % (self._label, self.value, self.grad), shape="record")
        self._draw_graph(dot, 'A')
        return dot
    
    # Internal recursive call to draw the graph
    def _draw_graph(self, dot, parent):
        if self._children:
            dot.node(name=f'{parent}-op', label=self._ops)
            dot.edge(f'{parent}-op', parent)
            for i, child in enumerate(self._children):
                dot.node(name=f'{parent}-{i}', label="%s | Data: %.4f | Grad: %.4f" % (child._label, child.value, child.grad), shape="record")
                dot.edge(f'{parent}-{i}', f'{parent}-op')    
        
                child._draw_graph(dot, f'{parent}-{i}')