import numpy as np
from graphviz import Digraph

class Value:

    def __init__(self, value, children=(), ops=None, label=''):
        self.value = value
        self.grad = 0
        self._children = children
        self._ops = ops
        self._label = label
        self._backPropagationFunc = lambda: None
        
    def __repr__(self) -> str:
        return f"Value(data={self.value})"

    def __add__(self, other):
        out = Value(
            self.value + other.value, 
            children=(self, other), 
            ops='+', 
            label="(%s + %s)" % (self._label, other._label)
        )

        def bpfunc(): 
            print(f"Backpropagating {out._label}")
            self.grad += out.grad
            other.grad += out.grad          
            
            for child in out._children:
                child._backPropagationFunc()
                
        out._backPropagationFunc = bpfunc
        return out
        
    def __mul__(self, other):
        out =  Value(
            self.value * other.value, 
            children=(self, other), 
            ops='*', 
            label="%s * %s" % (self._label, other._label)
        )
        
        def bpfunc():
            print(f"Backpropagating {out._label}")
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
            
            for child in out._children:
                child._backPropagationFunc()
                
        out._backPropagationFunc = bpfunc    
        return out
        
    def tanh(self):
        out = Value(
            np.tanh(self.value), 
            children=(self,), 
            ops='tanh', 
            label="tanh(%s)" % self._label
        )
        
        def bpfunc():
            print(f"Backpropagating {out._label}")
            self.grad += (1 - out.value ** 2) * out.grad
            for child in out._children:
                child._backPropagationFunc()
        
        out._backPropagationFunc = bpfunc    
        return out
    
    # Draw the graph using graphviz that represents the computation
    def draw_graph(self):
        dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
        dot.node(name='A', label="%s | Data: %.12f | Grad: %.12f"  % (self._label, self.value, self.grad), shape="record")
        self._draw_graph(dot, 'A')
        return dot
    
    # Internal recursive call to draw the graph
    def _draw_graph(self, dot, parent):
        if self._children:
            dot.node(name=f'{parent}-op', label=self._ops)
            dot.edge(f'{parent}-op', parent)
            for i, child in enumerate(self._children):
                dot.node(name=f'{parent}-{i}', label="%s | Data: %.12f | Grad: %.12f" % (child._label, child.value, child.grad), shape="record")
                dot.edge(f'{parent}-{i}', f'{parent}-op')    
        
                child._draw_graph(dot, f'{parent}-{i}')
                
    def backward(self):
        self.grad = 1
        self._backPropagationFunc()
        return self