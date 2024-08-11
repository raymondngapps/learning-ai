import uuid
import numpy as np
from graphviz import Digraph

class Value:

    def __init__(self, value, children=(), ops=None, label=''):
        self.value = value
        self.grad = 0
        self._id = f'{uuid.uuid1()}'
        self._children = children
        self._ops = ops
        self._label = label
        self._backPropagationFunc = lambda: None
        
    def __repr__(self) -> str:
        return f"Value(data={self.value})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(value = other, label=str(other))
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
        other = other if isinstance(other, Value) else Value(value = other, label=str(other))
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
    
    def __sub__(self, other):
        return self + (other * (-1))
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * (other ** -1)

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(value = other, label=str(other))
        out = Value(
            self.value ** other.value, 
            children=(self, other), 
            ops='^', 
            label="%s ^ %s" % (self._label, other._label)
        )
        
        def bpfunc():
            print(f"Backpropagating {out._label}")
            self.grad += other.value * self.value ** (other.value - 1) * out.grad
            
            for child in out._children:
                child._backPropagationFunc()
                
        out._backPropagationFunc = bpfunc    
        return out
    
    def exp(self):
        out = Value(
            np.exp(self.value), 
            children=(self,), 
            ops='exp', 
            label="exp(%s)" % self._label
        )
        
        def bpfunc():
            print(f"Backpropagating {out._label}")
            self.grad += out.value * out.grad
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
        dot.node(name=self._id, label="%s | Data: %.12f | Grad: %.12f"  % (self._label, self.value, self.grad), shape="record")
        dots = set()
        self._draw_graph(dots, dot)
        return dot
    
    # Internal recursive call to draw the graph
    def _draw_graph(self, dots: set, dot: Digraph):
        if self._children:
            ops = f'{self._id}-op'
            # Avoid dup drawing
            if ops not in dots: 
                dot.node(name=f'{self._id}-op', label=self._ops)
                dot.edge(ops, self._id)
                dots.add(ops)            
            
            for i, child in enumerate(self._children):
                n = f'{child._id}'
                # Avoid dup drawing
                if n not in dots:
                    dot.node(name=n, label="%s | Data: %.12f | Grad: %.12f" % (child._label, child.value, child.grad), shape="record")
                    dot.edge(n, f'{self._id}-op')    
                    dots.add(n)
                    
                child._draw_graph(dots, dot)
                
    def backward(self):
        self.grad = 1
        self._backPropagationFunc()
        return self