from graphviz import Digraph

class Value:

    def __init__(self, value, children=(), ops=None):
        self.value = value
        self._children = children
        self._ops = ops
        
    def __repr__(self) -> str:
        return f"Value(data={self.value})"

    def __add__(self, other):
        return Value(self.value + other.value, children=(self, other), ops='+')
    
    def __sub__(self, other):
        return Value(self.value - other.value, children=(self, other), ops='-')
    
    def __mul__(self, other):
        return Value(self.value * other.value, children=(self, other), ops='*')
    
    def __truediv__(self, other):
        return Value(self.value / other.value, children=(self, other), ops='/')
    
    def draw_graph(self):
        dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
        dot.node(name='A', label="Data: %.4f" % self.value, shape="record")
        self._draw_graph(dot, 'A')
        return dot
    
    def _draw_graph(self, dot, parent):
        if self._children:
            dot.node(name=f'{parent}-op', label=self._ops)
            dot.edge(f'{parent}-op', parent)
            for i, child in enumerate(self._children):
                dot.node(name=f'{parent}-{i}', label="Data: %.4f" % child.value, shape="record")
                dot.edge(f'{parent}-{i}', f'{parent}-op')    
        
                child._draw_graph(dot, f'{parent}-{i}')