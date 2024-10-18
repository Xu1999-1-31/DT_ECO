import dgl
import torch as th
from gymnasium import spaces

class GraphSpace(spaces.Space):
    """A custom space to represent a DGL graph."""
    
    def __init__(self, graph):
        """Initialize the space with a DGL graph and define the shape based on node and edge counts."""
        assert isinstance(graph, dgl.DGLGraph), "Input must be a DGLGraph"
        
        # Define the shape as (number_of_nodes, number_of_edges)
        # node numbers, node features, edge numbers, cellarc features, netarc features
        shape = (graph.number_of_nodes(), graph.number_of_edges())
        self.graph = graph
        
        # Call the parent constructor and pass the shape
        super().__init__(shape=shape, dtype=None)
    
    def sample(self):
        """Return a sample from the space. For now, return the graph itself."""
        # Normally, you'd sample a graph randomly, but we return the provided graph here
        return self.graph
    
    def contains(self, x):
        """Check if x is a valid graph that can be used in this space."""
        return isinstance(x, dgl.DGLGraph)
    
    def __repr__(self):
        """Return a string representation of the space with shape details."""
        return (f"GraphSpace(graph with {self.graph.number_of_nodes()} nodes "
                f"and {self.graph.number_of_edges()} edges, shape={self.shape})")