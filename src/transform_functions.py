from   config import *
import numpy as np
import scipy
import torch
from   torch.fft import ihfftn as inverse_fourier_transform
from   torch_geometric.data import Data 
from   torch_geometric.transforms import BaseTransform 
from   torch_geometric.utils import subgraph, k_hop_subgraph, to_scipy_sparse_matrix


def add_indices(dataset):
    data_list = []
    for i in range(len(dataset)):
        data = dataset.get(i)  # Use `get` to avoid reapplying transforms
        data.idx = torch.tensor([i], dtype=torch.long)
        data_list.append(data)
    dataset.data, dataset.slices = dataset.collate(data_list)
    return dataset

class DensifyTransform(BaseTransform):
    def __init__(self, method, seed):
        self.method = method
        self.seed = seed
        assert self.method is not None

    def __call__(self, data):
        torch.manual_seed(seed)

        def densify(t, method='fourier'):
            if method=='fourier':
                t = torch.real(inverse_fourier_transform(t))
            elif method=='random_noise':
                t += (torch.rand_like(t.float())-0.5)*1e-5
            elif method=='subtract_small_constant':
                t -=1e-5
            elif method=='add_small_constant':
                t +=1e-5
            elif method=='exp':
                t = t.exp()
            return t
        data.x = densify(data.x, method=self.method)
        return data

    def __repr__(self):
        return 'DensifyTransform()'

class ChooseLargestMaskForTrain(BaseTransform):
    def __call__(self, data):
        try:
            train_mask = data.train_mask.clone()
        except:
            train_mask=[]
        try:
            test_mask = data.test_mask.clone()
        except:
            test_mask=[]
        try:
            val_mask = data.val_mask.clone()
        except:
            val_mask=[]
        if torch.sum(test_mask)>torch.sum(train_mask) and torch.sum(test_mask)>torch.sum(val_mask):
            data.train_mask=test_mask
            data.test_mask=train_mask
        elif torch.sum(val_mask)>torch.sum(train_mask) and torch.sum(val_mask)>torch.sum(test_mask):
            data.train_mask=val_mask
            data.val_mask=train_mask
        return data

    def __repr__(self):
        return 'ChooseLargestMaskForTrain()'

class CreateMaskTransform:
    def __init__(self, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=0):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed=seed
        # print('train ratio:',self.train_ratio)
        # print('test ratio:',self.test_ratio)
        # print('val ratio:',self.val_ratio)
        # print('num nodes:',num_nodes)

    def __call__(self, data):
        num_nodes = data.num_nodes
        
        # Generate random indices
        torch.manual_seed(self.seed)
        indices = torch.randperm(num_nodes)

        train_size = int(self.train_ratio * num_nodes)
        val_size = int(self.val_ratio * num_nodes)
        test_size = num_nodes - train_size - val_size

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]

        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        # Assign masks to the data object
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        
        return data
    
    def __repr__(self):
        return 'CreateMaskTransform()'

class GraphFourierTransform(BaseTransform):
    def __init__(self):
        super().__init__()

    def compute_laplacian(self, edge_index, num_nodes):
        adj_matrix = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
        laplacian = scipy.sparse.csgraph.laplacian(adj_matrix, normed=True)
        return laplacian

    def compute_eigendecomposition(self, laplacian):
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
        return eigenvalues, eigenvectors

    def graph_fourier_transform(self, features, eigenvectors):
        # Apply the Graph Fourier Transform
        gft = eigenvectors.T @ features
        return gft

    def forward(self, data: Data) -> Data:
        laplacian = self.compute_laplacian(data.edge_index, data.num_nodes)
        eigenvalues, eigenvectors = self.compute_eigendecomposition(laplacian)
        features = data.x.numpy()
        gft = self.graph_fourier_transform(features, eigenvectors)
        data.x = torch.tensor(gft, dtype=torch.float)
        data.eigenvectors = torch.tensor(eigenvectors, dtype=torch.float)
        
        return data
    
    def __repr__(self):
        return 'GraphFourierTransform()'


class KHopsFractionDatasetTransform:
    def __init__(self, fraction, num_hops=2, seed=0):
        assert 0 < fraction <= 1, "Fraction must be between 0 and 1."
        self.fraction = fraction
        self.num_hops = num_hops
        self.seed = seed

    def __call__(self, data):
        num_nodes = data.num_nodes
        num_selected_nodes = int(num_nodes * self.fraction)

        # Start from a random node
        np.random.seed(self.seed)
        start_node = np.random.randint(0, num_nodes)
        selected_nodes, sub_edge_index, _, _ = k_hop_subgraph(start_node, self.num_hops, edge_index=data.edge_index, num_nodes=data.num_nodes, relabel_nodes=True)
        
        # If too many nodes are selected, truncate the list and recompute the subgraph
        if len(selected_nodes) > num_selected_nodes:
            selected_nodes = selected_nodes[:num_selected_nodes]
            sub_edge_index, _ = subgraph(selected_nodes, data.edge_index, relabel_nodes=True, num_nodes=num_nodes)

        sub_x = data.x[selected_nodes] if data.x is not None else None
        sub_y = data.y[selected_nodes] if data.y is not None else None

        # Subset the masks accordingly
        sub_train_mask = data.train_mask[selected_nodes] if data.train_mask is not None else None
        sub_test_mask = data.test_mask[selected_nodes] if data.test_mask is not None else None
        sub_val_mask = data.val_mask[selected_nodes] if data.val_mask is not None else None

        # Create the subgraph data object
        sub_data = Data(
            x=sub_x,
            edge_index=sub_edge_index,
            y=sub_y,
            train_mask=sub_train_mask,
            test_mask=sub_test_mask,
            val_mask=sub_val_mask,
        )

        return sub_data

    def __repr__(self):
        return 'KHopsFractionDatasetTransform()'
    

class SparseToDenseTransform(BaseTransform):
    def __call__(self, data: Data) -> Data:
        if data.x.layout == torch.sparse_csr:
            data.x = data.x.to_dense()  # Convert sparse CSR tensor to dense
        return data