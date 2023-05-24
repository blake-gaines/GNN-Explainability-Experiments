import torch
from torch_geometric.datasets import TUDataset
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch.autograd import Variable
import networkx as nx

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# from torch_geometric.nn.dense import DenseGCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_channels, hidden_channels, out_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
    
    def get_embedding_outputs(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        embedding = self.embed(x, edge_index, batch, edge_weight)
        output =  self.classify(embedding)
        return embedding, output

    def forward(self, x, edge_index, batch, edge_weight=None):
        if edge_weight is None: edge_weight = torch.ones(edge_index.shape[1])
        embedding = self.embed(x, edge_index, batch, edge_weight)
        output =  self.classify(embedding)
        return output

    def embed(self, x, edge_index, batch, edge_weight=None):
        if edge_weight is None: edge_weight = torch.ones(edge_index.shape[1])
        
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        return x
    
    def classify(self, x):
        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        return x
    
def train(model, train_loader, optimizer, criterion):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(model, loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)  
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.
