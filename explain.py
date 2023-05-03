import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse

class ProbGraph:
    def __init__(self, data):
        self.has_edge_attributes = data.edge_attr is not None 
        self.has_node_attributes = data.node_attr is not None
        self.Omega = torch.Tensor(to_dense_adj(data.edge_index))
        self.params = [self.Omega]
        if self.has_edge_attributes: 
            self.H = torch.Tensor(data.edge_attr)
            self.params.append(self.H)
        if self.has_node_attributes: 
            self.Xi = torch.Tensor(data.X)
            self.params.append(self.Xi)

    def parameters(self):
        return self.params

    def sample_train(self):
        sampled_graph = dict()

        a_epsilon = torch.rand((self.K, *self.Omega.shape))
        a = torch.sigmoid((self.Omega + torch.log(a_epsilon)-torch.log(1-a_epsilon))/self.tau_a)
        sampled_graph["edge_index"] = dense_to_sparse(a)

        if self.has_edge_attributes:
            z_epsilon = torch.rand((self.K, *self.H.shape))
            z = torch.nn.Softmax((self.H - torch.log(-torch.log(z_epsilon)))/self.tau_z)
            sampled_graph["edge_attr"] = z
        
        if self.has_node_attributes:
            x_epsilon = torch.rand((self.K, *self.Xi.shape))
            x = torch.nn.Softmax((self.Xi - torch.log(-torch.log(x_epsilon)))/self.tau_x)
            sampled_graph["x"] = x

        return Data(sampled_graph)
    
    def sample_explanation(self):
        # Sigma = torch.sigmoid(self.Omega)
        # P = torch.nn.Softmax(self.Xi)
        # Q = torch.nn.Softmax(self.H)
        raise NotImplementedError

class GNNInterpreter():
    def __init__(self):
        super().__init__()

        self.initialize_parameters()

        self.tau_a = 5
        self.tau_z = 5
        self.tau_x = 5
        self.mu = 1

        self.K = 10 # Monte Carlo # Samples

        self.get_embedding_outputs = None

        self.average_phi = None

    def get_average_phi(self, train_loader):
        embedding_sum = None
        n_instances = 0
        for batch in train_loader:

            embedding_sum += self.get_embedding(batch)
            n_instances += train_loader.batch_size
        return embedding_sum / n_instances

    def regularizer(self):
        return 0

    def train(self, init_graph, class_index, max_iter=1000):

        pg = ProbGraph(init_graph)

        optimizer = torch.optim.SGD(pg.parameters(), lr=0.001, momentum=0.9)

        for _ in range(max_iter):
            self.optimizer.zero_grad()

            sampled_graph = pg.sample_train()


            embeddings, outputs = self.get_embedding_outputs(sampled_graph)

            class_probabilities = outputs[class_index]
            embedding_similarities = F.cosine_similarity(embeddings, self.average_phi[class_index])

            loss = torch.mean(class_probabilities+self.mu*embedding_similarities) + self.regularizer()

            loss.backward()
            optimizer.step()

        return pg
