import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch.autograd import Variable

class ProbGraph:
    def __init__(self, init_graph=None, init_method="normal", num_nodes=10, node_feature_dim=8, edge_feature_dim=4):
        if init_graph is not None:
            num_nodes = init_graph.num_nodes
            num_edges = init_graph.num_edges 
            node_feature_dim = init_graph.num_node_features
            edge_feature_dim = init_graph.num_edge_features

        ## TODO: FIX THIS
        self.has_edge_attributes = init_graph.edge_attr is not None 
        self.has_node_attributes = init_graph.x is not None

        if init_method == "normal":
            self.Omega = Variable(torch.normal(torch.zeros((num_nodes, num_nodes)), torch.ones((num_nodes, num_nodes))), requires_grad=True)
        elif init_method == "copy":
            self.Omega = Variable(torch.Tensor(to_dense_adj(init_graph.edge_index)), requires_grad=True)
        elif init_method == "zero":
            self.Omega = Variable(torch.Tensor(0*torch.ones((init_graph.x.shape[0], init_graph.x.shape[0]))), requires_grad=True)
        else:
            raise NotImplementedError("Unknown Init Method")
        self.parameters = {"Omega": self.Omega}
        if self.has_edge_attributes: 
            if init_method == "normal":
                self.H = Variable(torch.normal(torch.zeros((num_edges, edge_feature_dim)), torch.ones((num_edges, edge_feature_dim))), requires_grad=True)
            elif init_method == "copy":
                self.H = Variable(torch.Tensor(init_graph.edge_attr), requires_grad=True)
            elif init_method == "zero":
                self.H = Variable(torch.zeros((num_edges, edge_feature_dim)), requires_grad=True)
            self.parameters["H"] = self.H
        if self.has_node_attributes: 
            if init_method == "normal":
                self.Xi = Variable(torch.normal(torch.zeros((num_nodes, node_feature_dim)), torch.ones((num_nodes, node_feature_dim))), requires_grad=True)
            elif init_method == "copy":
                self.Xi = Variable(torch.Tensor(init_graph.x), requires_grad=True)
            elif init_method == "zero":
                self.Xi = Variable(torch.zeros((num_nodes, node_feature_dim)), requires_grad=True)
            self.parameters["Xi"] = self.Xi

    def sample_train(self, K, tau_a, tau_z, tau_x):
        sampled_graphs = []
        for _ in range(K):
            sampled_graph = dict()
            a_epsilon = torch.rand(self.Omega.shape)
            a = torch.sigmoid((self.Omega + torch.log(a_epsilon)-torch.log(1-a_epsilon))/tau_a)
            sampled_graph["edge_index"], sampled_graph["edge_weight"] = dense_to_sparse(a)

            if self.has_edge_attributes:
                z_epsilon = torch.rand(self.H.shape)
                z = torch.softmax((self.H - torch.log(-torch.log(z_epsilon)))/tau_z, 1)
                sampled_graph["edge_attr"] = z
            
            if self.has_node_attributes:
                x_epsilon = torch.rand(self.Xi.shape)
                x = torch.softmax((self.Xi - torch.log(-torch.log(x_epsilon)))/tau_x, 1) 
                sampled_graph["x"] = x

            sampled_graphs.append(Data(**sampled_graph))

        return Batch.from_data_list(sampled_graphs)
    
    def get_latents(self):
        latent_dict = {"Theta": torch.sigmoid(self.Omega)}
        if self.has_node_attributes:
            latent_dict["P"] = torch.softmax(self.Xi, 1)
        if self.has_edge_attributes:
            latent_dict["Q"] = torch.softmax(self.H, 1)
        return latent_dict
    
    def sample_explanations(self, n=1):
        latent_dict = self.get_latents()

        A_dist = torch.distributions.Bernoulli(latent_dict["Theta"])
        if self.has_node_attributes:
            X_dist = torch.distributions.Categorical(latent_dict["P"])
        if self.has_edge_attributes:
            Z_dist = torch.distributions.Categorical(latent_dict["Q"])

        sampled_graphs = []
        for _ in range(n):
            sampled_graph = dict()
            sampled_graph["edge_index"], sampled_graph["edge_weight"] = dense_to_sparse(A_dist.sample().float())
            if self.has_node_attributes:
                sampled_graph["x"] = F.one_hot(X_dist.sample(), num_classes=self.Xi.shape[1]).float()
            if self.has_edge_attributes:
                sampled_graph["edge_attr"] = F.one_hot(Z_dist.sample(), num_classes=self.H.shape[1]).float()
            sampled_graphs.append(Data(**sampled_graph))

        return Batch.from_data_list(sampled_graphs)