import torch
from torch_geometric.datasets import TUDataset
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.logging import init_wandb, log
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm
from torchviz import make_dot
from torch_geometric.utils import to_networkx
import networkx as nx

class ProbGraph:
    def __init__(self, data):
        self.has_edge_attributes = data.edge_attr is not None 
        self.has_node_attributes = data.x is not None
        # self.Omega = Variable(torch.Tensor(to_dense_adj(data.edge_index)), requires_grad=True)
        self.Omega = Variable(torch.Tensor(0*torch.ones((data.x.shape[0], data.x.shape[0]))), requires_grad=True)
        self.params = {"Omega": self.Omega}
        if self.has_edge_attributes: 
            # self.H = Variable(torch.Tensor(data.edge_attr), requires_grad=True)
            self.H = Variable(torch.Tensor(0*torch.ones(data.edge_attr.shape)), requires_grad=True)
            self.params["H"] = self.H
        if self.has_node_attributes: 
            # self.Xi = Variable(torch.Tensor(data.x), requires_grad=True)
            self.Xi = Variable(torch.Tensor(0*torch.ones(data.x.shape)), requires_grad=True)
            self.params["Xi"] = self.Xi

    def parameters(self):
        return self.params

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

class GNNInterpreter():
    def __init__(self, get_embedding_outputs, train_dataset):
        super().__init__()

        self.tau_a = 0.2
        self.tau_z = 0.2
        self.tau_x = 0.2
        self.mu = 10

        self.K = 10 # Monte Carlo # Samples
        self.B = 20 # Max Budget

        self.reg_weights = {
            "Omega L1": [10, 5],
            "Omega L2": [5, 2],
            "Budget Penalty": [20, 10],
            "Connectivity Incentive": [1,2],
        }

        self.get_embedding_outputs = get_embedding_outputs

        self.average_phi = self.get_average_phi(train_dataset).detach()

    def get_average_phi(self, dataset):
        dataloader = DataLoader(train_dataset, batch_size=1)
        embedding_sum = None
        n_instances = torch.zeros(dataset.num_classes)
        for batch in dataloader:
            embeddings = self.get_embedding_outputs(batch)[0]
            if embedding_sum is None: 
                embedding_sum = torch.zeros(dataset.num_classes, embeddings.shape[-1])
            embedding_sum[batch.y] += torch.sum(embeddings, dim=0)
            n_instances[batch.y] += dataloader.batch_size
        return embedding_sum / torch.unsqueeze(n_instances, 1)
    
    def bernoulli_kl(self, p1, p2):
        return p1*torch.log(p1/p2) + (1-p1)*torch.log((1-p1)/(1-p2))

    def regularizer(self, pg, class_index):
        r = dict()

        r["Omega L1"] = torch.norm(pg.Omega, 1) * self.reg_weights["Omega L1"][class_index]
        r["Omega L2"] = torch.norm(pg.Omega, 2) * self.reg_weights["Omega L2"][class_index]
        # r["Omega L1"] = sum(torch.norm(parameter, 1) for parameter in pg.parameters().values()) * self.reg_weights["Omega L1"][class_index]
        # r["Omega L2"] = sum(torch.norm(parameter, 2) for parameter in pg.parameters().values()) * self.reg_weights["Omega L2"][class_index]

        r["Budget Penalty"] = F.softplus(torch.sigmoid(r["Omega L1"])-self.B)**2 * self.reg_weights["Omega L1"][class_index] * min(self.iteration/500, 1)

        # Theta = torch.sigmoid(pg.Omega).squeeze()
        # connectivity_incentive = 0
        # for i in range(pg.Omega.shape[0]):
        #     for j in range(pg.Omega.shape[0]):
        #         if i != j:
        #             Pij = Theta[i][j]
        #             for k in range(pg.Omega.shape[0]):
        #                 if i != k:
        #                     Pik = Theta[i][k]
        #                     # connectivity_incentive += self.bernoulli_kl(Pij, Pik)
        #                     connectivity_incentive += F.kl_div(Pij, Pik)
        # r["Connectivity Incentive"] = connectivity_incentive * self.reg_weights["Connectivity Incentive"][class_index]
        
        return sum(r.values())

    def train(self, init_graph, class_index, max_iter=1000):
        pg = ProbGraph(init_graph)
        self.pg = pg

        optimizer = torch.optim.SGD(pg.parameters().values(), lr=1, maximize=True) # , momentum=0.9

        self.pbar = tqdm(range(max_iter))

        for self.iteration in self.pbar:
            optimizer.zero_grad()

            sampled_graphs = pg.sample_train(self.K, self.tau_a, self.tau_z, self.tau_x)
            # print(sampled_graphs.get_example(0).requires_grad_())

            embeddings, outputs = self.get_embedding_outputs(sampled_graphs)

            class_logits = outputs[:, class_index]
            embedding_similarities = F.cosine_similarity(embeddings, self.average_phi[class_index], dim=1)
            regularization = self.regularizer(pg, class_index)
            # mean_error = torch.mean(class_logits)
            mean_L = torch.mean(class_logits+self.mu*embedding_similarities)
            loss = mean_L - regularization

            # if self.iteration == 0:
            #     all_params = pg.parameters()
            #     all_params.update(model.named_parameters())
            #     self.computation_graph = make_dot(loss, all_params)#, show_attrs=True, show_saved=True)

            # print(f"Loss: {float(loss)}    ({mean_error:.2} Error, {regularization:.2f} Regularization)")
            # print(loss, mean_error, regularization)
            self.pbar.set_postfix_str(f"Loss: {float(loss):.2f}    ({mean_L:.2f} Error, {regularization:.2f} Regularization)")

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(pg.parameters().values(), 100)
            optimizer.step()
            
            # explanation_graph = pg.sample_explanations(3)
            # print("Example Output:\n", torch.softmax(self.get_embedding_outputs(explanation_graph)[1], dim=1))
            # print(self.get_embedding_outputs(explanation_graph)[1])

        return pg