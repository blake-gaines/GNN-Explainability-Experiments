import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.loader import DataLoader
from probgraph import ProbGraph

class GNNInterpreter:
    def __init__(self, get_embedding_outputs, train_dataset):
        super().__init__()

        self.tau_a = 0.2
        self.tau_z = 0.2
        self.tau_x = 0.2
        self.mu = 10

        self.K = 10 # Monte Carlo # Samples
        self.B = 20 # Max Budget

        self.reg_weights = {
            "RL1": [10, 5],
            "RL2": [5, 2],
            "Budget Penalty": [20, 10],
            "Connectivity Incentive": [1,2],
            "Deviation": [0.001,0.001],
        }

        self.get_embedding_outputs = get_embedding_outputs

        self.average_phi = self.get_average_phi(train_dataset).detach()

    def get_average_phi(self, dataset):
        dataloader = DataLoader(dataset, batch_size=1)
        embedding_sum = None
        n_instances = torch.zeros(dataset.num_classes)
        for batch in dataloader:
            embeddings = self.get_embedding_outputs(batch)[0]
            if embedding_sum is None: 
                embedding_sum = torch.zeros(dataset.num_classes, embeddings.shape[-1])
            embedding_sum[batch.y] += torch.sum(embeddings, dim=0)
            n_instances[batch.y] += dataloader.batch_size
        return embedding_sum / torch.unsqueeze(n_instances, 1)

    def regularizer(self, pg, class_index):
        r = dict()

        for name, parameter in pg.parameters.items():
            r[f"{name} L1"] = torch.norm(parameter, 1) * self.reg_weights["RL1"][class_index]
            r[f"{name} L2"] = torch.norm(parameter, 2) * self.reg_weights["RL2"][class_index]

        r["Budget Penalty"] = F.softplus(torch.norm(torch.sigmoid(pg.Omega), 1)-self.B)**2 * self.reg_weights["Budget Penalty"][class_index] * min(self.iteration/500, 1)

        # if pg.init_graph is not None:
        #     r["Deviation"] = torch.norm(pg.init_adj_matrix-torch.sigmoid(pg.Omega), 1) ** 2
        #     if pg.has_node_attributes:
        #         r["Deviation"] += torch.norm(pg.init_graph.x-torch.sigmoid(pg.Xi), 1) ** 2
        #     if pg.has_edge_attributes:
        #         r["Deviation"] += torch.norm(pg.init_graph.edge_attr-torch.sigmoid(pg.H), 1) ** 2
        # r["Deviation"] *= self.reg_weights["Deviation"][class_index]

        # connectivity_incentive = 0
        # for i in range(pg.Omega.shape[0]):
        #     for j in range(pg.Omega.shape[0]):
        #         if i != j:
        #             Pij = pg.Omega[i][j]
        #             for k in range(pg.Omega.shape[0]):
        #                 if i != k and j!=k:
        #                     Pik = pg.Omega[i][k]
        #                     connectivity_incentive += F.kl_div(Pij, Pik, log_target=True)
        # r["Connectivity Incentive"] = connectivity_incentive * self.reg_weights["Connectivity Incentive"][class_index]
        
        return r

    def train(self, init_graph, class_index, max_iter=5000):
        pg = ProbGraph(init_graph)
        self.pg = pg

        optimizer = torch.optim.SGD(pg.parameters.values(), lr=0.1, maximize=True) # , momentum=0.9

        self.pbar = tqdm(range(max_iter))

        for self.iteration in self.pbar:
            optimizer.zero_grad()

            sampled_graphs = pg.sample_train(self.K, self.tau_a, self.tau_z, self.tau_x)

            embeddings, outputs = self.get_embedding_outputs(sampled_graphs)

            class_logits = outputs[:, class_index]
            embedding_similarities = F.cosine_similarity(embeddings, self.average_phi[class_index], dim=1)
            regularization_dict = self.regularizer(pg, class_index)
            regularization = sum(regularization_dict.values())
            mean_L = torch.mean(class_logits+self.mu*embedding_similarities)
            loss = mean_L - regularization
            # loss = -regularization
            # loss = -regularization_dict["Deviation"]

            # if self.iteration == 0:
            #     all_params = pg.parameters
            #     all_params.update(model.named_parameters())
            #     self.computation_graph = make_dot(loss, all_params)#, show_attrs=True, show_saved=True)

            self.pbar.set_postfix_str(f"Loss: {float(loss):.2f}    ({mean_L:.2f} Objective, {regularization:.2f} Regularization)")

            loss.backward()
            optimizer.step()
            
            # explanation_graph = pg.sample_explanations(3)
            # print("Example Output:\n", torch.softmax(self.get_embedding_outputs(explanation_graph)[1], dim=1))
            # print(self.get_embedding_outputs(explanation_graph)[1])
        print(regularization_dict)

        return pg