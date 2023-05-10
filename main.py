import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.logging import init_wandb, log
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch.autograd import Variable
from tqdm import tqdm
from torchviz import make_dot
from torch_geometric.utils import to_networkx
import networkx as nx
from gnn import GCN, train, test
from explain import GNNInterpreter
import matplotlib.pyplot as plt

dataset = TUDataset(root='data/TUDataset', name='MUTAG')

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# model = GCN(in_channels=dataset.num_node_features, hidden_channels=64, out_channels=dataset.num_classes)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.CrossEntropyLoss()
# print(model)
# pbar = tqdm(range(1,100))
# for epoch in pbar:
#     train(model, train_loader, optimizer, criterion)
#     train_acc = test(model, train_loader)
#     test_acc = test(model, test_loader)
#     pbar.set_postfix_str(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# torch.save(model, "models/MUTAG_model.pth")

model = torch.load("models/MUTAG_model.pth")

init_graph = dataset[0]
target_class = 0

interpreter = GNNInterpreter(model.get_embedding_outputs, train_dataset)
print(init_graph.y)
pg = interpreter.train(dataset[0], target_class)

atom_indices = {
  0: "C",
  1: "N",
  2: "O",
  3: "F",
  4: "I",
  5: "Cl",
  6: "Br",
}

labels = dict(zip(range(pg.Xi.shape[0]), map(atom_indices.get, torch.argmax(init_graph.x, dim=1).detach().numpy())))
nx.draw_networkx(to_networkx(init_graph, to_undirected=True), with_labels=True, node_color = torch.argmax(init_graph.x, dim=1), labels=labels)
plt.savefig(f"explanations/Original Graph.png")
plt.cla()

explanation_graphs = pg.sample_explanations(5)
print("Example Outputs:\n", torch.softmax(interpreter.get_embedding_outputs(explanation_graphs)[1], dim=1).detach())
for i, data in enumerate(explanation_graphs.to_data_list()):
    G = to_networkx(data, to_undirected=True)
    # G.remove_edges_from(nx.selfloop_edges(G))
    print(torch.argmax(pg.Xi, dim=1))
    labels = dict(zip(range(pg.Xi.shape[0]), map(atom_indices.get, torch.argmax(pg.Xi, dim=1).detach().numpy())))
    print(labels)
    nx.draw_networkx(G, with_labels=True, node_color = torch.argmax(pg.Xi, dim=1), labels=labels)
    print("Node features equal:", torch.equal(torch.round(pg.Xi), init_graph.x))
    plt.savefig(f"explanations/Target Class {target_class} Explanation Graph {i}.png")
    plt.cla()