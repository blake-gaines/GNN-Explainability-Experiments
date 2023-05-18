import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch
from torch_geometric.logging import init_wandb, log
from tqdm import tqdm
from torch_geometric.utils import to_networkx
import networkx as nx
from gnn import GCN, train, test
from explain import GNNInterpreter
import matplotlib.pyplot as plt
import os

if not os.path.isdir("data"): os.mkdir("data")
if not os.path.isdir("explanations"): os.mkdir("explanations")
if not os.path.isdir("models"): os.mkdir("models")

load_model = False
model_path = "models/MUTAG_model.pth"

dataset = TUDataset(root='data/TUDataset', name='MUTAG')
atom_indices = {
    0: "C",
    1: "N",
    2: "O",
    3: "F",
    4: "I",
    5: "Cl",
    6: "Br",
}

edge_indices = {
    0: "aromatic",
    1: "single",
    2: "double",
    3: "triple",
}

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

print('\n=============================================================\n')

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')
print()

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

if not load_model:
    model = GCN(in_channels=dataset.num_node_features, hidden_channels=64, out_channels=dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    print(model)
    pbar = tqdm(range(1,1000))
    for epoch in pbar:
        train(model, train_loader, optimizer, criterion)
        train_acc = test(model, train_loader)
        test_acc = test(model, test_loader)
        pbar.set_postfix_str(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    torch.save(model, model_path)
else:
    model = torch.load(model_path)
    test_acc = test(model, test_loader)
    print(f"Test Accuracy: {test_acc:.4f}")

print('\n=============================================================\n')

interpreter = GNNInterpreter(model.get_embedding_outputs, train_dataset)

init_graph_index = 0
for init_graph_index in range(3):
    init_graph = dataset[init_graph_index]
    print(f"Optmizing graph {init_graph_index}")
    print(f"Initial Graph Target: Class {int(init_graph.y)}")
    print()

    labels = dict(zip(range(init_graph.x.shape[0]), map(atom_indices.get, torch.argmax(init_graph.x, dim=1).detach().numpy())))
    nx.draw_networkx(to_networkx(init_graph, to_undirected=True), with_labels=True, node_color = torch.argmax(init_graph.x, dim=1), labels=labels)
    plt.savefig(f"explanations/Graph {init_graph_index} Original - Target {int(init_graph.y)}.png")
    plt.cla()

    for target_class in [0,1]:
        print(f"Optimizing for class {target_class}")

        pg = interpreter.train(dataset[0], target_class)

        explanation_graphs = pg.sample_explanations(1)
        print("Example Outputs:\n", torch.softmax(interpreter.get_embedding_outputs(explanation_graphs)[1], dim=1).detach())
        for i, data in enumerate(explanation_graphs.to_data_list()):
            G = to_networkx(data, to_undirected=True)
            # G.remove_edges_from(nx.selfloop_edges(G))
            x_indices = torch.argmax(pg.Xi, dim=1).detach().numpy()
            labels = dict(zip(range(pg.Xi.shape[0]), map(atom_indices.get, x_indices)))
            nx.draw_networkx(G, with_labels=True, node_color = x_indices, labels=labels)
            # print("Node features equal:", torch.equal(torch.round(pg.Xi), init_graph.x))
            plt.savefig(f"explanations/Graph {init_graph_index} Target {target_class} (#{i+1}).png")
            plt.cla()