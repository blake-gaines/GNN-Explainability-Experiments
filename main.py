import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch
import wandb
from tqdm import tqdm
from torch_geometric.utils import to_networkx
import networkx as nx
from gnn import GCN, train, test
from explain import GNNInterpreter
import matplotlib.pyplot as plt
import os
import io
from PIL import Image

if not os.path.isdir("data"): os.mkdir("data")
if not os.path.isdir("explanations"): os.mkdir("explanations")
if not os.path.isdir("models"): os.mkdir("models")

epochs = 100
num_inits = 4
num_explanations = 3

load_model = True
model_path = "models/MUTAG_model.pth"

log_run = True


if log_run: 
    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project="GNN-Expanations", 
        # name=f"test_run", 
        # Track hyperparameters and run metadata
        config={
        "learning_rate": 0.02,
        "architecture": "GNN",
        "dataset": "MUTAG",
        "epochs": epochs,
    })

    explanation_table = wandb.Table(columns=["Original Graph Index", "Init Graph", "Target Class", "Explanations", "Logits"])

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
    pbar = tqdm(range(1,epochs))
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
for init_graph_index in range(num_inits):
    init_graph = test_dataset[init_graph_index]
    print(f"Optmizing graph {init_graph_index}")
    print(f"Initial Graph Target: Class {int(init_graph.y)}")
    print()

    labels = dict(zip(range(init_graph.x.shape[0]), map(atom_indices.get, torch.argmax(init_graph.x, dim=1).detach().numpy())))
    fig, ax = plt.subplots()
    G = to_networkx(init_graph, to_undirected=True)
    pos = nx.spring_layout(G, seed=7)
    nx.draw_networkx(G, pos=pos, with_labels=True, node_color=torch.argmax(init_graph.x, dim=1), labels=labels)
    plt.savefig(f"explanations/Graph {init_graph_index} Original - Target {int(init_graph.y)}.png")
    if log_run:
        init_graph_image = wandb.Image(fig)
    plt.close()

    for target_class in [0,1]:
        print(f"Optimizing for class {target_class}")

        pg = interpreter.train(dataset[0], target_class, max_iter=100)

        explanation_graphs = pg.sample_explanations(num_explanations)
        example_outputs = interpreter.get_embedding_outputs(explanation_graphs)[1]
        eg_images = []
        print("Example Outputs:\n", torch.softmax(example_outputs, dim=1).detach())
        for i, data in enumerate(explanation_graphs.to_data_list()):
            G = to_networkx(data, to_undirected=True)
            # G.remove_edges_from(nx.selfloop_edges(G))
            x_indices = torch.argmax(pg.Xi, dim=1).detach().numpy()
            labels = dict(zip(range(pg.Xi.shape[0]), map(atom_indices.get, x_indices)))
            fig, ax = plt.subplots()
            pos = nx.spring_layout(G, seed=7)
            nx.draw_networkx(G, pos=pos, with_labels=True, node_color = x_indices, labels=labels)
            # print("Node features equal:", torch.equal(torch.round(pg.Xi), init_graph.x))
            plt.savefig(f"explanations/Graph {init_graph_index} Target {target_class} (#{i+1}).png")

            if log_run:
                # img_buf = io.BytesIO()
                # plt.savefig(img_buf, format='png')
                # eg_images.append(wandb.Image(img_buf, caption=f"Graph {init_graph_index} Target {target_class} (#{i+1}).png"))
                eg_images.append(wandb.Image(fig))
            plt.close()

        if log_run:
            explanation_table.add_data(
                init_graph_index,
                init_graph_image,
                target_class,
                eg_images,
                example_outputs.detach()
            )
if log_run:
    wandb.log({"Explanations": explanation_table})