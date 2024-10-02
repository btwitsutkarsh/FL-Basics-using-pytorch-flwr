import argparse
import sys
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import OrderedDict

import flwr as fl
print(f"Flower version: {fl.__version__}")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

def get_eval_fn(model):
    def evaluate(weights: fl.common.NDArrays) -> tuple[float, dict]:
        params_dict = zip(model.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        testset = MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
        testloader = DataLoader(testset, batch_size=32, shuffle=False)
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy, {"accuracy": accuracy}
    return evaluate

def start_server(num_rounds, num_clients, port):
    model = Net()
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=get_eval_fn(model),
    )
    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--num_rounds", type=int, default=3, help="Number of federated learning rounds")
    parser.add_argument("--num_clients", type=int, default=2, help="Number of clients")
    parser.add_argument("--port", type=int, default=8080, help="Port for the server")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Starting server with {args.num_rounds} rounds and {args.num_clients} clients on port {args.port}")
    start_server(args.num_rounds, args.num_clients, args.port)