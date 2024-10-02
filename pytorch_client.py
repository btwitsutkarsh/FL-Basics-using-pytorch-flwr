import argparse
import flwr as fl
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from collections import OrderedDict

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

class MNISTClient(fl.client.NumPyClient):
    def __init__(self, train_data, val_data, num_clients):
        self.train_data = train_data
        self.val_data = val_data
        self.model = Net()
        self.num_clients = num_clients

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_loader = DataLoader(self.train_data, batch_size=32, shuffle=True)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(5):  # Local epochs
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config={}), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        val_loader = DataLoader(self.val_data, batch_size=32)
        criterion = nn.CrossEntropyLoss()
        loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = self.model(data)
                loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        accuracy = correct / len(self.val_data)
        return loss, len(self.val_data), {"accuracy": accuracy}

def client_fn(cid: str, num_clients: int) -> fl.client.Client:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = MNIST("./data", train=True, download=True, transform=transform)
    valset = MNIST("./data", train=False, download=True, transform=transform)
    
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    trainset = datasets[int(cid)]
    return MNISTClient(trainset, valset, num_clients).to_client()

def start_client(num_clients: int, port: int):
    fl.client.start_numpy_client(f"127.0.0.1:{port}", client=client_fn("0", num_clients), grpc_max_message_length=0xff)

def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument("--num_clients", type=int, default=2, help="Number of clients")
    parser.add_argument("--port", type=int, default=8080, help="Port for the server")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    start_client(num_clients=args.num_clients, port=args.port)