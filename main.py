import argparse
import subprocess
import sys
import time
import socket

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def start_server(num_rounds, num_clients):
    free_port = find_free_port()
    cmd = [sys.executable, "pytorch_server.py", "--num_rounds", str(num_rounds), "--num_clients", str(num_clients), "--port", str(free_port)]
    return subprocess.Popen(cmd), free_port

def start_clients(num_clients, server_port):
    clients = []
    for _ in range(num_clients):
        cmd = [sys.executable, "pytorch_client.py", "--num_clients", str(num_clients), "--port", str(server_port)]
        clients.append(subprocess.Popen(cmd))
    return clients

def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning with MNIST")
    parser.add_argument("--num_clients", type=int, default=2, help="Number of clients to simulate")
    parser.add_argument("--num_rounds", type=int, default=3, help="Number of federated learning rounds")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Starting federated learning with {args.num_clients} clients and {args.num_rounds} rounds")
    
    # Start the server
    server_process, server_port = start_server(args.num_rounds, args.num_clients)
    
    # Give the server some time to start
    time.sleep(3)
    
    # Start the clients
    client_processes = start_clients(args.num_clients, server_port)
    
    # Wait for the server to finish
    server_process.wait()
    
    # Terminate client processes
    for client in client_processes:
        client.terminate()

if __name__ == "__main__":
    main()