import torch
import torch.nn as nn


class FederatedClient(nn.Module):
    """
    Simulated Federated Client
    """

    def __init__(self):
        super(FederatedClient, self).__init__()

    def forward(self, features):

        # Simulate local client learning
        noise = torch.randn_like(features) * 0.01

        updated_features = features + noise

        return updated_features


class FederatedServer(nn.Module):
    """
    Federated Averaging Server
    """

    def __init__(self, num_clients=5):

        super(FederatedServer, self).__init__()

        self.num_clients = num_clients

        self.clients = nn.ModuleList([
            FederatedClient()
            for _ in range(num_clients)
        ])

    def forward(self, features):

        client_outputs = []

        # Local client updates
        for client in self.clients:

            local_features = client(features)

            client_outputs.append(local_features)

        # Federated Averaging (FedAvg)
        aggregated_features = torch.stack(
            client_outputs,
            dim=0
        ).mean(dim=0)

        return aggregated_features