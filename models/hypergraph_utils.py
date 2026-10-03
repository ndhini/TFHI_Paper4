import torch


def compute_vertex_degree(H):
    """
    Compute vertex degree matrix.

    Dv(i) = Σe H(i,e)
    """

    degree = H.sum(dim=1)

    degree = torch.clamp(degree, min=1e-6)

    return torch.diag(degree)


def compute_edge_degree(H):
    """
    Compute hyperedge degree matrix.

    De(e) = Σv H(v,e)
    """

    degree = H.sum(dim=0)

    degree = torch.clamp(degree, min=1e-6)

    return torch.diag(degree)


def normalize_hypergraph(H):
    """
    Hypergraph normalization

    G = Dv^(-1/2) H De^(-1) H^T Dv^(-1/2)
    """

    device = H.device

    Dv = compute_vertex_degree(H)

    De = compute_edge_degree(H)

    dv = torch.diag(Dv)

    de = torch.diag(De)

    dv_inv_sqrt = torch.pow(dv, -0.5)

    de_inv = torch.pow(de, -1)

    Dv_inv = torch.diag(dv_inv_sqrt)

    De_inv = torch.diag(de_inv)

    G = (
        Dv_inv
        @ H
        @ De_inv
        @ H.t()
        @ Dv_inv
    )

    return G


def hypergraph_propagation(features, H):
    """
    Hypergraph Message Passing

    X' = GX
    """

    G = normalize_hypergraph(H)

    out = torch.matmul(G, features)

    return out