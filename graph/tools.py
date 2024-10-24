import numpy as np
import torch

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak



def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    # print("degs_inv_sqrt: ", degs_inv_sqrt)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    # print("norm_degs_matrix: ", norm_degs_matrix)
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)

##Edit Code Begin 7
def normalize_adjacency_matrix_cuda(A):
    node_degrees = A.sum(-1).to("cuda")
    degs_inv_sqrt = torch.pow(node_degrees, -0.5).to("cuda")
    norm_degs_matrix = torch.eye(len(node_degrees)).to("cuda") * degs_inv_sqrt
    
    return (norm_degs_matrix @ A @ norm_degs_matrix).float()

def normalize_dsg_adjacency_matrix(A):
    node_degrees = A.sum(-1).to("cuda")
    degs_inv_sqrt = torch.pow(node_degrees, -0.5).to("cuda")
    # print("degs_inv_sqrt.shape: ", degs_inv_sqrt.shape)
    # norm_degs_matrix = torch.diag(degs_inv_sqrt)
    norm_degs_matrix = torch.stack([torch.stack([ torch.eye(node_degrees.shape[-1]).to("cuda")* degs_inv_sqrt[n][t] 
                                                for t in range(degs_inv_sqrt.shape[1])]).to("cuda")
                                                    for n in range(degs_inv_sqrt.shape[0])]).to("cuda")
    return (norm_degs_matrix @ A @ norm_degs_matrix).float()

def normalize_dsg_A_tilta(A):
    # node_degrees = A.to("cpu").sum(-1)
    node_degrees = A.to("cuda").sum(-1)
    degs_inv= torch.pow(node_degrees, -1).to("cuda")
    # print("degs_inv.shape: ", degs_inv.shape)
    # print(degs_inv[0,0])
    norm_degs_matrix = torch.stack([torch.stack([ torch.eye(node_degrees.shape[-1]).to("cuda")* degs_inv[n][t] 
                                                for t in range(degs_inv.shape[1])]) .to("cuda")
                                                    for n in range(degs_inv.shape[0])]).to("cuda")
    return (A @ norm_degs_matrix).float()

##Edit Code End 7

def get_adjacency_matrix(edges, num_nodes):
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for edge in edges:
        A[edge] = 1.
    return A