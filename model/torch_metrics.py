import torch

def sum_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor):
    """
    Returns the matrix of "x_i + y_j".
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :return: [R, C, D] sum matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    return x_col + y_row


def cost_mat(
    cost_s: torch.Tensor, cost_t: torch.Tensor, tran: torch.Tensor
) -> torch.Tensor:
    """
    Implement cost_mat for Gromov-Wasserstein discrepancy (GWD)

    Suppose the loss function in GWD is |a-b|^2 = a^2 - 2ab + b^2. We have:

    f1(a) = a^2,
    f2(b) = b^2,
    h1(a) = a,
    h2(b) = 2b

    When the loss function can be represented in the following format: loss(a, b) = f1(a) + f2(b) - h1(a)h2(b), we have

    cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
    cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T

    Args:
        cost_s: (ns, ns) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        cost_t: (nt, nt) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        tran: (ns, nt) matrix (torch tensor), representing the optimal transport from source to target domain.
    Returns:
        cost: (ns, nt) matrix (torch tensor), representing the cost matrix conditioned on current optimal transport
    """
    f1_st = torch.sum(cost_s ** 2, dim=1, keepdim=True) / cost_s.size(0)
    f2_st = torch.sum(cost_t ** 2, dim=1, keepdim=True) / cost_t.size(0)
    tmp = torch.sum(sum_matrix(f1_st, f2_st), dim=2)
    cost = tmp - 2 * cost_s @ tran @ torch.t(cost_t)
    return cost


def fgw(source, target, device, alpha, M=None):
    cost_source = distance_matrix(source, source)
    cost_target = distance_matrix(target, target)
    cost_source_target = distance_matrix(source, target)

    ns = cost_source.size(0)
    nt = cost_target.size(0)
    p_s = torch.ones(ns, 1) / ns
    p_t = torch.ones(nt, 1) / nt
    tran = torch.ones(ns, nt) / (ns * nt)
    p_s = p_s.to(device)
    p_t = p_t.to(device)

    if M is None:
        tran = tran.to(device)
    else:
        tran = M
    dual = (torch.ones(ns, 1) / ns).to(device)
    for m in range(10):
        cost = (
            alpha * cost_mat(cost_source, cost_target, tran)
            + (1 - alpha) * cost_source_target
        )
        kernel = torch.exp(-cost / torch.max(torch.abs(cost))) * tran
        b = p_t / (torch.t(kernel) @ dual)
        for i in range(5):
            dual = p_s / (kernel @ b)
            b = p_t / (torch.t(kernel) @ dual)
        tran = (dual @ torch.t(b)) * kernel
    if torch.isnan(tran).sum() > 0:
        tran = (torch.ones(ns, nt) / (ns * nt)).to(device)

    cost = (
        alpha * cost_mat(cost_source, cost_target, tran.detach().data)
        + (1 - alpha) * cost_source_target
    )
    d_fgw = (cost * tran.detach().data).sum()
    return d_fgw



def distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :param p:
    :return: [R, C] distance matrix
    """
    x_col = pts_src.unsqueeze(1)
    y_row = pts_dst.unsqueeze(0)
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)
    return distance
