import torch
import torch.nn as nn
import numpy as np

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())

class SinkhornDistance(nn.Module):
    r"""
    From https://dfdazac.github.io/sinkhorn.html
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none', feat_dim=512, dynamic_sinkhorn_weight=0):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.A = nn.Parameter(torch.eye(feat_dim), requires_grad=True)
        self.dynamic_sinkhorn_weight = dynamic_sinkhorn_weight
        
        assert (self.eps == 'diversify' or isinstance(self.eps, float))
        if self.eps == 'diversify':
            print('INFO: diversify training')

    def forward(self, x, y):
        # randomly choose eps if self.eps is 'diversify'
        if self.eps == 'diversify':
            this_eps =  2**np.random.uniform(np.log2(1e-4), np.log2(1)) # np.random.choice([1, 0.1, 0.01, 0.001, 0.0001])
        else:
            this_eps = self.eps
        
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        if self.dynamic_sinkhorn_weight:
            mu, nu = self.get_dynamic_sinkhorn_weight(C)
        else:
            mu = torch.empty(batch_size, x_points, dtype=torch.float,
                             requires_grad=False, device=x.device).fill_(1.0 / x_points).squeeze()
            nu = torch.empty(batch_size, y_points, dtype=torch.float,
                             requires_grad=False, device=y.device).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu, device=mu.device)
        v = torch.zeros_like(nu, device=nu.device)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-2 #1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = this_eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v, this_eps), dim=-1)) + u
            v = this_eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v, this_eps).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break
        # print(f"Converge in {actual_nits}")

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V, this_eps))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return (cost, mu, nu), pi, C
    
    def get_dynamic_sinkhorn_weight(self, C):
        "C is the cost matrix, its range is [0, 2], first convert it to similarity matrix in range [0, 1]"
        print('INFO: Use dynamic sinkhorn weights')
        with torch.no_grad():
            simi_matrix = 1 - C / 2
            # print(simi_matrix.min(), simi_matrix.max())
            # normalize simi_matrix to [0, 1]
            min_val = torch.min(torch.min(simi_matrix, -1, keepdim=True)[0], -1, keepdim=True)[0]
            simi_matrix = simi_matrix - min_val
            max_val = torch.max(simi_matrix, -1, keepdim=True)[0]
            simi_matrix = simi_matrix / max_val
            print(simi_matrix.min(), simi_matrix.max())
            bs = C.shape[0]
            norm_term1 = simi_matrix.shape[1]
            s1 = torch.sum(simi_matrix, dim=1)
            s2 = torch.sum(simi_matrix, dim=-1, keepdim=True)
            norm_term2 = (simi_matrix.permute(0, 2, 1) @ s2).view(bs, -1) / s1
            feat_w_C = 1 / norm_term1
            feat_w_S = s1 / (norm_term1 * norm_term2)
            feat_w_S = feat_w_S / feat_w_S.sum()
        
        mu = torch.empty(bs, norm_term1, dtype=torch.float,
                         requires_grad=False, device=C.device).fill_(feat_w_C).squeeze()
        nu = feat_w_S / torch.sum(feat_w_S, -1, keepdim=True)
        return mu, nu

    def M(self, C, u, v, this_eps):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / this_eps

    
    def _cost_matrix(self, x, y):
        "Returns the matrix of cosine distance."
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = y / torch.norm(y, dim=-1, keepdim=True)
        # C = 1 - torch.bmm(x, torch.bmm(self.A.unsqueeze(0).expand(x.shape[0], -1, -1), y.permute(0, 2, 1)))
        C = 1 - torch.bmm(x, y.permute(0, 2, 1))
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1
    
