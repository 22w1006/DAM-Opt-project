import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer

class DAM(Optimizer):
    """
    Directional-Adaptive Momentum (DAM) Optimizer

    Args:
        params (iterable): 待优化参数或参数字典
        lr (float, optional): 学习率 (默认: 1e-3)
        rank (int, optional): 低秩曲率矩阵的秩 (默认: 5)
        epsilon (float, optional): Hessian估计的扰动大小 (默认: 1e-2)
        lambda_ (float, optional): 曲率敏感系数 (默认: 0.1)
        delta (float, optional): 数值稳定性常数 (默认: 1e-6)
        min_eig (float, optional): 最小特征值 (默认: 1e-4)
        gamma (float, optional): 动量缩放因子 (默认: 100)
        beta (float, optional): 动量衰减系数 (默认: 0.9)
        closure (callable, optional): 可选的闭包函数，用于计算损失和梯度
        c (float, optional): 动量更新的缩放因子 (默认: 0.01)
        threshold (float, optional): clipping的阈值系数 (默认: 0.1)
    """
    def __init__(self, params, lr=1e-3, rank=10, epsilon=5e-2, lambda_base=0.001, 
                 delta=1e-6, min_eig=1e-4, gamma=100, beta=0.9, c=0.01, threshold=0.1):
        # 注意: 这里的key必须和后续group['xxx']一致
        defaults = dict(lr=lr, rank=rank, epsilon=epsilon, lambda_base=lambda_base,
                       delta=delta, min_eig=min_eig, gamma=gamma, beta=beta, c=c, threshold=threshold)
        super(DAM, self).__init__(params, defaults)
        
        self.step_count = 0
        self.gamma = gamma

    def step(self, closure=None):
        self.step_count += 1
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    d = p.data.numel()
                    k = group['rank']
                    V = torch.randn(d, k, device=p.device)
                    state.update({
                        'V': torch.linalg.qr(V)[0] * 0.01,
                        'Lambda': torch.eye(k, device=p.device),
                        'momentum': torch.zeros_like(p.data),
                        'prev_grad': torch.zeros_like(p.data),
                        'Hv_ema': None
                    })

                # 曲率估计
                with torch.no_grad():
                    v = torch.randn_like(p.data).view(-1)
                    v = v / (v.norm() + 1e-8)

                    p.data.add_(group['epsilon'] * v.view_as(p.data))
                    perturbed_grad = self._get_grad(p)
                    p.data.sub_(group['epsilon'] * v.view_as(p.data))

                    Hv = (perturbed_grad.view(-1) - state['prev_grad'].view(-1)) / (group['epsilon'] + 1e-8)
                    if state['Hv_ema'] is not None:
                        Hv = group['beta'] * state['Hv_ema'] + (1 - group['beta']) * Hv
                    state['Hv_ema'] = Hv.detach().clone()

                # 低秩更新
                with torch.no_grad():
                    V, Lambda = state['V'], state['Lambda']
                    v_t = V.T @ Hv

                    Lambda.add_(torch.outer(v_t, v_t) + group['min_eig'] * torch.eye(Lambda.size(0), device=Lambda.device))
                    inv_Lambda = torch.linalg.pinv(Lambda)

                    temp = Hv.view(-1, 1) - V @ v_t.view(-1, 1)
                    update = temp @ (v_t.view(1, -1) @ inv_Lambda)
                    update = update * (torch.norm(update, dim=0).clamp_max(1.0) / (torch.norm(update, dim=0) + 1e-8)).view(1, -1)
                    V.add_(update)

                # 参数更新
                with torch.no_grad():
                    diag_C = (V ** 2) @ torch.diag(Lambda)
                    current_lambda = group['lambda_base'] * (1 + group['c'] * self.step_count)
                    alpha = torch.exp(-current_lambda * diag_C.sum().clamp(min=1e-8, max=1e6))

                    state['momentum'] = alpha * state['momentum'] + (1 - alpha) * grad

                    update = state['momentum'] / (diag_C.view_as(p.data) + group['delta'])
                    update_norm = torch.norm(update)
                    # 这里应为'threshold'，而不是'thresold'
                    max_update = group['threshold'] * torch.norm(p.data)
                    if update_norm > max_update:
                        update = update * (max_update / (update_norm + 1e-8))

                    p.data.add_(-group['lr'] * self.gamma * update)
                    state['prev_grad'] = grad.clone()

        return loss

    def _get_grad(self, p):
        return p.grad.data.clone() if p.grad is not None else torch.zeros_like(p.data)