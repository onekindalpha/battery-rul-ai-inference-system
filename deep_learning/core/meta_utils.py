# deep_learning/core/meta_utils.py
"""
BMAML-SVGD utilities.

CRITICAL FIX:
  - meta_thetas -> inner loop -> meta_loss 그래프가 detach로 끊기던 문제 해결.
  - detach_phi only (FOMAML-style), theta keeps graph.
  - [NEW] Removed early-stopping condition in inner loop for stability.
"""

import math
from collections import OrderedDict
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.stateless import functional_call

from deep_learning.core.config import Config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # [수정] GPU 자동 감지

SVGD_CLIP = 5.0  # [수정] 기존 10.0 → 5.0으로 줄임 (Clipping 강화)
DEFAULT_STOP_TOL = 1e-3  # [수정] 기존 1e-4 → 1e-3으로 완화 (inner loop 안정화)
INNER_LR_MAX = 5e-3  # [수정] 기존 1e-2 → 5e-3으로 줄임 (inner_lr 클리핑)
NEG_LOGP_CLIP = 1e4


def make_leaf_thetas(thetas: List[torch.Tensor], detach: bool = True) -> List[torch.Tensor]:
    if detach:
        return [t.clone().detach().requires_grad_(True) for t in thetas]
    else:
        return [t.clone().requires_grad_(True) for t in thetas]

class ParamVectorizer:
    def __init__(self, model):
        self.names, self.shapes = [], []
        for name, p in model.named_parameters():
            self.names.append(name)
            self.shapes.append(p.shape)
        self.total = sum(int(np.prod(s)) for s in self.shapes)

    def vector_to_params(self, vec: torch.Tensor) -> OrderedDict:
        params = OrderedDict()
        offset = 0
        for name, shape in zip(self.names, self.shapes):
            num = int(np.prod(shape))
            params[name] = vec[offset:offset + num].view(shape)
            offset += num
        return params

    def assign_to_model_(self, model, vec: torch.Tensor) -> None:
        offset = 0
        for p in model.parameters():
            num = int(p.numel())
            p.data.copy_(vec[offset:offset + num].view_as(p))
            offset += num

def physics_loss(pred: torch.Tensor,
                 cycles: Optional[torch.Tensor],
                 c: Config) -> torch.Tensor:
    """
    Physical priors:
      1) RUL should be non-increasing with cycle (monotonicity)
      2) Should be smooth (low curvature)
      3) SEI-like degradation: q_loss ~ sqrt(t)
    """
    if cycles is None or cycles.numel() <= 2:
        return torch.tensor(0.0, device=pred.device)
    if cycles.var() < 1e-8:
        return torch.tensor(0.0, device=pred.device)

    # ---- sort by cycle to make priors valid even if input order is shuffled
    order = torch.argsort(cycles)
    t = cycles[order].float()
    p = pred[order] # pred_rul

    eps = 1e-6

    # 1) monotonic decreasing: penalize increases in RUL
    diff = p[1:] - p[:-1]
    mono_loss = F.softplus(diff).mean()  # stronger than ReLU

    # 2) smoothness on curvature (2nd difference)
    if diff.numel() >= 2:
        curvature = diff[1:] - diff[:-1]
        smooth_loss = (curvature ** 2).mean()
    else:
        smooth_loss = torch.tensor(0.0, device=p.device)

    # 3) SEI prior: q_loss ~ sqrt(t)
    rul_max = p.max()
    q_loss = rul_max - p  # should grow with cycle
    sqrt_t = torch.sqrt(t + eps)

    # normalize to compare shapes only
    sqrt_t_norm = sqrt_t / (sqrt_t.max() + eps)
    q_loss_norm = q_loss / (q_loss.max() + eps)

    sei_loss = F.mse_loss(q_loss_norm, sqrt_t_norm)

    # Final weighted loss
    total = mono_loss + 0.1 * smooth_loss + sei_loss
    
    # [CRITICAL FIX] c.physics_weight를 최종적으로 곱해 모델의 전체 Physics Loss를 결정
    return c.physics_weight * total

def neg_logp_particle(model, vecizer: ParamVectorizer, theta: torch.Tensor,
                      seq: torch.Tensor, summ: torch.Tensor, y: torch.Tensor,
                      cycles: Optional[torch.Tensor], c: Config) -> torch.Tensor:
    params = vecizer.vector_to_params(theta)
    out, rs, rm, _ = functional_call(model, params, (seq, summ))
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    pred_rul = out.squeeze(-1) # pred -> pred_rul (명시적 이름 변경)

    mse = F.mse_loss(pred_rul, y)
    aux = torch.tensor(0.0, device=pred_rul.device)

    if c.physics_weight > 0: # cycles is not None 조건은 함수 내부로 이동
        # [CRITICAL FIX] physics_loss 호출 시 인수를 pred_rul로 변경
        aux = aux + physics_loss(pred_rul, cycles, c)

    if c.dual_weight > 0 and rs is not None and rm is not None:
        aux = aux + c.dual_weight * (
            F.mse_loss(rs, seq.mean(1)) + F.mse_loss(rm, summ)
        )

    neg_logp = mse + aux
    # [추가] NaN 체크
    if not torch.isfinite(neg_logp):
        print("[NaN Detected] in neg_logp_particle: MSE={mse}, AUX={aux}")
    return torch.nan_to_num(
        neg_logp, nan=NEG_LOGP_CLIP, posinf=NEG_LOGP_CLIP, neginf=NEG_LOGP_CLIP
    )


def compute_neg_logp(model, vecizer: ParamVectorizer, flat_thetas: List[torch.Tensor],
                     seq: torch.Tensor, summ: torch.Tensor, y: torch.Tensor,
                     cycles: Optional[torch.Tensor], c: Config
                     ) -> Tuple[torch.Tensor, ...]:  # [수정] mse, calib, aux 반환 추가
    preds, rec_seq_list, rec_sum_list = [], [], []

    for theta in flat_thetas:
        params = vecizer.vector_to_params(theta)
        out, rs, rm, _ = functional_call(model, params, (seq, summ))
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        preds.append(out.squeeze(-1))
        if rs is not None:
            rec_seq_list.append(rs)
        if rm is not None:
            rec_sum_list.append(rm)

    pred_stack = torch.stack(preds, dim=0)
    mean_pred = torch.nan_to_num(pred_stack.mean(dim=0), nan=0.0, posinf=0.0, neginf=0.0)
    var_pred = torch.nan_to_num(
        pred_stack.var(dim=0, unbiased=False) + 1e-8,
        nan=1.0, posinf=1.0, neginf=1.0
    )

    mse = F.mse_loss(mean_pred, y)  # [추가] mse 분리

    calib = torch.tensor(0.0, device=mean_pred.device)
    if c.cal_weight > 0:
        se = torch.nan_to_num((mean_pred - y) ** 2,
                              nan=0.0, posinf=NEG_LOGP_CLIP, neginf=NEG_LOGP_CLIP)
        calib = F.mse_loss(var_pred, se, reduction='mean') + 1e-5  # [추가] calib 분리

    aux = torch.tensor(0.0, device=mean_pred.device)
    if c.physics_weight > 0 and cycles is not None:
        aux = aux + physics_loss(mean_pred, cycles, c)

    if c.dual_weight > 0 and rec_seq_list:
        rseq = torch.stack(rec_seq_list).mean(dim=0)
        rsum = torch.stack(rec_sum_list).mean(dim=0)
        aux = aux + c.dual_weight * (
            F.mse_loss(rseq, seq.mean(1)) + F.mse_loss(rsum, summ)
        )  # [추가] aux 분리

    neg_logp = torch.nan_to_num(
        mse + c.cal_weight * calib + aux,
        nan=NEG_LOGP_CLIP, posinf=NEG_LOGP_CLIP, neginf=NEG_LOGP_CLIP
    )
    return neg_logp, -neg_logp, mean_pred, var_pred, mse, calib, aux  # [수정] mse, calib, aux 추가


def rbf_bandwidth(thetas_mat: torch.Tensor) -> float:
    n = thetas_mat.size(0)
    if n <= 1:
        return 1.0
    dists = torch.pdist(thetas_mat)
    med = torch.median(dists)
    h = med ** 2 / max(math.log(n), 1e-2)
    return max(h.item(), 1e-6)

def svgd_update(model, vecizer: ParamVectorizer, flat_thetas: List[torch.Tensor],
                seq: torch.Tensor, summ: torch.Tensor, y: torch.Tensor,
                cycles: Optional[torch.Tensor], c: Config, beta: float
                ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    P = len(flat_thetas)

    # meta_thetas → theta 그래프는 유지하되, grad 계산 가능하게만 설정
    flat_thetas = [th.requires_grad_(True) for th in flat_thetas]

    # neg_logp 안전하게 계산
    neg_logps_list = []
    for th in flat_thetas:
        nlp = neg_logp_particle(model, vecizer, th, seq, summ, y, cycles, c)
        if nlp is None or (not torch.isfinite(nlp)):
            # 완전 망가진 경우에는 0으로 대체해서 터지지 않게
            nlp = torch.tensor(0.0, device=seq.device)
        neg_logps_list.append(nlp)

    neg_logps = torch.stack(neg_logps_list)
    logps = -neg_logps
    # NaN / Inf 정리
    logps = torch.nan_to_num(logps, nan=0.0, posinf=0.0, neginf=0.0)

    if not torch.isfinite(logps).all():
        # 그래프 터지는 것만 피하고 싶으면 여기서 그냥 평균값 리턴
        return flat_thetas, torch.nan_to_num(logps.mean(), nan=0.0, posinf=0.0, neginf=0.0)

    # ----- 여기부터가 핵심 수정: create_graph=False 로 2차 그래프 막기 -----
    grads_raw = torch.autograd.grad(
        outputs=logps.sum(),      # logps[i]를 하나씩 돌리는 대신 한 번에
        inputs=flat_thetas,
        create_graph=False,       # 2차 미분 그래프 안 만듦 → 메모리 절약
        allow_unused=True,
    )

    grads: List[torch.Tensor] = []
    for g, theta in zip(grads_raw, flat_thetas):
        if g is None:
            g = torch.zeros_like(theta)
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        grads.append(g)

    # NaN 방어한 theta / grad 행렬
    thetas_mat = torch.stack([
        torch.nan_to_num(th, nan=0.0, posinf=0.0, neginf=0.0)
        for th in flat_thetas
    ])
    grads_mat = torch.stack(grads)

    # RBF bandwidth 계산 (NaN 방어)
    h = rbf_bandwidth(thetas_mat)
    if not math.isfinite(h) or h <= 0.0:
        h = 1.0

    diff = thetas_mat.unsqueeze(1) - thetas_mat.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=2)

    K = torch.exp(-dist_sq / h)
    K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)

    phi1 = (K @ grads_mat) / P
    grad_K = (-2.0 / h) * (K.unsqueeze(2) * diff).sum(dim=1) / P

    phi = phi1 + grad_K
    phi = torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)

    phi = torch.clamp(phi, -SVGD_CLIP, SVGD_CLIP)

    beta = min(beta, getattr(c, "inner_lr_max", INNER_LR_MAX))
    new_thetas = [
        theta + beta * phi[i]
        for i, theta in enumerate(flat_thetas)
    ]

    return new_thetas, logps.mean()


def run_svgd_support(model, vecizer: ParamVectorizer, flat_thetas: List[torch.Tensor],
                     data_S: dict, data_Q: dict, c: Config,
                     max_steps: int, beta: float
                     ) -> Tuple[List[torch.Tensor], int, Optional[torch.Tensor]]:
    stop_tol = float(getattr(c, "stop_tol", DEFAULT_STOP_TOL))
    beta = min(beta, getattr(c, "inner_lr_max", INNER_LR_MAX))
    beta_decay = 0.95  # [수정] 기존 0.9 → 0.95로 느리게 decay (inner loop 안정화)

    neg_val_prev = None
    steps_done = 0

    for _ in range(max_steps):
        flat_thetas, _ = svgd_update(model, vecizer, flat_thetas, **data_S, c=c, beta=beta)
        steps_done += 1

        with torch.no_grad():
            neg_val_det, _, _, _, _, _, _ = compute_neg_logp(model, vecizer, flat_thetas, **data_Q, c=c)

        if (neg_val_det is None) or (not torch.isfinite(neg_val_det)):
            break

        if neg_val_prev is not None and neg_val_det >= neg_val_prev - stop_tol:
            beta *= beta_decay
            if beta < 1e-6:
                break

        neg_val_prev = neg_val_det

    neg_q_n, _, _, _, _, _, _ = compute_neg_logp(model, vecizer, flat_thetas, **data_Q, c=c)
    
    if (neg_q_n is None) or (not torch.isfinite(neg_q_n)):
        return flat_thetas, steps_done, None

    return flat_thetas, steps_done, neg_q_n

def bmaml_inner(model, vecizer: ParamVectorizer, theta0: List[torch.Tensor],
                task: dict, c: Config,
                detach_theta0: bool = True,
                return_losses: bool = False
                ) -> Tuple[
                    Optional[List[torch.Tensor]],
                    Optional[List[torch.Tensor]],
                    bool,
                    Optional[torch.Tensor],
                    Optional[torch.Tensor]
                ]:
    beta = min(float(c.inner_lr), getattr(c, "inner_lr_max", INNER_LR_MAX))
    stop_tol = float(getattr(c, "stop_tol", DEFAULT_STOP_TOL))

    if detach_theta0:
        theta0 = make_leaf_thetas(theta0)
    else:
        theta0 = [t.requires_grad_(True) for t in theta0]

    s_seq = torch.nan_to_num(task["s_seq"].to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)
    s_sum = torch.nan_to_num(task["s_sum"].to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)
    s_rul = torch.nan_to_num(task["s_rul"].to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)

    q_seq = torch.nan_to_num(task["q_seq"].to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)
    q_sum = torch.nan_to_num(task["q_sum"].to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)
    q_rul = torch.nan_to_num(task["q_rul"].to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)

    data_S = dict(seq=s_seq, summ=s_sum, y=s_rul, cycles=task.get('cycles'))  # [수정] cycles 추가
    data_Q = dict(seq=q_seq, summ=q_sum, y=q_rul, cycles=task.get('cycles'))  # [수정] cycles 추가

    theta_prime, n_steps, neg_q_n = run_svgd_support(
        model, vecizer, theta0, data_S, data_Q, c,
        max_steps=int(c.adaptation_steps), beta=beta
    )

    if neg_q_n is None or (not torch.isfinite(neg_q_n)):
        return None, None, False, None, None

    data_union = dict(
        seq=torch.cat([s_seq, q_seq]),
        summ=torch.cat([s_sum, q_sum]),
        y=torch.cat([s_rul, q_rul]),
        cycles=task.get('cycles')  # [수정] cycles 추가 (union에도)
    )

    theta_second = theta_prime
    neg_union_prev = None
    neg_union = None

    for _ in range(n_steps):
        theta_second, _ = svgd_update(model, vecizer, theta_second, **data_union, c=c, beta=beta)
        with torch.no_grad():
            neg_union, _, _, _, _, _, _ = compute_neg_logp(model, vecizer, theta_second, **data_union, c=c)

        if (neg_union is None) or (not torch.isfinite(neg_union)):
            break
        if neg_union_prev is not None and neg_union >= neg_union_prev - stop_tol:
            break
        neg_union_prev = neg_union

    if (neg_union is None) or (not torch.isfinite(neg_union)):
        return None, None, False, None, None

    # [수정] 초기 학습 방해하는 필터링 조건 제거 (주석 처리)
    # neg_q_n_det = float(neg_q_n.detach())
    # if neg_union >= neg_q_n_det + 10 * stop_tol:
    #     return None, None, False, None, None

    theta_second = [t + 1e-4 * torch.randn_like(t) for t in theta_second]  # [수정] 기존 1e-3 → 1e-4로 noise 줄임
    effective = (
        all(torch.isfinite(t).all() for t in theta_prime) and
        all(torch.isfinite(t).all() for t in theta_second)
    )

    if return_losses:
        return theta_prime, theta_second, effective, neg_q_n, neg_union
    return theta_prime, theta_second, effective, None, None