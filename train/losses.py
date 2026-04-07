import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _split_targets_by_graph(data_batch, batch_size, device):
    """
    Split flattened variable-length targets using explicit per-graph counts.
    """
    if hasattr(data_batch, "num_peaks"):
        counts = data_batch.num_peaks.reshape(-1).tolist()
    else:
        counts = [int(data_batch.y_freq.shape[0])] + [0] * (batch_size - 1)

    targets = []
    offset = 0
    for batch_idx in range(batch_size):
        num_true = int(counts[batch_idx]) if batch_idx < len(counts) else 0
        w_t = data_batch.y_freq[offset : offset + num_true].to(device)
        b_t = data_batch.y_amp[offset : offset + num_true].to(device)
        offset += num_true
        targets.append((w_t, b_t))

    return targets


def _lorentzian_spectrum(freqs, amps, omega_grid, gamma=0.015):
    """
    Differentiable Lorentzian reconstruction S(omega).
    """
    if freqs.numel() == 0:
        return torch.zeros_like(omega_grid)
    denom = (omega_grid.unsqueeze(0) - freqs.unsqueeze(1)) ** 2 + gamma**2
    peaks = amps.unsqueeze(1) * (gamma / denom)
    return peaks.sum(dim=0)

def bipartite_matching_loss(pred_dict, data_batch):
    """
    Computes bipartite (Hungarian) matching loss between predicted set (Fixed Size 50)
    and target set (Variable Size per graph batch member) for Bohr frequencies and amplitudes.
    """
    p_pred = pred_dict["prob"]  # (Batch, K_max)
    w_pred = pred_dict["freq"]  # (Batch, K_max)
    b_pred = pred_dict["amp"]   # (Batch, K_max)
    p_logits = pred_dict.get("prob_logits")
    count_pred = pred_dict.get("count")
    
    batch_size = p_pred.shape[0]
    K_max = p_pred.shape[1]
    device = p_pred.device
    targets = _split_targets_by_graph(data_batch, batch_size, device)
    
    total_loss = p_pred.new_tensor(0.0)
    total_prob_loss = p_pred.new_tensor(0.0)
    total_mse_loss = p_pred.new_tensor(0.0)
    amp_log_scale = 1e4
    
    for batch_idx in range(batch_size):
        w_t, b_t = targets[batch_idx]
        num_true = int(w_t.numel())
        
        w_p = w_pred[batch_idx]
        b_p = b_pred[batch_idx]
        p_p = p_pred[batch_idx]
        p_target = torch.zeros_like(p_p)

        if num_true > 0:
            cost_matrix = (
                10.0 * torch.cdist(w_p.unsqueeze(-1), w_t.unsqueeze(-1), p=1)
                + torch.cdist(b_p.unsqueeze(-1), b_t.unsqueeze(-1), p=1)
            )
            pred_indices, true_indices = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

            pred_idx_t = torch.tensor(pred_indices, dtype=torch.long, device=device)
            true_idx_t = torch.tensor(true_indices, dtype=torch.long, device=device)
            p_target[pred_idx_t] = 1.0

            loss_w = F.smooth_l1_loss(w_p[pred_idx_t], w_t[true_idx_t], beta=0.02)
            loss_b = F.smooth_l1_loss(
                torch.log1p(amp_log_scale * b_p[pred_idx_t]),
                torch.log1p(amp_log_scale * b_t[true_idx_t]),
                beta=0.02,
            )
            loss_sum = F.smooth_l1_loss(b_p[pred_idx_t].sum(), b_t.sum(), beta=0.01)

            unmatched_mask = torch.ones(K_max, dtype=torch.bool, device=device)
            unmatched_mask[pred_idx_t] = False
            loss_unmatched_amp = (b_p[unmatched_mask] ** 2).mean() if unmatched_mask.any() else b_p.new_tensor(0.0)
        else:
            loss_w = w_p.new_tensor(0.0)
            loss_b = b_p.new_tensor(0.0)
            loss_sum = b_p.new_tensor(0.0)
            loss_unmatched_amp = (b_p ** 2).mean()

        if p_logits is not None:
            loss_prob = F.binary_cross_entropy_with_logits(p_logits[batch_idx], p_target)
        else:
            loss_prob = F.binary_cross_entropy(p_p, p_target)

        if count_pred is not None:
            target_count = torch.tensor(float(num_true), device=device)
            loss_count = F.smooth_l1_loss(count_pred[batch_idx], target_count, beta=1.0)
        else:
            loss_count = p_p.new_tensor(0.0)

        bipartite_loss = (
            8.0 * loss_w
            + 8.0 * loss_b
            + 1.2 * loss_prob
            + 1.0 * loss_unmatched_amp
            + 6.0 * loss_sum
            + 0.5 * loss_count
        )
        
        total_loss += bipartite_loss
        total_prob_loss += loss_prob
        total_mse_loss += (loss_w + loss_b)

    return total_loss / batch_size, total_prob_loss / batch_size, total_mse_loss / batch_size

def auto_differential_spectrum_loss(pred_dict, data_batch, t_max=400, dt=0.2):
    """
    Physical regularizer over the full batch:
    - time-domain signal consistency
    - frequency-domain Lorentzian spectrum consistency in log scale
    """
    device = pred_dict["freq"].device
    batch_size = pred_dict["freq"].shape[0]
    time = torch.linspace(0, t_max, int(t_max / dt), device=device)
    targets = _split_targets_by_graph(data_batch, batch_size, device)

    total_loss = pred_dict["freq"].new_tensor(0.0)
    for batch_idx in range(batch_size):
        w_p = pred_dict["freq"][batch_idx]
        b_p = pred_dict["amp"][batch_idx] * pred_dict["prob"][batch_idx]

        w_t, b_t = targets[batch_idx]

        pred_signal = torch.sum(b_p.unsqueeze(1) * torch.sin(w_p.unsqueeze(1) * time.unsqueeze(0)), dim=0)
        if w_t.numel() > 0:
            true_signal = torch.sum(b_t.unsqueeze(1) * torch.sin(w_t.unsqueeze(1) * time.unsqueeze(0)), dim=0)
        else:
            true_signal = torch.zeros_like(pred_signal)

        signal_loss = F.mse_loss(pred_signal, true_signal)

        omega_upper = max(5.0, float(w_t.max().item() * 1.1) if w_t.numel() > 0 else 5.0)
        omega = torch.linspace(0.01, omega_upper, 512, device=device)
        spec_pred = _lorentzian_spectrum(w_p, b_p, omega, gamma=0.015)
        spec_true = _lorentzian_spectrum(w_t, b_t, omega, gamma=0.015)

        spec_loss = F.mse_loss(
            torch.log1p(5e3 * spec_pred),
            torch.log1p(5e3 * spec_true),
        )
        area_loss = F.smooth_l1_loss(spec_pred.sum(), spec_true.sum(), beta=0.01)

        total_loss += signal_loss + 0.5 * spec_loss + 0.5 * area_loss

    return total_loss / batch_size

