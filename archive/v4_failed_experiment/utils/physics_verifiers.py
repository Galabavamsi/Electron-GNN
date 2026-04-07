import torch


def split_targets_by_graph(data_batch, batch_size, device):
    """Split flattened variable-length targets using per-graph counts."""
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


def lorentzian_spectrum(freqs, amps, omega_grid, gamma=0.015):
    """Differentiable Lorentzian spectral reconstruction on a fixed frequency grid."""
    if freqs.numel() == 0:
        return torch.zeros_like(omega_grid)

    denom = (omega_grid.unsqueeze(0) - freqs.unsqueeze(1)) ** 2 + gamma**2
    peaks = amps.unsqueeze(1) * (gamma / denom)
    return peaks.sum(dim=0)


def _hilbert_transform(signal):
    """Hilbert transform computed via FFT analytic signal construction."""
    n = signal.shape[-1]
    fft_signal = torch.fft.fft(signal, dim=-1)

    h = torch.zeros(n, dtype=signal.dtype, device=signal.device)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0

    analytic = torch.fft.ifft(fft_signal * h, dim=-1)
    return analytic.imag


def kk_self_consistency_score(spec_batch, eps=1e-8):
    """
    Kramers-Kronig proxy: Hilbert(Hilbert(x)) ~= -x for centered finite signals.
    Lower is better.
    """
    centered = spec_batch - spec_batch.mean(dim=-1, keepdim=True)
    h1 = _hilbert_transform(centered)
    h2 = _hilbert_transform(h1)

    numer = torch.linalg.norm(h2 + centered, dim=-1)
    denom = torch.linalg.norm(centered, dim=-1) + eps
    return numer / denom


def positivity_score(spec_batch):
    """Average negative-area violation. Lower is better, ideal is zero."""
    return torch.relu(-spec_batch).mean(dim=-1)


def integrated_strength_score(spec_pred, spec_true, omega_grid, eps=1e-8):
    """Relative violation of integrated spectral area vs reference/target spectrum."""
    area_pred = torch.trapz(spec_pred, omega_grid, dim=-1)
    area_true = torch.trapz(spec_true, omega_grid, dim=-1)
    return torch.abs(area_pred - area_true) / (torch.abs(area_true) + eps)


def smoothness_score(spec_batch):
    """Second-derivative roughness penalty as a stability/physical plausibility proxy."""
    if spec_batch.shape[-1] < 3:
        return torch.zeros(spec_batch.shape[0], device=spec_batch.device, dtype=spec_batch.dtype)
    d2 = spec_batch[:, 2:] - 2.0 * spec_batch[:, 1:-1] + spec_batch[:, :-2]
    return (d2**2).mean(dim=-1)


def stability_score_from_pred_dict(pred_dict):
    """
    Pole stability violation score.
    If no pole parameters are present, defaults to zero.
    """
    freq = pred_dict["freq"]
    batch_size = freq.shape[0]

    poles_real = pred_dict.get("poles_real")
    if poles_real is None:
        return torch.zeros(batch_size, device=freq.device, dtype=freq.dtype)

    return (poles_real >= 0.0).to(freq.dtype).mean(dim=-1)


def build_batch_spectra(pred_dict, data_batch, omega_points=512, omega_min=0.01, omega_max=5.0, gamma=0.015):
    """Build predicted and true spectra for each graph in a mini-batch."""
    device = pred_dict["freq"].device
    batch_size = pred_dict["freq"].shape[0]
    targets = split_targets_by_graph(data_batch, batch_size, device)

    omega = torch.linspace(omega_min, omega_max, omega_points, device=device)
    pred_specs = []
    true_specs = []

    for batch_idx in range(batch_size):
        w_p = pred_dict["freq"][batch_idx]
        b_p = pred_dict["amp"][batch_idx] * pred_dict["prob"][batch_idx]
        w_t, b_t = targets[batch_idx]

        pred_specs.append(lorentzian_spectrum(w_p, b_p, omega, gamma=gamma))
        true_specs.append(lorentzian_spectrum(w_t, b_t, omega, gamma=gamma))

    return omega, torch.stack(pred_specs, dim=0), torch.stack(true_specs, dim=0)


def physics_verifier_scores(pred_dict, data_batch, omega_points=512, omega_min=0.01, omega_max=5.0, gamma=0.015):
    """Compute batch-level physical verifier scores from predictions and targets."""
    omega, spec_pred, spec_true = build_batch_spectra(
        pred_dict,
        data_batch,
        omega_points=omega_points,
        omega_min=omega_min,
        omega_max=omega_max,
        gamma=gamma,
    )

    s_kk = kk_self_consistency_score(spec_pred)
    s_pos = positivity_score(spec_pred)
    s_sum = integrated_strength_score(spec_pred, spec_true, omega)
    s_smooth = smoothness_score(spec_pred)
    s_stab = stability_score_from_pred_dict(pred_dict)

    return {
        "omega": omega,
        "spec_pred": spec_pred,
        "spec_true": spec_true,
        "s_kk": s_kk,
        "s_pos": s_pos,
        "s_sum": s_sum,
        "s_smooth": s_smooth,
        "s_stab": s_stab,
        "s_kk_mean": s_kk.mean(),
        "s_pos_mean": s_pos.mean(),
        "s_sum_mean": s_sum.mean(),
        "s_smooth_mean": s_smooth.mean(),
        "s_stab_mean": s_stab.mean(),
    }


def composite_trust_score(scores, weights=None):
    """Aggregate verifier components into a scalar trust score per sample and mean."""
    if weights is None:
        weights = {
            "kk": 1.0,
            "pos": 1.0,
            "sum": 1.0,
            "smooth": 0.25,
            "stab": 1.0,
        }

    trust = (
        weights["kk"] * scores["s_kk"]
        + weights["pos"] * scores["s_pos"]
        + weights["sum"] * scores["s_sum"]
        + weights["smooth"] * scores["s_smooth"]
        + weights["stab"] * scores["s_stab"]
    )
    return trust, trust.mean()


def peak_set_verifier_scores(freqs, amps, omega_grid=None, gamma=0.015, expected_area=None):
    """
    Compute verifier scores for a single decoded peak set.
    Returns plain floats for easy logging/reporting.
    """
    if omega_grid is None:
        omega_grid = torch.linspace(0.01, 5.0, 512)

    if not torch.is_tensor(freqs):
        freqs = torch.as_tensor(freqs, dtype=torch.float32)
    if not torch.is_tensor(amps):
        amps = torch.as_tensor(amps, dtype=torch.float32)

    omega_grid = omega_grid.to(freqs.device)
    freqs = freqs.to(omega_grid.device)
    amps = amps.to(omega_grid.device)

    spec = lorentzian_spectrum(freqs, amps, omega_grid, gamma=gamma).unsqueeze(0)

    s_kk = kk_self_consistency_score(spec)[0]
    s_pos = positivity_score(spec)[0]
    s_smooth = smoothness_score(spec)[0]

    if expected_area is None:
        s_sum = torch.tensor(0.0, device=spec.device)
    else:
        area = torch.trapz(spec.squeeze(0), omega_grid)
        s_sum = torch.abs(area - expected_area) / (torch.abs(expected_area) + 1e-8)

    return {
        "s_kk": float(s_kk.detach().cpu().item()),
        "s_pos": float(s_pos.detach().cpu().item()),
        "s_sum": float(s_sum.detach().cpu().item()),
        "s_smooth": float(s_smooth.detach().cpu().item()),
    }
