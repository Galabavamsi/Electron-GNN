import torch
import torch.nn.functional as F

from utils.physics_verifiers import lorentzian_spectrum, peak_set_verifier_scores


def _second_derivative_penalty(spec):
    if spec.numel() < 3:
        return spec.new_tensor(0.0)
    d2 = spec[2:] - 2.0 * spec[1:-1] + spec[:-2]
    return (d2**2).mean()


def refine_peak_set(
    freqs,
    amps,
    omega_grid=None,
    gamma=0.015,
    steps=200,
    lr=0.05,
    lambda_anchor=1.0,
    lambda_sum=1.0,
    lambda_pos=0.2,
    lambda_smooth=0.05,
    frequency_max=5.0,
):
    """
    Constrained projection of a decoded peak set onto a physically safer manifold.

    The objective keeps the refined spectrum close to the raw prediction while enforcing:
    - non-negative amplitudes
    - area consistency (sum-rule proxy)
    - spectral smoothness
    """
    if omega_grid is None:
        omega_grid = torch.linspace(0.01, frequency_max, 512)

    if not torch.is_tensor(freqs):
        freqs = torch.as_tensor(freqs, dtype=torch.float32)
    if not torch.is_tensor(amps):
        amps = torch.as_tensor(amps, dtype=torch.float32)

    if freqs.numel() == 0:
        return {
            "refined_freq": freqs.detach().cpu().numpy(),
            "refined_amp": amps.detach().cpu().numpy(),
            "raw_scores": peak_set_verifier_scores(freqs, amps, omega_grid=omega_grid, gamma=gamma),
            "refined_scores": peak_set_verifier_scores(freqs, amps, omega_grid=omega_grid, gamma=gamma),
            "final_loss": 0.0,
        }

    device = freqs.device
    omega_grid = omega_grid.to(device)

    raw_freqs = freqs.clone()
    raw_amps = torch.clamp(amps.clone(), min=1e-10)
    raw_spec = lorentzian_spectrum(raw_freqs, raw_amps, omega_grid, gamma=gamma).detach()
    raw_area = torch.trapz(raw_spec, omega_grid).detach()

    freq_init = torch.clamp(raw_freqs / frequency_max, min=1e-4, max=1.0 - 1e-4)
    freq_param = torch.nn.Parameter(torch.log(freq_init / (1.0 - freq_init)))
    amp_param = torch.nn.Parameter(torch.log(raw_amps + 1e-10))

    optimizer = torch.optim.Adam([freq_param, amp_param], lr=lr)

    final_loss = 0.0
    for _ in range(steps):
        optimizer.zero_grad()

        ref_freq = torch.sigmoid(freq_param) * frequency_max
        ref_amp = F.softplus(amp_param)
        ref_spec = lorentzian_spectrum(ref_freq, ref_amp, omega_grid, gamma=gamma)

        anchor_loss = F.mse_loss(ref_spec, raw_spec)
        area_loss = F.smooth_l1_loss(torch.trapz(ref_spec, omega_grid), raw_area, beta=0.01)
        pos_loss = torch.relu(-ref_spec).mean()
        smooth_loss = _second_derivative_penalty(ref_spec)

        loss = (
            lambda_anchor * anchor_loss
            + lambda_sum * area_loss
            + lambda_pos * pos_loss
            + lambda_smooth * smooth_loss
        )
        loss.backward()
        optimizer.step()

        final_loss = float(loss.detach().cpu().item())

    with torch.no_grad():
        refined_freq = (torch.sigmoid(freq_param) * frequency_max).detach()
        refined_amp = F.softplus(amp_param).detach()
        order = torch.argsort(refined_freq)
        refined_freq = refined_freq[order]
        refined_amp = refined_amp[order]

    raw_scores = peak_set_verifier_scores(raw_freqs, raw_amps, omega_grid=omega_grid, gamma=gamma, expected_area=raw_area)
    refined_scores = peak_set_verifier_scores(
        refined_freq,
        refined_amp,
        omega_grid=omega_grid,
        gamma=gamma,
        expected_area=raw_area,
    )

    return {
        "refined_freq": refined_freq.cpu().numpy(),
        "refined_amp": refined_amp.cpu().numpy(),
        "raw_scores": raw_scores,
        "refined_scores": refined_scores,
        "final_loss": final_loss,
    }


def maybe_refine_peak_set(freqs, amps, trust_score, threshold=0.5, **refiner_kwargs):
    """Apply constrained refinement only for high-risk (low-trust) outputs."""
    if trust_score <= threshold:
        return {
            "refined_freq": freqs,
            "refined_amp": amps,
            "did_refine": False,
        }

    result = refine_peak_set(freqs, amps, **refiner_kwargs)
    result["did_refine"] = True
    return result
