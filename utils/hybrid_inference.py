import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _extract_common_outputs(pred_dict, batch_idx=0):
    """Extract common prediction arrays from both legacy and current model outputs."""
    prob_key = "prob" if "prob" in pred_dict else "peak_probs"
    freq_key = "freq" if "freq" in pred_dict else "frequencies"
    amp_key = "amp" if "amp" in pred_dict else "amplitudes"

    probs_t = pred_dict[prob_key]
    freqs_t = pred_dict[freq_key]
    amps_t = pred_dict[amp_key]

    if probs_t.ndim == 2:
        probs_t = probs_t[batch_idx]
    if freqs_t.ndim == 2:
        freqs_t = freqs_t[batch_idx]
    if amps_t.ndim >= 2 and amps_t.shape[0] == probs_t.shape[0] and amps_t.ndim > 1:
        # For per-slot vector amplitudes, keep slot axis and reduce later.
        pass
    elif amps_t.ndim >= 2 and amps_t.shape[0] != probs_t.shape[0]:
        amps_t = amps_t[batch_idx]

    if prob_key == "peak_probs":
        probs_t = torch.sigmoid(probs_t)

    probs = probs_t.detach().cpu().numpy()
    freqs = freqs_t.detach().cpu().numpy()
    amps_np = amps_t.detach().cpu().numpy()

    if amps_np.ndim == 2:
        amps = np.linalg.norm(amps_np, axis=1)
    else:
        amps = np.abs(amps_np)

    count_val = None
    if "count" in pred_dict:
        count_t = pred_dict["count"]
        if count_t.ndim == 1:
            count_val = float(count_t[batch_idx].detach().cpu().item())
        else:
            count_val = float(count_t.detach().cpu().item())

    return {
        "probs": probs,
        "freqs": freqs,
        "amps": amps,
        "count": count_val,
    }


def decode_peak_set(pred_dict, batch_idx=0, prob_threshold=0.65, fallback_top_k=5, use_count_head=True):
    """Decode a variable-length peak set from fixed-size slot predictions."""
    out = _extract_common_outputs(pred_dict, batch_idx=batch_idx)
    probs = out["probs"]
    freqs = out["freqs"]
    amps = out["amps"]
    count_val = out["count"]

    if use_count_head and count_val is not None:
        top_k = int(np.clip(np.rint(count_val), 1, probs.shape[0]))
        idx = np.argsort(probs)[-top_k:]
    else:
        mask = probs > prob_threshold
        if np.count_nonzero(mask) == 0:
            top_k = min(fallback_top_k, probs.shape[0])
            idx = np.argsort(probs)[-top_k:]
        else:
            idx = np.where(mask)[0]

    idx = idx[np.argsort(probs[idx])[::-1]]
    return {
        "freq": freqs[idx],
        "amp": amps[idx],
        "prob": probs[idx],
        "indices": idx,
        "all_freq": freqs,
        "all_amp": amps,
        "all_prob": probs,
        "count": count_val,
    }


def combine_two_tower_predictions(
    freq_pred_dict,
    amp_pred_dict,
    batch_idx=0,
    prob_threshold=0.65,
    fallback_top_k=5,
    amp_conf_penalty=0.05,
    allow_amp_overflow=True,
    min_freq_separation=0.005,
):
    """
        Hybrid decode:
        - primary frequencies come from the frequency tower
        - cardinality prefers amplitude tower count head when available
        - optional overflow frequencies are borrowed from amplitude tower when desired
            count exceeds available frequency slots
        - amplitudes come from amplitude tower matched by nearest predicted frequencies
    """
    freq_out = _extract_common_outputs(freq_pred_dict, batch_idx=batch_idx)
    amp_out = _extract_common_outputs(amp_pred_dict, batch_idx=batch_idx)

    probs_f = freq_out["probs"]
    freqs_f = freq_out["freqs"]

    target_count = amp_out.get("count")
    if target_count is None:
        target_count = freq_out.get("count")

    desired_count = None
    if target_count is not None:
        desired_count = int(max(1, np.rint(target_count)))
        top_k = int(np.clip(desired_count, 1, probs_f.shape[0]))
        freq_idx = np.argsort(probs_f)[-top_k:]
    else:
        mask = probs_f > prob_threshold
        if np.count_nonzero(mask) == 0:
            top_k = min(fallback_top_k, probs_f.shape[0])
            freq_idx = np.argsort(probs_f)[-top_k:]
        else:
            freq_idx = np.where(mask)[0]

    freq_idx = freq_idx[np.argsort(probs_f[freq_idx])[::-1]]

    pred_w = freqs_f[freq_idx].copy()
    pred_p_freq = probs_f[freq_idx].copy()
    amp_w_all = amp_out["freqs"]
    amp_b_all = amp_out["amps"]
    amp_p_all = amp_out["probs"]

    if (
        allow_amp_overflow
        and desired_count is not None
        and desired_count > pred_w.shape[0]
        and amp_w_all.size > 0
    ):
        extra_needed = int(desired_count - pred_w.shape[0])
        taken_w = pred_w.copy()
        extra_w = []
        extra_p = []

        for idx in np.argsort(amp_p_all)[::-1]:
            cand_w = float(amp_w_all[idx])
            if taken_w.size > 0 and np.min(np.abs(taken_w - cand_w)) < float(min_freq_separation):
                continue

            extra_w.append(cand_w)
            extra_p.append(float(amp_p_all[idx]))
            taken_w = np.append(taken_w, cand_w)

            if len(extra_w) >= extra_needed:
                break

        if extra_w:
            pred_w = np.concatenate([pred_w, np.asarray(extra_w, dtype=pred_w.dtype)])
            pred_p_freq = np.concatenate([pred_p_freq, np.asarray(extra_p, dtype=pred_p_freq.dtype)])

    if pred_w.size == 0:
        return pred_w, np.array([], dtype=float), np.array([], dtype=float)

    if amp_w_all.size == 0:
        return pred_w, np.zeros_like(pred_w), pred_p_freq

    # One-to-one assignment with a small confidence-aware penalty.
    cost = np.abs(pred_w[:, None] - amp_w_all[None, :]) + amp_conf_penalty * (1.0 - amp_p_all[None, :])
    row_idx, col_idx = linear_sum_assignment(cost)

    pred_b = np.zeros(pred_w.shape[0], dtype=float)
    amp_conf = np.zeros(pred_w.shape[0], dtype=float)
    pred_b[row_idx] = amp_b_all[col_idx]
    amp_conf[row_idx] = amp_p_all[col_idx]

    if len(row_idx) < pred_w.shape[0]:
        unmatched = np.setdiff1d(np.arange(pred_w.shape[0]), row_idx)
        nearest = np.argmin(cost[unmatched], axis=1)
        pred_b[unmatched] = amp_b_all[nearest]
        amp_conf[unmatched] = amp_p_all[nearest]

    hybrid_prob = np.minimum(pred_p_freq, amp_conf)

    order = np.argsort(pred_w)
    return pred_w[order], pred_b[order], hybrid_prob[order]
