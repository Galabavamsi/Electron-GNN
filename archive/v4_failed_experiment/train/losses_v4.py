from train.losses import bipartite_matching_loss, auto_differential_spectrum_loss
from utils.physics_verifiers import physics_verifier_scores, composite_trust_score


def v4_loss(
    pred_dict,
    data_batch,
    lambda_spectrum=0.3,
    lambda_kk=0.05,
    lambda_pos=0.05,
    lambda_sum=0.2,
    lambda_smooth=0.01,
    lambda_stab=0.05,
    verifier_weights=None,
):
    """
    V4 training objective:
    data fit + spectrum consistency + physics verifier penalties.
    """
    loss_bip, loss_prob, loss_reg = bipartite_matching_loss(pred_dict, data_batch)
    loss_spectrum = auto_differential_spectrum_loss(pred_dict, data_batch)

    scores = physics_verifier_scores(pred_dict, data_batch)
    trust_vector, trust_mean = composite_trust_score(scores, weights=verifier_weights)

    loss_physics = (
        lambda_kk * scores["s_kk_mean"]
        + lambda_pos * scores["s_pos_mean"]
        + lambda_sum * scores["s_sum_mean"]
        + lambda_smooth * scores["s_smooth_mean"]
        + lambda_stab * scores["s_stab_mean"]
    )

    total = loss_bip + lambda_spectrum * loss_spectrum + loss_physics

    metrics = {
        "total": total,
        "loss_bip": loss_bip,
        "loss_prob": loss_prob,
        "loss_reg": loss_reg,
        "loss_spectrum": loss_spectrum,
        "loss_physics": loss_physics,
        "s_kk": scores["s_kk_mean"],
        "s_pos": scores["s_pos_mean"],
        "s_sum": scores["s_sum_mean"],
        "s_smooth": scores["s_smooth_mean"],
        "s_stab": scores["s_stab_mean"],
        "trust_mean": trust_mean,
        "trust_max": trust_vector.max(),
    }
    return total, metrics


def v4_eval_metrics(pred_dict, data_batch, verifier_weights=None):
    """Compute V4 verifier diagnostics without adding training penalties."""
    scores = physics_verifier_scores(pred_dict, data_batch)
    trust_vector, trust_mean = composite_trust_score(scores, weights=verifier_weights)

    return {
        "s_kk": scores["s_kk_mean"],
        "s_pos": scores["s_pos_mean"],
        "s_sum": scores["s_sum_mean"],
        "s_smooth": scores["s_smooth_mean"],
        "s_stab": scores["s_stab_mean"],
        "trust_mean": trust_mean,
        "trust_max": trust_vector.max(),
    }
