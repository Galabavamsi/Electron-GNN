import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def bipartite_matching_loss(pred_dict, data_batch):
    """
    Computes bipartite (Hungarian) matching loss between predicted set (Fixed Size 50)
    and target set (Variable Size per graph batch member) for Bohr frequencies and amplitudes.
    """
    p_pred = pred_dict["prob"]  # (Batch, K_max)
    w_pred = pred_dict["freq"]  # (Batch, K_max)
    b_pred = pred_dict["amp"]   # (Batch, K_max)
    
    # 1. Slice apart the flattened ground truths per batch index using ptr/slices
    batch_size = p_pred.shape[0]
    K_max = p_pred.shape[1]
    
    total_loss = 0.0
    total_prob_loss = 0.0
    total_mse_loss = 0.0
    
    for batch_idx in range(batch_size):
        w_t = data_batch.y_freq
        b_t = data_batch.y_amp
        num_true = len(w_t) if len(w_t.shape) > 0 else 1
        if len(w_t.shape) == 0:
            w_t = w_t.unsqueeze(0)
            b_t = b_t.unsqueeze(0)
        
        w_p = w_pred[batch_idx]
        b_p = b_pred[batch_idx]
        p_p = p_pred[batch_idx]
        
        # CPU conversion for SciPy
        w_t_np = w_t.detach().cpu().numpy()
        b_t_np = b_t.detach().cpu().numpy()
        w_p_np = w_p.detach().cpu().numpy()
        b_p_np = b_p.detach().cpu().numpy()
        
        # 2. Build Bipartite Cost Matrix mapping predicted to true
        cost_matrix = torch.zeros((K_max, num_true))
        for i in range(K_max):
            for j in range(num_true):
                # Heavy penalty for bad frequency matches, soft penalty for amplitude
                # Minus probability because if it matches, we want prediction probability to be high
                # We use numpy for CPU-bound bipartite linear_sum_assignment
                cost = 10.0 * abs(w_p_np[i] - w_t_np[j]) + abs(b_p_np[i] - b_t_np[j])
                cost_matrix[i, j] = float(cost)
                
        # 3. Hungarian Optimization (Finds lowest total distance mapping without duplicates)
        pred_indices, true_indices = linear_sum_assignment(cost_matrix.numpy())
        
        # Prepare targets
        p_target = torch.zeros_like(p_p) # Target is 0 probability
        
        loss_w = 0.0
        loss_b = 0.0
        
        for k_idx, p_idx in enumerate(pred_indices):
            t_idx = true_indices[k_idx]
            
            # The matched predicted slots ought to exist (probability = 1)
            p_target[p_idx] = 1.0
            
            # Sum up regression MSE exactly across paired items
            loss_w += F.mse_loss(w_p[p_idx], w_t[t_idx])
            loss_b += F.mse_loss(b_p[p_idx], b_t[t_idx])
        
        loss_w = loss_w / max(1, num_true)
        loss_b = loss_b / max(1, num_true)
        
        # Binary Cross Entropy over entire 50-slot probability layer
        loss_prob = F.binary_cross_entropy(p_p, p_target)
        
        bipartite_loss = (10.0 * loss_w) + (2.0 * loss_b) + (1.0 * loss_prob)
        
        total_loss += bipartite_loss
        total_prob_loss += loss_prob
        total_mse_loss += (loss_w + loss_b)

    return total_loss / batch_size, total_prob_loss / batch_size, total_mse_loss / batch_size

def auto_differential_spectrum_loss(pred_dict, data_batch, t_max=400, dt=0.2):
    """
    Physical-constraint regularizer: Compares the continuous time reconstruction 
    of the predicted vs true parameter outputs inside PyTorch (which preserves gradients).
    """
    time = torch.linspace(0, t_max, int(t_max / dt), device=pred_dict["freq"].device)
    
    w_p = pred_dict["freq"][0] * pred_dict["prob"][0] # Silence non-existant ones softly
    b_p = pred_dict["amp"][0] * pred_dict["prob"][0]
    
    # True reconstruction
    w_t = data_batch.y_freq
    b_t = data_batch.y_amp
    
    # Broadcast math: Sum( Amplitude * Sin(Frequency * Time) )
    pred_signal = torch.sum(b_p.unsqueeze(1) * torch.sin(w_p.unsqueeze(1) * time.unsqueeze(0)), dim=0)
    true_signal = torch.sum(b_t.unsqueeze(1) * torch.sin(w_t.unsqueeze(1) * time.unsqueeze(0)), dim=0)
    
    # Physical Trace MSE
    return F.mse_loss(pred_signal, true_signal)

