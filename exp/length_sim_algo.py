import numpy as np

def estimate_peak_memory_fast(prefill_tokens, current_decode_tokens, decode_bin_edges, decode_bin_probs, n_simulations=100):
    prefill_tokens = np.array(prefill_tokens, dtype=float)
    current_decode_tokens = np.array(current_decode_tokens, dtype=float)
    decode_bin_edges = np.array(decode_bin_edges, dtype=float)
    
    # 1. Setup CDF
    cdf_values = np.concatenate(([0.0], np.cumsum(decode_bin_probs)))
    n_requests = len(prefill_tokens)
    if n_requests == 0:
        return 0.0

    current_tokens = prefill_tokens + current_decode_tokens

    # 2. Conditional Sampling (The Fix)
    # We need to calculate F(current_decode) for each request to know where we are in the CDF
    current_cdfs = np.interp(current_decode_tokens, decode_bin_edges, cdf_values)
    
    # Sample u uniformly from [F(current), 1.0]
    u_random = np.random.random((n_simulations, n_requests))
    target_cdfs = current_cdfs + u_random * (1.0 - current_cdfs)
    
    # Invert CDF to get Total Lengths
    decode_total_samples = np.interp(
        target_cdfs.ravel(),
        cdf_values,
        decode_bin_edges
    ).reshape(n_simulations, n_requests)
    
    # Ensure numerical stability (Total >= Current)
    decode_total_samples = np.maximum(decode_total_samples, current_decode_tokens)
    remaining_decode = decode_total_samples - current_decode_tokens

    # 3. Vectorized Peak Calculation
    sort_indices = np.argsort(remaining_decode, axis=1)
    R_sorted = np.take_along_axis(remaining_decode, sort_indices, axis=1)
    
    # Expand current_tokens to match shape before sorting
    C_matrix = np.broadcast_to(current_tokens, (n_simulations, n_requests))
    C_sorted = np.take_along_axis(C_matrix, sort_indices, axis=1)

    active_counts = np.arange(n_requests, 0, -1)
    total_C_sum = np.sum(current_tokens)
    
    # Optimization: Use in-place subtraction instead of column_stack/concatenation
    cum_finished = np.cumsum(C_sorted, axis=1)
    sum_active_C = np.empty_like(cum_finished)
    sum_active_C[:, 0] = total_C_sum
    sum_active_C[:, 1:] = total_C_sum - cum_finished[:, :-1]

    # M(t) = Sum_Active_Starts + (Count * t)
    memory_matrix = sum_active_C + (active_counts * R_sorted)
    
    # Peak is the max over time (axis 1)
    peak_per_sim = np.max(memory_matrix, axis=1)
    
    # Edge Case: Peak might be NOW (t=0). 
    # If all requests decrease in memory (unlikely in this model, but possible in others),
    # we must clamp to current usage.
    peak_per_sim = np.maximum(peak_per_sim, total_C_sum)

    return float(np.mean(peak_per_sim))

# --- Test ---
if __name__ == "__main__":
    prefill = [50, 150, 400]
    current_decode = [10, 20, 30]
    decode_edges = [0, 100, 500, 1000, 2000] 
    decode_probs = [0.2, 0.5, 0.2, 0.1]
    print(f"Fast Estimate: {estimate_peak_memory_fast(prefill, current_decode, decode_edges, decode_probs):.2f}")
