import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def reorder_admixture(Q_mat):
    """
    Reorder Q_mat rows so that rows are grouped by each sample's dominant ancestry,
    and columns are sorted by descending average ancestry proportion.
    """
    n_samples, K = Q_mat.shape
    
    # Reorder columns by descending average proportion
    col_means = Q_mat.mean(axis=0)
    col_order = np.argsort(col_means)[::-1]   # largest first
    Q_cols_sorted = Q_mat[:, col_order]
    
    # Group samples by whichever column is argmax
    row_groups = []
    boundary_list = [0]
    for k in range(K):
        rows_k = np.where(np.argmax(Q_cols_sorted, axis=1) == k)[0]
        # Sort these rows by their proportion in col k
        rows_k_sorted = rows_k[np.argsort(Q_cols_sorted[rows_k, k])[::-1]]
        row_groups.append(rows_k_sorted)
        boundary_list.append(boundary_list[-1] + len(rows_k_sorted))
    
    # Combine them into one final row order
    row_order = np.concatenate(row_groups)
    Q_mat_sorted = Q_cols_sorted[row_order, :]
    
    return Q_mat_sorted, row_order, boundary_list, col_order

def plot_admixture(ax, Q_mat_sorted, boundary_list, col_order=None, colors=None):
    """
    Plot a structure-style bar chart of Q_mat_sorted in the given Axes ax.
    If colors is not None, it should be a list or array of length K.
    If col_order is not None, colors are reordered according to col_order.
    """
    n_samples, K = Q_mat_sorted.shape
    
    # If we have a specific color list and a col_order, reorder the colors to match the columns
    if (colors is not None) and (col_order is not None):
        # re-map the user-provided colors to match the new column order
        colors = [colors[idx] for idx in col_order]
    
    # cumulative sum across columns for stacked fill
    Q_cum = np.cumsum(Q_mat_sorted, axis=1)
    x_vals = np.arange(n_samples)
    
    step_mode = "post"
    ax.step(x_vals, Q_cum, linewidth=0.0, where=step_mode)
    
    # fill-between for a stacked bar effect
    for j in range(K):
        c = colors[j] if (colors is not None) else None
        if j == 0:
            ax.fill_between(x_vals, 0, Q_cum[:, j], step=step_mode, color=c)
        else:
            ax.fill_between(x_vals, Q_cum[:, j - 1], Q_cum[:, j], step=step_mode, color=c)
    
    # Vertical lines for group boundaries
    for boundary in boundary_list:
        ax.axvline(boundary, color='black', ls='--', lw=1.0)
    
    ax.set_xlim(0, n_samples - 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Ancestry Proportion")
