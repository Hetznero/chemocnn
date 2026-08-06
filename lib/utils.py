import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, csc_matrix, bmat, issparse
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from numba import njit, prange
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter

def apply_stimulus(radius, coords, center=(0.5, 0.5)):
    dists = np.sqrt((coords[:, 0] - center[0])**2 + (coords[:, 1] - center[1])**2)
    return (dists <= radius).astype(float)


def compute_prob_2d(r, l, sigma_conn):
    differences = r[:, np.newaxis, :] - l[np.newaxis, :, :]
    d = np.linalg.norm(differences, axis=-1)
    max_prob = 1
    unscaled_gaussian = np.exp(-(d**2) / (2 * sigma_conn**2))
    
    return max_prob * unscaled_gaussian

def compute_prob_3d(r, l, sigma_x, sigma_y, shape=None):
    dx = l[np.newaxis, :, 0] - r[:, np.newaxis, 0]
    dy = l[np.newaxis, :, 1] - r[:, np.newaxis, 1]
    
    theta = l[:, np.newaxis, 2] * np.pi
    
    theta = theta.reshape(1, -1)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    dx_rot = dx * cos_theta + dy * sin_theta
    dy_rot = -dx * sin_theta + dy * cos_theta
    
    d_sq = (dx_rot**2 / sigma_x**2) + (dy_rot**2 / sigma_y**2)
    
    unscaled_gaussian = np.exp(-d_sq/2)

    return unscaled_gaussian

def plot_prob_mask(prob_mask, cmap='viridis'):
    max_val = np.max(np.abs(prob_mask))
    plt.figure(figsize=(10, 8))
    
    plt.imshow(prob_mask, cmap=cmap, aspect='auto', 
           interpolation='nearest', vmin=-max_val, vmax=max_val)
    
    plt.colorbar(label='Connection Probability')
    plt.xlabel('Layer i+1 Neurons')
    plt.ylabel('Layer i Neurons')
    plt.title('Probability Mask for Connections')
    plt.tight_layout()
    plt.show()

def sort_prob_mask_2d(prob_mask, layer_sender, layer_receiver):
    r_vals = np.array([neuron.r for neuron in layer_sender])
    l_vals = np.array([neuron.l for neuron in layer_receiver])
    
    idx_r = np.lexsort((r_vals[:, 1], r_vals[:, 0]))[::-1]
    idx_l = np.lexsort((l_vals[:, 1], l_vals[:, 0]))[::-1]
    
    sorted_prob_mask = prob_mask[np.ix_(idx_r, idx_l)]
    
    return sorted_prob_mask, idx_r, idx_l

def unsort_prob_mask_2d(sorted_prob_mask, idx_r, idx_l):

    inv_idx_r = np.argsort(idx_r)
    inv_idx_l = np.argsort(idx_l)

    original_prob_mask = sorted_prob_mask[np.ix_(inv_idx_r, inv_idx_l)]
    
    return original_prob_mask



def plot_spatial_prob_mask(receiver_idx, prob_mask, sender_coords, receiver_coords, title = '', cmap='seismic',vmin=-0.1,vmax=0.1):

    plt.figure(figsize=(10, 8))

    probs = prob_mask[:, receiver_idx]
    if issparse(probs):
        probs = np.asarray(probs.todense()).ravel()

    plt.scatter(
        receiver_coords[receiver_idx, 0],
        receiver_coords[receiver_idx, 1],
        color='red',
        marker='*',
        s=50,
        edgecolor='black',
        label='Receiving Neuron'
    )
    
    scatter = plt.scatter(
        sender_coords[:, 0], 
        sender_coords[:, 1], 
        c=probs, 
        cmap=cmap,
        vmin=vmin,
        vmax=vmax, 
        s=30, 
        alpha=0.3
    )
    

    
    plt.colorbar(scatter, label='Weight w')
    plt.xlabel('Expression Level x')
    plt.ylabel('Expression Level y')
    plt.title(f'{title}', fontsize=14, pad=20)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal') 
    plt.tight_layout()
    plt.show()


def get_sorted_index(r_vals_ext, idx):

    r_vals = np.asarray(r_vals_ext)
    
    sorted_indices = np.lexsort((r_vals[:, 1], r_vals[:, 0]))[::-1]
    
    first_original_index = sorted_indices[idx]
    
    return first_original_index



def sort_prob_mask_2d_comb(prob_mask, layer_sender, layer_receiver, N):
    mid = int(N/2)

    r_vals_E = np.array([neuron.r for neuron in layer_sender[0:mid]])
    r_vals_I = np.array([neuron.r for neuron in layer_sender[mid:N]])
    l_vals = np.array([neuron.l for neuron in layer_receiver])
    
    idx_r_E = np.lexsort((r_vals_E[:, 0], r_vals_E[:, 1]))[::-1]
    
    idx_r_I = np.lexsort((r_vals_I[:, 0], r_vals_I[:, 1]))[::-1] + mid

    idx_r = np.hstack((idx_r_E, idx_r_I))
    
    idx_l = np.lexsort((l_vals[:, 0], l_vals[:, 1]))[::-1]
    
    sorted_prob_mask = prob_mask[np.ix_(idx_r, idx_l)]
    
    return sorted_prob_mask, idx_r, idx_l


def sort_mask_2d(layer_sender, layer_receiver, N):
    mid = int(N/2)

    r_vals_E = np.array([neuron.r for neuron in layer_sender[0:mid]])
    r_vals_I = np.array([neuron.r for neuron in layer_sender[mid:N]])
    l_vals = np.array([neuron.l for neuron in layer_receiver])

    idx_r_E = np.lexsort((r_vals_E[:, 0], r_vals_E[:, 1]))[::-1]
    idx_r_I = np.lexsort((r_vals_I[:, 0], r_vals_I[:, 1]))[::-1]
    idx_l = np.lexsort((l_vals[:, 0], l_vals[:, 1]))[::-1]

    combined_idx_r = np.concatenate((idx_r_E, idx_r_I + mid))

    sort_mask = np.ix_(combined_idx_r, idx_l)
    
    return sort_mask

def ReLU(x):
    return x * (x > 0)

def get_indices_in_circle(coords, radius, cx = None , cy = None):

    if cx == None:

        min_x, max_x = np.min(coords[:, 0]), np.max(coords[:, 0])
        cx = np.random.uniform(min_x, max_x)
    
    if cy == None:

        min_y, max_y = np.min(coords[:, 1]), np.max(coords[:, 1])
        cy = np.random.uniform(min_y, max_y)

    distances = np.sqrt((coords[:, 0] - cx)**2 + (coords[:, 1] - cy)**2)
    
    indices_in_range = np.where(distances <= radius)[0]
    
    return indices_in_range, cx, cy

def plot_stim(coords, stim_indices, center_x, center_y, radius):

    plt.figure(figsize=(8, 8))
    plt.scatter(coords[:, 0], coords[:, 1], c='lightgray', label='All Neurons')
    plt.scatter(coords[stim_indices, 0], coords[stim_indices, 1], c='red', label='Stimulated Neurons')

    circle = plt.Circle((center_x, center_y), radius, color='red', fill=False, linestyle='--')
    plt.gca().add_patch(circle)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.title("Stimulus on Neurons")
    plt.show()


def plot_stim_act(A, x, y, vmin=-1, vmax=1, 
                  title='Values Plotted at Specific Coordinates',
                  xlabel='X Coordinate', 
                  ylabel='Y Coordinate',
                  cbar_label='Value magnitude'):

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(x, y, c=A, vmin=vmin, vmax=vmax, cmap='seismic', s=30, alpha=0.6)

    cbar = plt.colorbar(scatter)
    cbar.set_label(cbar_label, size=16)

    plt.title(title, fontsize=16, pad=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)


    plt.show()


def plot_stim_act_image(A, x, y, grid_size=(50, 50), sigma=1, vmin=-1, vmax=1,
                        title=None, xlabel='X Coordinate', 
                        ylabel='Y Coordinate', cbar_label='Average Value Magnitude'):
    
    plt.figure(figsize=(8, 6))

    if isinstance(grid_size, int):
        bins = [grid_size, grid_size]
    else:
        bins = grid_size

    sum_A, x_edges, y_edges = np.histogram2d(x, y, bins=bins, weights=A)
    count, _, _ = np.histogram2d(x, y, bins=bins)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_A = np.true_divide(sum_A, count)
        
    valid_mask = count > 0
    
    avg_A_filled = np.nan_to_num(avg_A, nan=0.0)
    
    smoothed_A = gaussian_filter(avg_A_filled, sigma=sigma)
    smoothed_mask = gaussian_filter(valid_mask.astype(float), sigma=sigma)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        smoothed_avg_A = smoothed_A / smoothed_mask
        
    smoothed_avg_A[smoothed_mask < 0.01] = np.nan
    
    im = plt.imshow(smoothed_avg_A.T, origin='lower', 
                    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                    cmap='seismic', alpha=0.6, vmin=vmin, vmax=vmax)

    plt.colorbar(im, label=cbar_label)

    if title is None:
        title = f'Smoothed Average Activity (sigma={sigma})'
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.grid(False) 

    plt.show()


def plot_radial_activity(A, x, y, cx, cy, num_bins=20, max_radius=None, vmin=-1, vmax=1,
                         title1='Spatial Activity', xlabel1='X Coordinate', ylabel1='Y Coordinate',
                         title2='Average Input vs. Distance from Center', 
                         xlabel2='Radius (Distance from center)', ylabel2='Average Input'):
    
    distances = np.sqrt((x - cx)**2 + (y - cy)**2)

    if max_radius is None:
        max_radius = np.max(distances)
        
    bins = np.linspace(0, max_radius, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    bin_means = np.zeros(num_bins)
    
    for i in range(num_bins):
        in_bin = (distances >= bins[i]) & (distances < bins[i+1])
        
        if np.any(in_bin):
            bin_means[i] = np.mean(A[in_bin])
        else:
            bin_means[i] = np.nan
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    scatter = ax1.scatter(x, y, c=A, vmin=vmin, vmax=vmax, cmap='seismic', s=30, alpha=0.6)
    
    ax1.scatter(cx, cy, c='black', marker='x', s=100, label='Stimulus Center')
    
    fig.colorbar(scatter, ax=ax1, label='Value magnitude')
    ax1.set_title(title1)
    ax1.set_xlabel(xlabel1)
    ax1.set_ylabel(ylabel1)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(bin_centers, bin_means, marker='o', linestyle='-', color='b')
    
    ax2.set_title(title2)
    ax2.set_xlabel(xlabel2)
    ax2.set_ylabel(ylabel2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_combined_radial_activity(A_list, x_list, y_list, cx, cy, labels=None, num_bins=20, max_radius=None, vmin=-1, vmax=1):
    n_maps = len(A_list)
    
    if labels is None:
        labels = [f"Map {i+1}" for i in range(n_maps)]
        
    fig, axes = plt.subplots(1, n_maps + 1, figsize=(4.5 * (n_maps + 1), 4.5), constrained_layout=True)
    
    if not isinstance(cx, (list, tuple, np.ndarray)):
        cx = [cx] * n_maps
    if not isinstance(cy, (list, tuple, np.ndarray)):
        cy = [cy] * n_maps
        
    if max_radius is None:
        max_radius = 0
        for x, y, c_x, c_y in zip(x_list, y_list, cx, cy):
            d_max = np.max(np.sqrt((x - c_x)**2 + (y - c_y)**2))
            if d_max > max_radius:
                max_radius = d_max
                
    bins = np.linspace(0, max_radius, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    ax_radial = axes[-1]
    colors = ['b', 'r', 'g', 'purple', 'orange'] 
    
    scatter_plots = []
    
    for i, (A, x, y, c_x, c_y) in enumerate(zip(A_list, x_list, y_list, cx, cy)):
        distances = np.sqrt((x - c_x)**2 + (y - c_y)**2)
        bin_means = np.zeros(num_bins)
        
        for j in range(num_bins):
            in_bin = (distances >= bins[j]) & (distances < bins[j+1])
            if np.any(in_bin):
                bin_means[j] = np.mean(A[in_bin])
            else:
                bin_means[j] = np.nan
                
        ax_map = axes[i]
        ax_map.set_aspect('equal', adjustable='box')
        
        scatter = ax_map.scatter(x, y, c=A, vmin=vmin, vmax=vmax, cmap='seismic', s=30, alpha=0.6)
        scatter_plots.append(scatter)
        
        ax_map.scatter(c_x, c_y, c='black', marker='x', s=100, label='Stimulus Center')
        ax_map.set_title(f'{labels[i]}')
        ax_map.set_xlabel('Expression Level x', fontsize=14)
        
        if i == 0:
            ax_map.set_ylabel('Expression Level y', fontsize=14) 
            
        ax_map.grid(True, alpha=0.3)
        if i == 0:  
            ax_map.legend()
            
        ax_radial.plot(bin_centers, bin_means, marker='o', markersize=4, linestyle='-', 
                       color=colors[i % len(colors)], label=labels[i])

    cbar = fig.colorbar(scatter_plots[0], ax=axes[n_maps - 1], shrink=0.8)
    cbar.set_label('Input Level', fontsize=14)
    
    ax_radial.set_box_aspect(1)
    ax_radial.set_title('Avg. Input of Neurons within Distance', fontsize=14)
    ax_radial.set_xlabel('Euclidean Distance from center)', fontsize=14)
    ax_radial.set_ylabel('Average Input', fontsize=14)
    ax_radial.grid(True, alpha=0.3)
    ax_radial.legend()
    
    plt.show()

def plot_avg_activity_vs_stimulus_radius(coords, W, cx, cy, min_radius=0.01, max_radius=0.5, num_steps=30):
    radii = np.linspace(min_radius, max_radius, num_steps)
    avg_activities = []
    
    N = len(coords)

    for r in radii:
        indices, _, _ = get_indices_in_circle(coords, r, cx, cy)
        
        mask = np.zeros(N)
        if len(indices) > 0:
            mask[indices] = 1
            
        A = mask @ W
        
        avg_activities.append(np.mean(A))

    plt.figure(figsize=(8, 6))
    plt.plot(radii, avg_activities, marker='o', linestyle='-', color='purple')
    
    plt.title('Average Grid Activity vs. Stimulus Radius')
    plt.xlabel('Stimulus Radius')
    plt.ylabel('Average Grid Activity')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return radii, avg_activities

def get_indices_in_line(coords, theta, thickness, length=None, cx=None, cy=None):

    if cx is None:
        min_x, max_x = np.min(coords[:, 0]), np.max(coords[:, 0])
        cx = np.random.uniform(min_x, max_x)
    
    if cy is None:
        min_y, max_y = np.min(coords[:, 1]), np.max(coords[:, 1])
        cy = np.random.uniform(min_y, max_y)

    dx = coords[:, 0] - cx
    dy = coords[:, 1] - cy
    
    perp_dist = np.abs(-dx * np.sin(theta) + dy * np.cos(theta))
    
    valid_mask = perp_dist <= (thickness / 2.0)
    
    if length is not None:
        par_dist = np.abs(dx * np.cos(theta) + dy * np.sin(theta))
        valid_mask &= (par_dist <= (length / 2.0))
        
    indices_in_range = np.where(valid_mask)[0]
    
    return indices_in_range, cx, cy

def create_stimulus_mask(image_matrix, target_size):
    image_matrix = np.asarray(image_matrix, dtype=float)
    
    scale_x = target_size / image_matrix.shape[0]
    scale_y = target_size / image_matrix.shape[1]
    
    scaled_matrix = zoom(image_matrix, (scale_x, scale_y), order=1)
    
    min_val = scaled_matrix.min()
    max_val = scaled_matrix.max()
    
    if max_val > min_val:
        normalized_mask = (scaled_matrix - min_val) / (max_val - min_val)
    else:
        normalized_mask = np.clip(scaled_matrix, 0.0, 1.0)
    
    return normalized_mask

def process_png_stimulus(image_path, target_size):
    img = Image.open(image_path)
    
    img_gray = img.convert('L')
    
    img_matrix = np.array(img_gray)
    
    scaled_mask = create_stimulus_mask(img_matrix, target_size)

    scaled_mask = scaled_mask.T
    
    scaled_mask = np.fliplr(scaled_mask)

    return scaled_mask

def sigmoid_centered_scaled(x: np.ndarray, k: float = 10.0) -> np.ndarray:
    def raw_sigmoid(val):
        return 1 / (1 + np.exp(-k * (val - 0.7)))
    
    s_min = raw_sigmoid(0.0)
    s_max = raw_sigmoid(1.0)
    
    s_x = raw_sigmoid(x)
    
    return (s_x - s_min) / (s_max - s_min)

def pick_neuron_for_angle_and_center(target_angle_deg, l_vals, responses, cx=0.5, cy=0.5, radius=0.03):
    num_angles = responses.shape[0]
    inferred_angles = np.linspace(0, 180, num_angles)
    
    max_responses = np.max(responses, axis=0)
    alive_mask = max_responses > 0.0001 
    
    max_response_indices = np.argmax(responses, axis=0)
    pref_angles = inferred_angles[max_response_indices]
    
    x_coords = l_vals[:, 0]
    y_coords = l_vals[:, 1]
    distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    
    valid_mask = (distances <= radius) & alive_mask
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        print(f"Warning: No alive neurons found within radius {radius} of ({cx}, {cy}).")
        return None
        
    subset_pref_angles = pref_angles[valid_indices]
    
    raw_diffs = np.abs(subset_pref_angles - target_angle_deg)
    angle_diffs = np.minimum(raw_diffs, 180.0 - raw_diffs)
    
    best_subset_idx = np.argmin(angle_diffs)
    neuroN_L0_Idx = valid_indices[best_subset_idx]
    
    actual_pref = pref_angles[neuroN_L0_Idx]
    actual_dist = distances[neuroN_L0_Idx]
    
    print(f"Selected Neuron {neuroN_L0_Idx} | Preferred Angle: {actual_pref:.1f}° (Target was {target_angle_deg}°) | Distance to center: {actual_dist:.4f}")
        
    return neuroN_L0_Idx

def get_closest_to_center_idx(coords, cx=0.5, cy=0.5):
    
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    
    closest_idx = np.argmin(distances)
    
    return closest_idx

def warp_coords(xy, theta, sx, sy):
    theta_rad = theta
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    x = xy[:, 0] 
    y = xy[:, 1]

    x *= sx
    y *= sy

    # # Clockwise
    # x_rot =  x * cos_t + y * sin_t
    # y_rot = -x * sin_t + y * cos_t

    x_rot = x * cos_t - y * sin_t
    y_rot = x * sin_t + y * cos_t

    return np.column_stack((x_rot, y_rot))



# def compute_prob_2d_sparse(r, l, sigma_conn, cutoff_sigmas=6.0):
#     if len(r) == 0 or len(l) == 0:
#         return csr_matrix((len(r), len(l)))
#     max_dist = cutoff_sigmas * sigma_conn
#     tree_r = cKDTree(r)
#     tree_l = cKDTree(l)
#     d = tree_r.sparse_distance_matrix(tree_l, max_distance=max_dist, output_type='coo_matrix')
#     data = np.exp(-(d.data**2) / (2 * sigma_conn**2))
#     return csr_matrix((data, (d.row, d.col)), shape=d.shape)


def compute_prob_2d_sparse(r, l, sigma_conn, kappa=None, cutoff_sigmas=6.0):
    if len(r) == 0 or len(l) == 0:
        return csr_matrix((len(r), len(l)))

    r_xy, l_xy = r[:, :2], l[:, :2]
    max_dist = cutoff_sigmas * sigma_conn
    tree_r = cKDTree(r_xy)
    tree_l = cKDTree(l_xy)
    d = tree_r.sparse_distance_matrix(tree_l, max_distance=max_dist,
                                      output_type='coo_matrix')

    data = np.exp(-(d.data**2) / (2 * sigma_conn**2))          

    if kappa is not None and r.shape[1] >= 3 and l.shape[1] >= 3:
        dtheta = r[d.row, 2] - l[d.col, 2]
        data = data * np.exp(kappa * (np.cos(2.0 * dtheta) - 1.0))

    return csr_matrix((data, (d.row, d.col)), shape=d.shape)


def sample_bernoulli_sparse(prob_sparse, K=1):
    prob_csr = prob_sparse.tocsr()
    draws = np.random.binomial(K, prob_csr.data).astype(float)
    out = csr_matrix((draws, prob_csr.indices, prob_csr.indptr), shape=prob_csr.shape)
    out.eliminate_zeros()
    return out

def fixed_num_connectivity_sparse(sources, targets, num, sigma, weight):
    n_sources = sources.shape[0]
    n_targets = targets.shape[0]
    W = sp.lil_matrix((n_sources, n_targets), dtype=float)

    for i, src in enumerate(sources):
        d = np.sqrt(((targets - src) ** 2).sum(axis=1))
        in_range = np.where(d < sigma)[0]
        if len(in_range) > 0:
            np.random.shuffle(in_range)
            chosen = in_range[:num]
            W[i, chosen] = weight

    return W.tocsr()