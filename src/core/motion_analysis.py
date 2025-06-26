"""Motion analysis functions for optical flow processing"""

import numpy as np
from typing import Tuple


def center_of_mass_variance(flow, num_cells=32):
    """
    Splits the optical flow into a configurable grid (num_cells x num_cells), 
    computes the variance of the optical flow in each grid cell, 
    and returns the center of mass of the variance.
    """
    h, w, _ = flow.shape
    grid_h, grid_w = h // num_cells, w // num_cells

    variance_grid = np.zeros((num_cells, num_cells))
    y_coords, x_coords = np.meshgrid(np.arange(num_cells), np.arange(num_cells), indexing='ij')

    for i in range(num_cells):
        for j in range(num_cells):
            cell = flow[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
            magnitude = np.sqrt(cell[..., 0]**2 + cell[..., 1]**2)
            variance_grid[i, j] = np.var(magnitude)

    total_variance = np.sum(variance_grid)

    if total_variance == 0:
        return (w // 2, h // 2)  # Default to image center if no variance
    else:
        center_x = np.sum(x_coords * variance_grid) * grid_w / total_variance + grid_w / 2
        center_y = np.sum(y_coords * variance_grid) * grid_h / total_variance + grid_h / 2
        return (center_x, center_y)


def max_divergence(flow):
    """
    Computes the divergence of the optical flow over the whole image and returns
    the pixel (x, y) with the highest absolute divergence along with its value.
    """
    # No grid, just pure per-pixel divergence!
    div = np.gradient(flow[..., 0], axis=0) + np.gradient(flow[..., 1], axis=1)
    
    # Get the index (y, x) of the max abs divergence
    y, x = np.unravel_index(np.argmax(np.abs(div)), div.shape)
    return x, y, div[y, x]


def radial_motion_weighted(flow, center, is_cut, pov_mode=False):
    """
    Computes signed radial motion: positive for outward motion, negative for inward motion.
    Closer pixels have higher weight.
    """
    if(is_cut):
        return 0.0
    h, w, _ = flow.shape
    y, x = np.indices((h, w))
    dx = x - center[0]
    dy = y - center[1]

    dot = flow[..., 0] * dx + flow[..., 1] * dy

    # In POV mode, just return the mean dot product
    if(pov_mode):
        return np.mean(dot)
    
    #Cancel out global motion by balancing the averages
    # multiply products to the right of the center (w-x) / w and to the left by x / w
    weighted_dot = np.where(x > center[0], dot * (w - x) / w, dot * x / w)
    # multiply products below the center (h-y) / h and above by y / h
    weighted_dot = np.where(y > center[1], weighted_dot * (h - y) / h, weighted_dot * y / h)

    return np.mean(weighted_dot)


def largest_cluster_center(positions, threshold=10.0):
    """
    BFS to find the largest cluster of swarm positions, return its centroid.
    """
    num_particles = len(positions)
    adj = [[] for _ in range(num_particles)]
    for i in range(num_particles):
        for j in range(i+1, num_particles):
            if np.linalg.norm(positions[i] - positions[j]) < threshold:
                adj[i].append(j)
                adj[j].append(i)

    visited = set()
    clusters = []
    def bfs(start):
        queue, c = [start], []
        while queue:
            node = queue.pop()
            if node in visited: continue
            visited.add(node)
            c.append(node)
            for nei in adj[node]:
                if nei not in visited:
                    queue.append(nei)
        return c

    for i in range(num_particles):
        if i not in visited:
            group = bfs(i)
            clusters.append(group)

    biggest = max(clusters, key=len)
    return (np.mean(positions[biggest], axis=0), len(biggest))


def swarm_positions(flow, num_particles=30, iterations=50):
    """
    Moves 'num_particles' along 'flow' for 'iterations'. Return final positions for clustering.
    """
    h, w, _ = flow.shape
    positions = np.column_stack([
        np.random.uniform(0, w, num_particles),
        np.random.uniform(0, h, num_particles)
    ])
    for _ in range(iterations):
        for i in range(num_particles):
            x_i = int(np.clip(positions[i, 0], 0, w - 1))
            y_i = int(np.clip(positions[i, 1], 0, h - 1))
            vx = flow[y_i, x_i, 1]
            vy = flow[y_i, x_i, 0]
            positions[i, 0] = np.clip(positions[i, 0] + vx, 0, w - 1)
            positions[i, 1] = np.clip(positions[i, 1] + vy, 0, h - 1)
    return positions