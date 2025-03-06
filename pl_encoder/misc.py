import torch


"""
This program could be accelaerated by cuda
"""

def fps(xyz, npoint):
    """
    Furthest Point Sampling
    Input:
        xyz: torch.Tensor of shape (B, N, 3) representing the batch of point clouds.
        npoint: int, number of points to sample (G).
    Output:
        new_xyz: torch.Tensor of shape (B, npoint, 3) representing the sampled center points.
    """
    device = xyz.device
    B, N, _ = xyz.shape
    new_xyz = torch.zeros(B, npoint, 3, device=device)  # To store sampled points
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)  # To store indices of the points
    # Initialize distances as infinity
    distances = torch.full((B, N), float('inf'), device=device)
    
    # Randomly choose the first point for each batch
    farthest = torch.randint(0, N, (B,), device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        new_xyz[:, i, :] = xyz[batch_indices, farthest, :]
        # Expand the selected point to subtract from all points: shape becomes (B, 1, 3)
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        # Compute squared distances from the current centroid to all points
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        # Update distances (keep the minimum distance so far for each point)
        mask = dist < distances
        distances[mask] = dist[mask]
        # Select the next point as the one with the maximum distance from the sampled set
        farthest = torch.max(distances, dim=1)[1]
    
    return new_xyz