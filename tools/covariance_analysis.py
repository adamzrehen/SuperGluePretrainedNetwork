import numpy as np


def check_spatial_distribution(pts, threshold_ratio=0.1, min_eig_threshold=1e-3):
    """
    Checks the spatial distribution of a set of points by evaluating the eigenvalues of
    the covariance matrix of the points.

    Parameters:
        pts (np.ndarray): An NxD array of points (D=2 for 2D points, D=3 for 3D points).
        threshold_ratio (float): Minimum acceptable ratio of the smallest to the largest eigenvalue.
        min_eig_threshold (float): Minimum acceptable absolute value for the smallest eigenvalue.

    Returns:
        degenerate (bool): True if the configuration is considered degenerate, False otherwise.
        eigen_ratio (float): The ratio of the smallest to the largest eigenvalue.
        eigvals (np.ndarray): The eigenvalues of the covariance matrix.
    """
    # Ensure points are float32
    pts = pts.astype(np.float32)
    # Compute the covariance matrix (features are in columns)
    cov = np.cov(pts.T)
    # Compute eigenvalues (they will be real and non-negative for a covariance matrix)
    eigvals, _ = np.linalg.eig(cov)
    # Sort eigenvalues in descending order
    eigvals = np.sort(eigvals)[::-1]
    # Compute the ratio of the smallest to largest eigenvalue
    eigen_ratio = eigvals[-1] / (eigvals[0] + 1e-8)
    # A configuration is degenerate if the smallest eigenvalue is very small, or if the ratio is below threshold.
    degenerate = (eigvals[-1] < min_eig_threshold) or (eigen_ratio < threshold_ratio)
    return degenerate, eigen_ratio, eigvals


