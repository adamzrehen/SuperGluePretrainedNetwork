import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting


def plot_point_cloud(points, point_size=5, point_color='b', title="3D Point Cloud"):
    """
    Plots a 3D point cloud from an m x 3 NumPy array of triangulated points.

    Parameters:
        points (np.ndarray): An m x 3 array where each row is a point [x, y, z].
        point_size (int, optional): Size of the points in the scatter plot.
        point_color (str, optional): Color of the points.
        title (str, optional): Title of the plot.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("The input 'points' must be an m x 3 NumPy array.")

    # Create a new figure with a 3D axis
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create the scatter plot
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=point_size, c=point_color, marker='o')

    # Label the axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    plt.show()


# Example usage:
if __name__ == "__main__":
    # Generate sample data: 100 random 3D points
    sample_points = np.random.rand(100, 3) * 10  # Scale them for better visualization
    plot_point_cloud(sample_points, point_size=10, point_color='g', title="Triangulated 3D Points")
