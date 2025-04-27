import os
import numpy as np
import pyransac3d
from sklearn.cluster import DBSCAN

def load_xyz(filename):
    return np.loadtxt(filename, skiprows=1)


def dbscan_clustering(points, eps=1.4, min_samples=20):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    return labels

def fit_plane_pyransac(points, threshold=0.005):
    plane = pyransac3d.Plane()
    coefficients, inliers = plane.fit(points, threshold)
    normal = np.array([coefficients[0], coefficients[1], coefficients[2]])
    normal /= np.linalg.norm(normal)
    return (coefficients[0], coefficients[1], coefficients[2], coefficients[3]), normal, inliers


def check_plane_orientation(normal):
    vertical_threshold = 0.1
    if abs(normal[2]) < vertical_threshold:
        return "Vertical"
    else:
        return "Horizontal"

xyz_directory = r"D:\Nowy folder\rogramowanie-w-obliczeniach-inteligentnych\Ćwiczenie 1"
horizontal_plane_file = os.path.join(xyz_directory, "horizontal_plane.xyz")
vertical_plane_file = os.path.join(xyz_directory, "vertical_plane.xyz")
cylindrical_surface_file = os.path.join(xyz_directory, "cylindrical_surface.xyz")

if __name__ == "__main__":
    horizontal_points = load_xyz(horizontal_plane_file)
    vertical_points = load_xyz(vertical_plane_file)
    cylindrical_points = load_xyz(cylindrical_surface_file)

    all_points = np.vstack([horizontal_points, vertical_points, cylindrical_points])

    labels = dbscan_clustering(all_points)

    unique_labels = set(labels)
    for cluster_idx in unique_labels:
        if cluster_idx == -1:
            continue

        cluster_points = all_points[labels == cluster_idx]

        if len(cluster_points) < 30:
            print(f"Cluster {cluster_idx + 1}: Skipped (too few points).")
            continue

        plane, normal, inliers = fit_plane_pyransac(cluster_points)

        inlier_ratio = len(inliers) / len(cluster_points)
        if inlier_ratio >= 0.7:
            a, b, c, d = plane
            orientation = check_plane_orientation(normal)

            print(f"Cluster {cluster_idx + 1}:")
            print(f"  Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
            print(f"  Orientation: {orientation}")
            print(f"  Number of inliers: {len(inliers)}")
            print(f"  Cluster points: {len(cluster_points)}")
        else:
            print(f"Cluster {cluster_idx + 1}:")
            print("  Not a plane (low inlier ratio).")
            print(f"  Number of inliers: {len(inliers)}")
            print(f"  Cluster points: {len(cluster_points)}")
