import os
import numpy as np

def load_xyz(filename):
    return np.loadtxt(filename, skiprows=1)

def kmeans(points, k, max_iters=100, tol=1e-4):
    np.random.seed(42)
    centers = points[np.random.choice(points.shape[0], k, replace=False)]

    for _ in range(max_iters):
        distances = np.linalg.norm(points[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centers = np.array([points[labels == i].mean(axis=0) for i in range(k)])

        if np.all(np.linalg.norm(new_centers - centers, axis=1) < tol):
            break
        centers = new_centers

    return labels, centers


def ransac_plane(points, max_iters=100, threshold=0.01):
    best_inliers = []
    best_plane = None

    for _ in range(max_iters):
        sample = points[np.random.choice(points.shape[0], 3, replace=False)]

        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)

        a, b, c = normal
        d = -np.dot(normal, sample[0])

        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
        distances /= np.sqrt(a ** 2 + b ** 2 + c ** 2)

        inliers = points[distances < threshold]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (a, b, c, d)

    return best_plane, best_inliers


def is_planar(inliers, total_points, threshold=0.7):
    inlier_ratio = len(inliers) / total_points
    return inlier_ratio >= threshold

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

    k = 3
    labels, centers = kmeans(all_points, k)

    for cluster_idx in range(k):
        cluster_points = all_points[labels == cluster_idx]

        plane, inliers = ransac_plane(cluster_points)

        if is_planar(inliers, len(cluster_points)):
            a, b, c, d = plane
            normal = np.array([a, b, c])

            orientation = check_plane_orientation(normal)

            print(f"Cluster {cluster_idx + 1}:")
            print(f"  Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
            print(f"  Orientation: {orientation}")
            print(f"  Number of inliers: {len(inliers)}")
        else:
            print(f"Cluster {cluster_idx + 1}:")
            print("  Not a plane (low inlier ratio).")
            print(f"  Number of inliers: {len(inliers)}")
