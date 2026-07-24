import math
import numpy as np
from typing import Tuple, List

def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculates the Euclidean distance between two 2D points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calculate_angle_between_vectors(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Calculates the angle in degrees between two 2D vectors."""
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    denom = (math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2))
    if denom == 0:
        return 0.0
    val = dot_product / denom
    # Clip to avoid floating point errors
    val = max(-1.0, min(1.0, val))
    return math.degrees(math.acos(val))

def project_vector(v: Tuple[float, float], u: Tuple[float, float]) -> Tuple[float, float]:
    """Projects vector v onto vector u."""
    u_len_sq = u[0]**2 + u[1]**2
    if u_len_sq == 0:
        return (0.0, 0.0)
    dot_product = v[0] * u[0] + v[1] * u[1]
    factor = dot_product / u_len_sq
    return (factor * u[0], factor * u[1])

def project_vector_magnitude(v: Tuple[float, float], u: Tuple[float, float]) -> float:
    """Calculates the signed magnitude of vector v projected onto vector u."""
    u_len = math.sqrt(u[0]**2 + u[1]**2)
    if u_len == 0:
        return 0.0
    dot_product = v[0] * u[0] + v[1] * u[1]
    return dot_product / u_len

def fit_occlusal_plane(points: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Fits a straight line representing the occlusal plane using RANSAC.
    Returns the slope and intercept, and a normalized directional vector of the plane.
    """
    if len(points) < 2:
        # Default flat occlusal plane
        return (0.0, 0.5), (1.0, 0.0)
    
    # RANSAC-like line fitting to handle outliers (e.g. severely misaligned teeth)
    best_slope = 0.0
    best_intercept = 0.5
    max_inliers = 0
    threshold = 0.03 # threshold in normalized coordinates
    
    pts = np.array(points)
    n = len(pts)
    
    # If small number of points, fit directly with simple linear regression
    if n <= 3:
        x = pts[:, 0]
        y = pts[:, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        v_op = (1.0, m)
        v_op_len = math.sqrt(1.0 + m**2)
        v_op_norm = (1.0 / v_op_len, m / v_op_len)
        return (m, c), v_op_norm

    for i in range(20): # 20 iterations is sufficient for small 2D point sets
        # Sample 2 random points
        idx = np.random.choice(n, 2, replace=False)
        p1, p2 = pts[idx[0]], pts[idx[1]]
        
        if p2[0] - p1[0] == 0:
            continue
            
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        c = p1[1] - m * p1[0]
        
        # Calculate distances of all points to this line
        # Distance = |m*x - y + c| / sqrt(m^2 + 1)
        denom = math.sqrt(m**2 + 1)
        distances = np.abs(m * pts[:, 0] - pts[:, 1] + c) / denom
        inliers = np.sum(distances < threshold)
        
        if inliers > max_inliers:
            max_inliers = inliers
            best_slope = m
            best_intercept = c
            
    # Re-fit using only inliers for higher accuracy
    denom = math.sqrt(best_slope**2 + 1)
    distances = np.abs(best_slope * pts[:, 0] - pts[:, 1] + best_intercept) / denom
    inlier_pts = pts[distances < threshold]
    
    if len(inlier_pts) >= 2:
        x = inlier_pts[:, 0]
        y = inlier_pts[:, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        best_slope, best_intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        
    v_op = (1.0, best_slope)
    v_op_len = math.sqrt(1.0 + best_slope**2)
    v_op_norm = (1.0 / v_op_len, best_slope / v_op_len)
    
    return (best_slope, best_intercept), v_op_norm

def get_calibration_scale(p_left: Tuple[float, float], p_right: Tuple[float, float], real_width_mm: float = 3.2) -> float:
    """
    Computes scale factor (mm/pixel or mm/normalized unit) based on a known calibration dimension
    (e.g., standard bracket width = 3.2 mm).
    """
    px_dist = calculate_distance(p_left, p_right)
    if px_dist == 0:
        return 1.0 # Fallback
    return real_width_mm / px_dist
