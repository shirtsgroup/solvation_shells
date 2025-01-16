# functions for linear algebra operations

import numpy as np

def unitize(v):
    '''Create a unit vector from vector v'''
    return v / np.linalg.norm(v)


def get_angle(v1, v2):
    '''Get the angle between two vectors v1 and v2'''
    
    v1 = unitize(v1)
    v2 = unitize(v2)

    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


def create_plane_from_point_and_normal(point, normal):
    '''Get the parameters for a plane from a point and normal vector, of the form Ax + By + Cz + D = 0'''

    # Extract the components of the point and normal vector
    x0, y0, z0 = point
    A, B, C = normal

    # Calculate D using the point
    D = -(A * x0 + B * y0 + C * z0)

    return A, B, C, D


def line_plane_intersection(p1, p2, A, B, C, D):
    '''Find intersection of a plane (Ax + By + Cz + D = 0) with a line segment between points p1 and p2'''
    p1 = np.array(p1)
    p2 = np.array(p2)
    u = p2 - p1
    t = -(A * p1[0] + B * p1[1] + C * p1[2] + D) / (A * u[0] + B * u[1] + C * u[2])
    if 0 <= t <= 1:
        return p1 + t * u
    else:
        return None
    

def project_to_plane(points):
    '''Project coplanar 3D points to the xy-plane'''

    from sklearn.decomposition import PCA

    # Subtract the mean of the points to center them at the origin
    mean_point = np.mean(points, axis=0)
    centered_points = points - mean_point

    # Use PCA to find the best-fit plane
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    
    # The first two principal components define the plane
    normal_vector = pca.components_[2]
    
    # Create a rotation matrix that aligns the normal vector with the Z-axis
    z_axis = np.array([0, 0, 1])
    rotation_matrix = get_rotation_matrix(normal_vector, z_axis)
    
    # Rotate the centered points
    rotated_points = centered_points.dot(rotation_matrix.T)
    
    # Project onto the XY-plane by discarding the Z-coordinate
    projected_points = rotated_points[:, :2]
    
    return projected_points, rotation_matrix, mean_point


def get_rotation_matrix(vec1, vec2):
    '''Get the rotation matrix to align vec1 to vec2'''
    
    # Normalize the input vectors
    vec1 = unitize(vec1)
    vec2 = unitize(vec2)

    # Compute the rotation axis and angle
    axis = np.cross(vec1, vec2)
    angle = np.arccos(np.dot(vec1, vec2))

    # Handle the case where the vectors are parallel or anti-parallel
    if np.linalg.norm(axis) < 1e-10:
        return np.eye(3) if np.dot(vec1, vec2) > 0 else -np.eye(3)

    axis = axis / np.linalg.norm(axis)
    
    # Compute the skew-symmetric cross-product matrix of the axis
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    # Compute the rotation matrix using the Rodrigues' rotation formula
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    return rotation_matrix