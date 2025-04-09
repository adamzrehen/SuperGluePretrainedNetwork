import cv2
import numpy as np
import torch
from covariance_analysis import check_spatial_distribution


def estimate_initial_pose(kp1, kp2, matches, K):
    """
    Estimate the relative pose (R, t) between two views and return the inlier mask,
    along with the corresponding matched points.

    Parameters:
    - kp1: NumPy array of shape (N, 2) containing [x, y] coordinates of keypoints in image 1.
    - kp2: NumPy array of shape (M, 2) containing [x, y] coordinates of keypoints in image 2.
    - matches: NumPy array of shape (N, 1) where each entry is the index of a matching keypoint in kp2
               (assumed to be -1 if no match exists).
    - K: Camera intrinsic matrix.

    Returns:
    - R: Recovered rotation matrix.
    - t: Recovered translation vector.
    - mask_pose: Inlier mask from cv2.recoverPose.
    - pts1: Filtered matched points from the first image.
    - pts2: Filtered matched points from the second image.
    """
    # Remove the extra dimension in matches (shape becomes (N,))
    matches = matches.squeeze()

    # Select valid matches (assuming invalid match indices are marked by -1)
    valid = matches >= 0
    valid_kp_indices = np.where(valid)[0]

    # Gather the matching points
    pts1 = kp1[valid]               # Shape: (num_valid, 2)
    pts2 = kp2[matches[valid]]        # Shape: (num_valid, 2)

    # Estimate the essential matrix using RANSAC
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Recover the relative camera pose from the essential matrix
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    return R, t, mask_pose, pts1, pts2, valid_kp_indices


def triangulate_points(K, R, t, pts1, pts2, mask_pose):
    """
    Triangulates 3D points from two views using the camera matrix K,
    relative pose (R, t), and the corresponding matched points.

    Parameters:
    - K: 3x3 camera intrinsic matrix.
    - R: 3x3 rotation matrix from image 1 to image 2.
    - t: 3x1 translation vector from image 1 to image 2.
    - pts1: Matched points from image 1 (Nx2 array).
    - pts2: Matched points from image 2 (Nx2 array).
    - mask_pose: Inlier mask from cv2.recoverPose (Nx1 array) indicating inlier matches.

    Returns:
    - pts3d: Array of triangulated 3D points in Euclidean coordinates (M x 3), where M is the number
             of inlier points.
    """
    # Filter out the inlier points using the mask from recoverPose
    inlier_mask = (mask_pose.ravel() > 0)

    # Filter points from both images to retain only the inliers.
    pts1_inliers = pts1[inlier_mask]  # shape: (N_inliers, 2)
    pts2_inliers = pts2[inlier_mask]  # shape: (N_inliers, 2)

    # Create the projection matrix for the first camera: [I | 0]
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    # Create the projection matrix for the second camera: [R | t]
    P2 = K @ np.hstack((R, t))

    # Triangulate points (note that cv2.triangulatePoints expects points as 2xN arrays)
    pts4d_hom = cv2.triangulatePoints(P1, P2, pts1_inliers.T, pts2_inliers.T)

    # Convert homogeneous coordinates to Euclidean (divide x, y, z by w)
    pts3d = pts4d_hom[:3, :] / pts4d_hom[3, :]

    # Return the points transposed as M x 3 array
    return pts3d.T


def triangulate_new_points_from_two_views(K, R1, t1, R2, t2, pts1, pts2):
    """
    Triangulate 3D points using two camera poses.
    Parameters:
        K: 3x3 intrinsic matrix.
        R1, t1: Pose of the reference image.
        R2, t2: Pose of the new image.
        pts1, pts2: Corresponding 2D points from the reference and new images, respectively.
    Returns:
        new_pts3d: An (n x 3) array of newly triangulated 3D points.
    """
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    new_pts3d = (pts4d[:3] / pts4d[3]).T
    return new_pts3d


def register_new_image(kp_existing, pts3d_existing, valid_kp_indices, kp_new, desc_existing, desc_new, scores_existing,
                       scores_new, K, matching, device='cpu'):
    """
    Given previously reconstructed 3D points (associated with 2D keypoints in kp_existing)
    and a new image (with its keypoints and descriptors), find 2D-3D correspondences and register the new image.
    """
    # Match features between the new image and the existing images.
    # Run the SuperGlue matching model. The output "matches0" is a tensor of shape [1, N]
    # where each element is the index (in kp_new) of the matched keypoint, or -1 if there is no match.
    kp_existing_tensor = torch.from_numpy(kp_existing).unsqueeze(0).to(device).float()  # Shape: [1, N, 2]
    kp_new_tensor = torch.from_numpy(kp_new).unsqueeze(0).to(device).float()
    desc_existing_tensor = torch.from_numpy(desc_existing).unsqueeze(0).to(device).float()  # Shape: [1, N, D]
    desc_new_tensor = torch.from_numpy(desc_new).unsqueeze(0).to(device).float()
    scores_existing_tensor = torch.from_numpy(scores_existing).unsqueeze(0).to(device).float()  # Shape: [1, N, D]
    scores_new_tensor = torch.from_numpy(scores_new).unsqueeze(0).to(device).float()
    data = {
        'keypoints0': kp_existing_tensor,
        'keypoints1': kp_new_tensor,
        'descriptors0': desc_existing_tensor,
        'descriptors1': desc_new_tensor,
        'scores0': scores_existing_tensor,
        'scores1': scores_new_tensor,
    }

    with torch.no_grad():
        pred = matching(data)
    matches = pred['matches0'][0].cpu().numpy()

    # Extract valid matches (i.e. indices where the match is not -1).
    all_valid_indices = np.where(matches > -1)[0]

    # Only consider indices that have an associated 3D point. TODO: what is the correspondence between indices and 3d points? Seems wrong
    valid_indices = [idx for idx in all_valid_indices if idx in valid_kp_indices]
    if len(valid_indices) < 6:
        raise ValueError("Not enough matches for PnP after filtering valid indices.")

    # Build a mapping from keypoint index in kp_existing to its corresponding index in pts3d_existing.
    kp_to_3d_mapping = {kp_idx: i for i, kp_idx in enumerate(valid_kp_indices)}

    pts3d_corr = []
    pts2d_corr = []
    for idx in valid_indices:
        match_idx = int(matches[idx])
        # Retrieve the corresponding 3D point using the mapping.
        pts3d_corr.append(pts3d_existing[kp_to_3d_mapping[idx]])
        pts2d_corr.append(kp_new[match_idx])
    pts3d_corr = np.array(pts3d_corr, dtype=np.float32)
    pts2d_corr = np.array(pts2d_corr, dtype=np.float32)

    # Compute the camera pose using RANSAC-based PnP.
    success, rvec, tvec, inliers = cv2.solvePnPRansac(pts3d_corr, pts2d_corr, K, None)
    if not success:
        degenerate, eigen_ratio, eigvals = check_spatial_distribution(pts3d_corr, threshold_ratio=0.1,
                                                                      min_eig_threshold=1e-3)
        if degenerate:
            print("Degenerate configuration detected.")
            print("Eigenvalue ratio: {:.4f}".format(eigen_ratio))
            print("Eigenvalues:", eigvals)

        raise RuntimeError("PnP failed to compute camera pose.")
    R_new, _ = cv2.Rodrigues(rvec)
    return R_new, tvec, inliers, matches


import cv2
import numpy as np


def match_features(desc_existing, desc_new, ratio_test=0.75):
    """
    Match features between two sets of descriptors using BFMatcher and Lowe's ratio test.

    Parameters:
        desc_existing (np.ndarray): Descriptors from the existing image (shape: [N, D]).
        desc_new (np.ndarray): Descriptors from the new image (shape: [M, D]).
        ratio_test (float): Threshold for Lowe's ratio test. Default is 0.75.

    Returns:
        list: A list of cv2.DMatch objects that pass the ratio test.
    """
    # Convert descriptors to float32 if they're not already.
    if desc_existing.dtype != np.float32:
        desc_existing = np.asarray(desc_existing, np.float32)
    if desc_new.dtype != np.float32:
        desc_new = np.asarray(desc_new, np.float32)

    # It is also a good idea to check that the descriptors have the same number of columns.
    if desc_existing.shape[1] != desc_new.shape[1]:
        raise ValueError(
            "Descriptor dimensions do not match: {} vs {}".format(desc_existing.shape[1], desc_new.shape[1]))

    # Create a BFMatcher object with L2 norm (suitable for float descriptors)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    # Use k-nearest neighbors matching (k=2)
    knn_matches = bf.knnMatch(desc_existing, desc_new, k=2)

    # Apply Lowe's ratio test to filter matches.
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_test * n.distance:
            good_matches.append(m)

    return good_matches

