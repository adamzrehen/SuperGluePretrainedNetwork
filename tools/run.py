import numpy as np
import os
import tqdm
from threed_reconstruction import (estimate_initial_pose, triangulate_points, register_new_image,
                                   triangulate_new_points_from_two_views, match_features)
from threed_plotter import plot_point_cloud
from match_pairs_refactored import SuperGlueEvaluator, parse_arguments


def main():
    # Setup input directory and obtain all keypoint files.
    input_dir = '/home/adam/Documents/Experiments/SuperGlue/ichilov_1/results'
    keypoint_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                      if f.endswith('.npz')]
    keypoint_files.sort()

    # Load the first file to initialize the reconstruction.
    data = np.load(keypoint_files[0])
    keypoints0 = data['keypoints0']
    keypoints1 = data['keypoints1']
    descriptors0 = data['descriptors0']
    descriptors1 = data['descriptors1']
    matches_initial = data['matches']
    scores0 = data['scores0']

    # Camera intrinsic matrix.
    K = np.array([[1.415e+03, 0.0,      7.075e+02],
                  [0.0,       1.415e+03, 5.400e+02],
                  [0.0,       0.0,      1.0]])

    # Estimate the initial pose from the first pair.
    R, t, mask_pose, pts1, pts2 = estimate_initial_pose(keypoints0, keypoints1,
                                                        matches_initial, K)
    pts3d_init = triangulate_points(K, R, t, pts1, pts2, mask_pose)

    # Initialize camera poses and 3D point cloud.
    camera_poses = {}
    camera_poses[0] = (np.eye(3), np.zeros((3, 1)))  # First camera at origin.
    camera_poses[1] = (R, t)
    points3d = pts3d_init.copy()
    kp_existing = keypoints0.copy()    # Base keypoints for the initial image.
    desc_existing = descriptors0.copy().T  # Base descriptors for the initial image.
    scores_existing = scores0.copy()

    # Setup SuperGlue matching via the evaluator.
    opt = parse_arguments()
    evaluator = SuperGlueEvaluator(opt)

    # Process subsequent images.
    for i in tqdm.tqdm(range(2, len(keypoint_files))):
        data = np.load(keypoint_files[i])
        # Assume new image keypoints are stored under key 'keypoints1'
        kp_new = data['keypoints1']
        desc_new = data['descriptors1']
        scores_new = data['scores1']

        # Register the new image using SuperGlue-based matching.
        R_new, t_new, inliers, _ = register_new_image(
            kp_existing, points3d, kp_new,
            desc_existing.T, desc_new, scores_existing, scores_new,
            K, evaluator.matching, device='cuda'
        )

        # Save the new camera pose.
        camera_poses[i] = (R_new, t_new)

        # Use the previous image (if available) to triangulate new points.
        ref_idx = i - 1
        if ref_idx in camera_poses:
            R_ref, t_ref = camera_poses[ref_idx]
            additional_matches = match_features(desc_existing, desc_new.T)
            if len(additional_matches) >= 6:
                pts_existing = np.float32([kp_existing[m.queryIdx] for m in additional_matches])
                pts_new = np.float32([kp_new[m.trainIdx] for m in additional_matches])
                new_pts3d = triangulate_new_points_from_two_views(
                    K, R_ref, t_ref, R_new, t_new, pts_existing, pts_new
                )
                # Merge the newly triangulated points into the global point cloud.
                points3d = np.vstack((points3d, new_pts3d))
                print(f"Image {i}: Triangulated {new_pts3d.shape[0]} new points.")
            else:
                print(f"Image {i}: Not enough additional matches for triangulation.")

        # Update the reference keypoints and descriptors by merging new features.
        kp_existing = np.vstack((kp_existing, kp_new))
        desc_existing = np.vstack((desc_existing, desc_new.T))
        scores_existing = np.hstack((scores_existing, scores_new))

    # Plot the final 3D point cloud.
    plot_point_cloud(points3d)


if __name__ == '__main__':
    main()