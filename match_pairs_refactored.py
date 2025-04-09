#!/usr/bin/env python3
"""
Refactored SuperGlue evaluation script.

This script processes image pair matching and evaluation using SuperGlue.
All the functionality is encapsulated in the SuperGlueEvaluator class.
"""

import argparse
import random
from pathlib import Path
import numpy as np
import matplotlib.cm as cm
import torch

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)


def parse_arguments():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_pairs', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='assets/scannet_sample_images/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default='dump_match_pairs/',
        help='Path to the directory where results and visualizations will be written')
    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize options for input images: if two numbers, use exact dimensions; '
             'if one number, resize so that the max dimension equals the given value; '
             'if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Convert image from uint8 to float before resizing')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by SuperPoint (use -1 to keep all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius (must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument('--viz', action='store_true', help='Visualize the matches and dump the plots')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation (requires ground truth pose and intrinsics)')
    parser.add_argument('--fast_viz', action='store_true',
                        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument('--cache', action='store_true',
                        help='Skip the pair if output .npz files are already found')
    parser.add_argument('--show_keypoints', action='store_true',
                        help='Plot the keypoints in addition to the matches')
    parser.add_argument('--viz_extension', type=str, default='png', choices=['png', 'pdf'],
                        help='Visualization file extension (use pdf for highest quality)')
    parser.add_argument('--opencv_display', action='store_true',
                        help='Visualize via OpenCV before saving output images')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle ordering of pairs before processing')
    parser.add_argument('--force_cpu', action='store_true',
                        help='Force pytorch to run in CPU mode.')

    return parser.parse_args()


class SuperGlueEvaluator:
    def __init__(self, opt):
        self.opt = opt
        self._validate_options()
        self.input_dir = Path(opt.input_dir)
        self.output_dir = Path(opt.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
        print('Running inference on device "{}"'.format(self.device))
        self._setup_resize()
        self._load_model()
        self.timer = AverageTimer(newline=True)
        self.pairs = self._load_pairs(opt.input_pairs, opt.max_length)
        if self.opt.shuffle:
            random.Random(0).shuffle(self.pairs)

    def _validate_options(self):
        if self.opt.opencv_display and not self.opt.viz:
            raise ValueError('Must use --viz with --opencv_display')
        if self.opt.opencv_display and not self.opt.fast_viz:
            raise ValueError('Cannot use --opencv_display without --fast_viz')
        if self.opt.fast_viz and not self.opt.viz:
            raise ValueError('Must use --viz with --fast_viz')
        if self.opt.fast_viz and self.opt.viz_extension == 'pdf':
            raise ValueError('Cannot use pdf extension with --fast_viz')

    def _setup_resize(self):
        """Process the resize command line argument."""
        if len(self.opt.resize) == 2 and self.opt.resize[1] == -1:
            self.opt.resize = self.opt.resize[0:1]
        if len(self.opt.resize) == 2:
            print('Will resize to {}x{} (WxH)'.format(self.opt.resize[0], self.opt.resize[1]))
        elif len(self.opt.resize) == 1 and self.opt.resize[0] > 0:
            print('Will resize max dimension to {}'.format(self.opt.resize[0]))
        elif len(self.opt.resize) == 1:
            print('Will not resize images')
        else:
            raise ValueError('Cannot specify more than two integers for --resize')

    def _load_model(self):
        """Load the SuperPoint and SuperGlue models."""
        config = {
            'superpoint': {
                'nms_radius': self.opt.nms_radius,
                'keypoint_threshold': self.opt.keypoint_threshold,
                'max_keypoints': self.opt.max_keypoints
            },
            'superglue': {
                'weights': self.opt.superglue,
                'sinkhorn_iterations': self.opt.sinkhorn_iterations,
                'match_threshold': self.opt.match_threshold,
            }
        }
        self.matching = Matching(config).eval().to(self.device)

    def _load_pairs(self, input_pairs_path, max_length):
        """Load image pair paths from a file."""
        with open(input_pairs_path, 'r') as f:
            pairs = [line.split() for line in f.readlines()]
        if max_length > -1:
            pairs = pairs[0:min(len(pairs), max_length)]
        return pairs

    def process_pairs(self):
        """Process all image pairs."""
        for i, pair in enumerate(self.pairs):
            self.process_pair(pair, i, len(self.pairs))

        if self.opt.eval:
            self.collate_results()

    def process_pair(self, pair, index, total):
        """Process a single image pair: matching, evaluation, and visualization."""
        stem0 = Path(pair[0]).stem
        stem1 = Path(pair[1]).stem
        matches_path = self.output_dir / f'{stem0}_{stem1}_matches.npz'
        eval_path = self.output_dir / f'{stem0}_{stem1}_evaluation.npz'
        viz_path = self.output_dir / f'{stem0}_{stem1}_matches.{self.opt.viz_extension}'
        viz_eval_path = self.output_dir / f'{stem0}_{stem1}_evaluation.{self.opt.viz_extension}'

        # Decide what steps to perform, taking caching into account.
        do_match = True
        do_eval = self.opt.eval
        do_viz = self.opt.viz
        do_viz_eval = self.opt.eval and self.opt.viz

        if self.opt.cache:
            if matches_path.exists():
                try:
                    results = np.load(matches_path)
                    kpts0, kpts1 = results['keypoints0'], results['keypoints1']
                    matches, conf = results['matches'], results['match_confidence']
                    do_match = False
                except Exception as e:
                    raise IOError(f'Cannot load matches .npz file: {matches_path} -- {e}')
            if self.opt.eval and eval_path.exists():
                try:
                    np.load(eval_path)
                    do_eval = False
                except Exception as e:
                    raise IOError(f'Cannot load evaluation .npz file: {eval_path} -- {e}')
            if self.opt.viz and viz_path.exists():
                do_viz = False
            if self.opt.viz and self.opt.eval and viz_eval_path.exists():
                do_viz_eval = False
            self.timer.update('load_cache')

        # Read rotation info (if available)
        if len(pair) >= 5:
            rot0, rot1 = int(pair[2]), int(pair[3])
        else:
            rot0, rot1 = 0, 0

        # Load the image pair.
        image0, inp0, scales0 = read_image(self.input_dir / pair[0],
                                           self.device, self.opt.resize, rot0, self.opt.resize_float)
        image1, inp1, scales1 = read_image(self.input_dir / pair[1],
                                           self.device, self.opt.resize, rot1, self.opt.resize_float)
        if image0 is None or image1 is None:
            print(f'Problem reading image pair: {self.input_dir / pair[0]} and {self.input_dir / pair[1]}')
            exit(1)
        self.timer.update('load_image')

        # Run matching
        if do_match:
            pred = self.matching({'image0': inp0, 'image1': inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            descriptors0, descriptors1 = pred['descriptors0'], pred['descriptors1']
            matches, conf = pred['matches0'], pred['matching_scores0']
            self.timer.update('matcher')

            # Write matches to disk.
            out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                           'matches': matches, 'match_confidence': conf,
                           'descriptors0': descriptors0, 'descriptors1': descriptors1,
                           'scores0': pred['scores0'], 'scores1': pred['scores1']}
            np.savez(str(matches_path), **out_matches)
        # If not matching (cache enabled), assume values are loaded.
        elif not do_match:
            # Variables kpts0, kpts1, matches, conf are already set from cache.
            pass

        # Keep only valid matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        # Run evaluation (if enabled)
        if do_eval:
            if len(pair) != 38:
                raise ValueError(
                    f'Pair does not have ground truth info. File "{self.opt.input_pairs}" requires 38 valid entries per row')
            K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
            K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
            T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

            # Scale intrinsics based on resize factors.
            K0 = scale_intrinsics(K0, scales0)
            K1 = scale_intrinsics(K1, scales1)

            # Update intrinsics and poses if rotation is provided.
            if rot0 != 0 or rot1 != 0:
                cam0_T_w = np.eye(4)
                cam1_T_w = T_0to1
                if rot0 != 0:
                    K0 = rotate_intrinsics(K0, image0.shape, rot0)
                    cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
                if rot1 != 0:
                    K1 = rotate_intrinsics(K1, image1.shape, rot1)
                    cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
                T_0to1 = cam1_T_w @ np.linalg.inv(cam0_T_w)

            epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
            correct = epi_errs < 5e-4
            num_correct = np.sum(correct)
            precision = np.mean(correct) if len(correct) > 0 else 0
            matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

            thresh = 1.0  # Pixel threshold (relative to resized image size)
            ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
            if ret is None:
                err_t, err_R = np.inf, np.inf
            else:
                R, t, inliers = ret
                err_t, err_R = compute_pose_error(T_0to1, R, t)

            out_eval = {'error_t': err_t,
                        'error_R': err_R,
                        'precision': precision,
                        'matching_score': matching_score,
                        'num_correct': num_correct,
                        'epipolar_errors': epi_errs}
            np.savez(str(eval_path), **out_eval)
            self.timer.update('eval')
        else:
            if self.opt.eval:
                results = np.load(eval_path)
                err_t, err_R = results['error_t'], results['error_R']
                precision = results['precision']
                matching_score = results['matching_score']
                num_correct = results['num_correct']
                epi_errs = results['epipolar_errors']

        # Visualization for matching.
        if do_viz:
            color = cm.jet(mconf)
            text = [
                'SuperGlue',
                f'Keypoints: {len(kpts0)}:{len(kpts1)}',
                f'Matches: {len(mkpts0)}',
            ]
            if rot0 != 0 or rot1 != 0:
                text.append(f'Rotation: {rot0}:{rot1}')
            k_thresh = self.matching.superpoint.config['keypoint_threshold']
            m_thresh = self.matching.superglue.config['match_threshold']
            small_text = [
                f'Keypoint Threshold: {k_thresh:.4f}',
                f'Match Threshold: {m_thresh:.2f}',
                f'Image Pair: {stem0}:{stem1}',
            ]
            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                color, text, str(viz_path), self.opt.show_keypoints,
                self.opt.fast_viz, self.opt.opencv_display, 'Matches', small_text)
            self.timer.update('viz_match')

        # Visualization for evaluation.
        if do_viz_eval:
            color = np.clip((epi_errs - 0) / (1e-3 - 0), 0, 1)
            color = error_colormap(1 - color)
            if self.opt.fast_viz:
                deg, delta = ' deg', 'Delta '
            else:
                deg, delta = '°', '$\\Delta$'
            e_t = 'FAIL' if np.isinf(err_t) else f'{err_t:.1f}{deg}'
            e_R = 'FAIL' if np.isinf(err_R) else f'{err_R:.1f}{deg}'
            text = [
                'SuperGlue',
                f'{delta}R: {e_R}',
                f'{delta}t: {e_t}',
                f'inliers: {num_correct}/{(matches > -1).sum()}',
            ]
            if rot0 != 0 or rot1 != 0:
                text.append(f'Rotation: {rot0}:{rot1}')
            small_text = [
                f'Keypoint Threshold: {k_thresh:.4f}',
                f'Match Threshold: {m_thresh:.2f}',
                f'Image Pair: {stem0}:{stem1}',
            ]
            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                color, text, str(viz_eval_path), self.opt.show_keypoints,
                self.opt.fast_viz, self.opt.opencv_display, 'Relative Pose', small_text)
            self.timer.update('viz_eval')

        self.timer.print(f'Finished pair {index:5} of {total:5}')

    def collate_results(self):
        """Collate evaluation results and print the final summary."""
        pose_errors = []
        precisions = []
        matching_scores = []
        for pair in self.pairs:
            stem0 = Path(pair[0]).stem
            stem1 = Path(pair[1]).stem
            eval_path = self.output_dir / f'{stem0}_{stem1}_evaluation.npz'
            results = np.load(eval_path)
            pose_error = np.maximum(results['error_t'], results['error_R'])
            pose_errors.append(pose_error)
            precisions.append(results['precision'])
            matching_scores.append(results['matching_score'])
        thresholds = [5, 10, 20]
        aucs = pose_auc(pose_errors, thresholds)
        aucs = [100. * val for val in aucs]
        prec = 100. * np.mean(precisions)
        ms = 100. * np.mean(matching_scores)
        print(f'Evaluation Results (mean over {len(self.pairs)} pairs):')
        print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore')
        print(f'{aucs[0]:.2f}\t {aucs[1]:.2f}\t {aucs[2]:.2f}\t {prec:.2f}\t {ms:.2f}')


def main():
    """Main entry point."""
    opt = parse_arguments()
    evaluator = SuperGlueEvaluator(opt)
    evaluator.process_pairs()


if __name__ == '__main__':
    main()
