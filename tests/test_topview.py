"""Tests for the corrected top-view projection.

These use synthetic data, so they run without the video, SuperGlue outputs,
or HybridNets weights.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import topview  # noqa: E402


# The reference first-frame -> top-view homography from the original notebook.
H1T = np.array(
    [[-4.08101432e-02, -5.16599153e-01, 2.70619659e02],
     [-2.83020592e-02, -6.85416529e-01, 3.54386187e02],
     [-9.73128797e-05, -2.00980612e-03, 1.00000000e00]]
)


def forward_motion(dx=5.0, dy=30.0):
    """A frame-to-previous-frame homography for constant forward motion."""
    H = np.eye(3)
    H[0, 2] = dx
    H[1, 2] = dy
    return H


class TestHomographyChain:
    def test_first_frame_is_reference(self):
        chain = topview.HomographyChain(H1T)
        assert np.allclose(chain.current(), H1T / H1T[2, 2])

    def test_projection_advances_across_frames(self):
        """The original bug: every frame mapped to the same top-view point."""
        chain = topview.HomographyChain(H1T)
        road_point = np.array([[640.0, 700.0]])

        positions = []
        for _ in range(8):
            chain.advance(forward_motion())
            positions.append(topview.project_points(road_point, chain.current())[0])

        positions = np.array(positions)
        spread = np.linalg.norm(positions[-1] - positions[0])

        # With the bug this spread was exactly 0.0 — the projection never moved.
        assert spread > 5.0, f"projection is not advancing (spread={spread:.2f}px)"

    def test_motion_is_monotonic(self):
        """Constant forward motion should move the point steadily in one direction."""
        chain = topview.HomographyChain(H1T)
        road_point = np.array([[640.0, 700.0]])

        ys = []
        for _ in range(6):
            chain.advance(forward_motion())
            ys.append(topview.project_points(road_point, chain.current())[0][1])

        deltas = np.diff(ys)
        assert np.all(deltas > 0) or np.all(deltas < 0), \
            f"motion is not monotonic: {np.round(deltas, 2)}"

    def test_chain_equals_explicit_product(self):
        steps = [forward_motion(3, 20), forward_motion(7, 25), forward_motion(2, 31)]

        chain = topview.HomographyChain(H1T)
        for H in steps:
            chain.advance(H)

        expected = H1T @ steps[0] @ steps[1] @ steps[2]
        expected = expected / expected[2, 2]

        assert np.allclose(chain.current(), expected)

    def test_identity_steps_do_not_move(self):
        chain = topview.HomographyChain(H1T)
        for _ in range(5):
            chain.advance(np.eye(3))
        assert np.allclose(chain.current(), H1T / H1T[2, 2])


class TestHomographyEstimation:
    def test_recovers_known_transform(self):
        true_H = np.array([[1.0, 0.05, 12.0],
                           [0.02, 1.0, -8.0],
                           [0.0, 0.0, 1.0]])

        rng = np.random.default_rng(0)
        src = rng.uniform(0, 1000, size=(40, 2)).astype(np.float32)
        dst = topview.project_points(src, true_H)

        H, _ = topview.estimate_pairwise_homography(src, dst)
        H = H / H[2, 2]

        assert np.allclose(H, true_H, atol=1e-3)

    def test_ransac_rejects_outliers(self):
        """The reason for adding RANSAC: bad matches must not skew the fit."""
        true_H = np.array([[1.0, 0.05, 12.0],
                           [0.02, 1.0, -8.0],
                           [0.0, 0.0, 1.0]])

        rng = np.random.default_rng(1)
        src = rng.uniform(0, 1000, size=(60, 2)).astype(np.float32)
        dst = topview.project_points(src, true_H)

        # Corrupt 10 correspondences, as a moving vehicle would
        dst[:10] += rng.uniform(200, 400, size=(10, 2))

        H, status = topview.estimate_pairwise_homography(src.astype(np.float32),
                                                         dst.astype(np.float32))
        H = H / H[2, 2]

        assert np.allclose(H, true_H, atol=1e-2), "RANSAC failed to reject outliers"
        assert status[:10].sum() <= 2, "outliers were counted as inliers"

    def test_too_few_matches_returns_none(self):
        pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        H, status = topview.estimate_pairwise_homography(pts, pts)
        assert H is None and status is None


class TestCanvasGeometry:
    def test_bounds_cover_all_frames(self):
        chain = topview.HomographyChain(H1T)
        homographies = [chain.current()]
        for _ in range(5):
            homographies.append(chain.advance(forward_motion()))

        x_min, y_min, x_max, y_max = topview.topview_bounds((720, 1280), homographies)

        assert x_max > x_min and y_max > y_min

        corners = np.array([[0, 0], [1280, 0], [1280, 720], [0, 720]], dtype=np.float32)
        for H in homographies:
            pts = topview.project_points(corners, H)
            assert pts[:, 0].min() >= x_min - 1e-6
            assert pts[:, 1].min() >= y_min - 1e-6
            assert pts[:, 0].max() <= x_max + 1e-6
            assert pts[:, 1].max() <= y_max + 1e-6

    def test_translation_moves_origin(self):
        T = topview.translation_to_origin(-150.0, -80.0)
        moved = topview.project_points(np.array([[-150.0, -80.0]]), T)[0]
        assert np.allclose(moved, [0.0, 0.0])


class TestPanoramaComposition:
    def test_black_regions_do_not_overwrite(self):
        a = np.zeros((10, 10, 3), dtype=np.uint8)
        a[0:5, :] = 200                      # top half filled

        b = np.zeros((10, 10, 3), dtype=np.uint8)
        b[5:10, :] = 100                     # bottom half filled

        out = topview.compose_panorama([a, b], (10, 10))

        assert (out[0:5, :] == 200).all(), "earlier content was erased by black pixels"
        assert (out[5:10, :] == 100).all()

    def test_later_frame_wins_on_overlap(self):
        a = np.full((6, 6, 3), 50, dtype=np.uint8)
        b = np.full((6, 6, 3), 90, dtype=np.uint8)
        out = topview.compose_panorama([a, b], (6, 6))
        assert (out == 90).all()
