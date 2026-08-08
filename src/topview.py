"""Top-view projection and panorama composition.

Extracted from the original notebook with the homography chain corrected.
See README "Corrections applied" for what changed and why.
"""

import cv2
import numpy as np


def load_matches(npz_path):
    """Load SuperGlue correspondences from an .npz file.

    Returns (points_in_image0, points_in_image1) as float32 Nx2 arrays,
    keeping only keypoints that were actually matched.
    """
    npz = np.load(npz_path)
    valid = npz["matches"] > -1
    point_set1 = npz["keypoints0"][valid]
    point_set2 = npz["keypoints1"][npz["matches"][valid]]
    return point_set1.astype(np.float32), point_set2.astype(np.float32)


def estimate_pairwise_homography(src_points, dst_points,
                                 ransac_threshold=5.0, min_matches=4):
    """Estimate the homography mapping src_points onto dst_points.

    Uses RANSAC rather than the least-squares default, so a handful of bad
    correspondences (moving vehicles, off-plane structure) cannot skew the fit.

    Returns (H, inlier_mask), or (None, None) if there are too few matches or
    the estimate fails.
    """
    if len(src_points) < min_matches or len(dst_points) < min_matches:
        return None, None

    H, status = cv2.findHomography(
        src_points, dst_points, cv2.RANSAC, ransac_threshold
    )
    if H is None:
        return None, None
    return H, status


class HomographyChain:
    """Accumulates frame-to-frame homographies into a frame-n-to-top-view map.

    The original notebook computed ``hom = h1t @ H`` on every iteration, which
    rebuilt the transform from the first-frame reference each time instead of
    chaining. Every frame therefore mapped to the same top-view location.

    This class carries the running product across frames:

        cumulative = H_1 @ H_2 @ ... @ H_n
        top_view   = h1t @ cumulative
    """

    def __init__(self, reference_homography):
        # h1t: maps the first frame into the top-view / satellite plane
        self.reference = np.asarray(reference_homography, dtype=np.float64)
        self.cumulative = np.eye(3, dtype=np.float64)

    def advance(self, pairwise_homography):
        """Chain one frame-to-previous-frame homography onto the running product."""
        self.cumulative = self.cumulative @ np.asarray(
            pairwise_homography, dtype=np.float64
        )
        return self.current()

    def current(self):
        """The homography mapping the current frame into the top-view plane."""
        H = self.reference @ self.cumulative
        return H / H[2, 2]  # normalise so H[2,2] == 1


def project_points(points, homography):
    """Map Nx2 image points through a homography. Returns Nx2 float array."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, np.asarray(homography, dtype=np.float64))
    return out.reshape(-1, 2)


def warp_to_topview(frame, mask, homography, canvas_size):
    """Warp a frame and its road mask into the top-view plane.

    Returns the frame with non-road pixels zeroed out.
    """
    warped_frame = cv2.warpPerspective(frame, homography, canvas_size)
    warped_mask = cv2.warpPerspective(mask, homography, canvas_size)

    if warped_mask.ndim == 2:
        warped_mask = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)

    return cv2.bitwise_and(warped_frame, warped_mask)


def topview_bounds(frame_shape, homographies):
    """Compute the bounding box covering every frame once warped.

    Replaces the notebook's hardcoded (556, 483) canvas and the hand-tuned
    paste offsets: the extent follows from the transforms themselves.

    Returns (x_min, y_min, x_max, y_max) as floats.
    """
    h, w = frame_shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    all_pts = []
    for H in homographies:
        all_pts.append(project_points(corners, H))
    all_pts = np.concatenate(all_pts, axis=0)

    return (
        float(all_pts[:, 0].min()),
        float(all_pts[:, 1].min()),
        float(all_pts[:, 0].max()),
        float(all_pts[:, 1].max()),
    )


def translation_to_origin(x_min, y_min):
    """Translation matrix shifting a bounding box so its top-left sits at (0, 0).

    Composed on the left of each homography so nothing is warped off-canvas —
    this is what removes the need for manual x_offset / y_offset tuning.
    """
    return np.array(
        [[1.0, 0.0, -x_min],
         [0.0, 1.0, -y_min],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def compose_panorama(warped_frames, canvas_size):
    """Composite warped frames onto one canvas, newest content over black.

    Each frame's non-zero region acts as its own mask, so black (non-road)
    areas never overwrite previously placed content.
    """
    width, height = canvas_size
    panorama = np.zeros((height, width, 3), dtype=np.uint8)

    for frame in warped_frames:
        occupied = frame.any(axis=2)
        panorama[occupied] = frame[occupied]

    return panorama
