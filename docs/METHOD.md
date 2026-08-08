# Method

## 1. Frame extraction

The input video is decoded to individual frames (`road-NN.jpg`), which feed both the
segmentation network and the feature matcher.

## 2. Road segmentation

HybridNets performs road segmentation, lane segmentation, and vehicle detection in a
single forward pass. Road and lane classes are merged into one binary mask; all
non-road pixels are set to zero. Only cars are retained from the detection head.

## 3. Frame-to-frame correspondence

SuperGlue matches keypoints between each consecutive frame pair, writing
correspondences to `.npz` files. Matches are filtered to keep only points on the road
plane — a homography models a plane-to-plane mapping, so off-plane features
(buildings, signage, other vehicles) introduce error. Restricting to planar points
was the main improvement over the first iteration.

## 4. Top-view projection

A reference homography `h1t` maps the first frame into the satellite top-view plane.
It was estimated once from manually chosen correspondences between the first frame
and the satellite image.

Each subsequent frame is brought into the top-view plane by combining `h1t` with the
frame-to-frame homographies. Both the RGB frame and its road mask are warped with the
same transform; multiplying them keeps only road pixels in the output.

> See "Known issues" in the README — the implemented combination does not chain
> across frames as intended.

## 5. Vehicle localization

For each detected car, the midpoint of the bounding box base is taken as the
vehicle's contact point with the road. Since the homography is only valid on the road
plane, this is the only point on the bounding box that projects correctly. The point
is mapped through the same transform and drawn as a marker in the top-view.

## 6. Panorama composition

Warped frames are pasted into a large canvas using each frame's non-zero region as an
alpha mask, so black (non-road) areas don't overwrite previously placed content.

## 7. Satellite overlay

The finished panorama is alpha-blended over the satellite image of the same
intersection for qualitative comparison of alignment.
