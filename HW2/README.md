### Matlab Code to Rectify a Projective Distorted Image

Code in HW2 folder

#### Two Ways to Rectify:

1. **2 steps Rectification:**

   1. Select 2 pairs of parallel lines (4 lines in total) in the image to perform affine rectification.
   2. Select 2 pairs of perpendicular lines (4 lines in total) to perform metric rectification.

   Results are good.

    <div style="display: flex; justify-content: space-between;">
    <figure style="width: 32%;">
        <img src="pictures/tile.jpg" alt="Image 1" style="width: 100%;">
        <figcaption>Original Image</figcaption>
    </figure>
    <figure style="width: 32%;">
        <img src="pictures/tile_affine_rect.jpg" alt="Image 2" style="width: 100%;">
        <figcaption>Affine Rectification</figcaption>
    </figure>
    <figure style="width: 32%;">
        <img src="pictures/tile_metric_rect.jpg" alt="Image 3" style="width: 100%;">
        <figcaption>Metric Rectification</figcaption>
    </figure>
    </div>


2. **1 step Rectification:**
   Select 5 pairs of orthogonal lines (10 lines in total) in the image.

   Result not as good.

    <figure>
        <img src="pictures/tile_5_pairs_rect.jpg" alt="1 step Rectification" style="width: 100%;">
        <figcaption>1 step Rectification</figcaption>
    </figure>
