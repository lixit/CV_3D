% estimate the Fundamental matrix using the 8-point algorithm
% inputs: x1, x2
% outputs: F, which is the 3x3 fundamental matrix
function F = estimate_fundamental_matrix(x1, x2)
    % x1, x2: 2D points in homogeneous coordinates
    % x1: A Nx3 matrix representing N 2D points in homogeneous coordinates from the first image.
    % x2: A Nx3 matrix representing N 2D points in homogeneous coordinates from the second image.
    % F: A 3x3 fundamental matrix that maps points from the first image to the second image.

    % Normalize the points
    [x1, T1] = normalize_points(x1);
    [x2, T2] = normalize_points(x2);

    % Construct matrix A for the 8-point algorithm
    N = size(x1, 1);
    A = zeros(N, 9);
    for i = 1:N
        A(i, :) = [x1(i, 1) * x2(i, 1), x1(i, 1) * x2(i, 2), x1(i, 1), ...
                   x1(i, 2) * x2(i, 1), x1(i, 2) * x2(i, 2), x1(i, 2), ...
                   x2(i, 1), x2(i, 2), 1];
    end

    % Compute the singular value decomposition (SVD) of A
    [~, ~, V] = svd(A);

    % Extract the fundamental matrix F from the SVD
    F = reshape(V(:, end), 3, 3)';

    % Enforce the rank-2 constraint on F
    [U, S, V] = svd(F);
    S(3, 3) = 0;
    F = U * S * V';

    % Denormalize the fundamental matrix F
    F = T2' * F * T1;
end

function [x_norm, T] = normalize_points(x)
    % Normalize points to have zero mean and unit average distance from the origin
    centroid = mean(x(:, 1:2));
    x_shifted = x(:, 1:2) - centroid;
    avg_dist = mean(sqrt(sum(x_shifted.^2, 2)));
    scale = sqrt(2) / avg_dist;
    T = [scale, 0, -scale * centroid(1);
         0, scale, -scale * centroid(2);
         0, 0, 1];
    x_norm = (T * x')';
end

% Find a set of feature points in each image, and match the corresponding points to form matching point-pairs
function [x1, x2] = find_matching_points(im1, im2)
    % im1, im2: The input images
    % x1, x2: 2D points in homogeneous coordinates
    % x1: An Nx3 matrix representing N 2D points in homogeneous coordinates from the first image.
    % x2: An Nx3 matrix representing N 2D points in homogeneous coordinates from the second image.
    
    % Find feature points in each image
    points1 = detectSURFFeatures(rgb2gray(im1));
    points2 = detectSURFFeatures(rgb2gray(im2));
    
    % Extract feature descriptors
    [features1, points1] = extractFeatures(rgb2gray(im1), points1);
    [features2, points2] = extractFeatures(rgb2gray(im2), points2);
    
    % Match the features
    indexPairs = matchFeatures(features1, features2);
    
    % Get the matching points
    matchedPoints1 = points1(indexPairs(:, 1));
    matchedPoints2 = points2(indexPairs(:, 2));

    x1 = [matchedPoints1.Location, ones(size(matchedPoints1, 1), 1)];
    x2 = [matchedPoints2.Location, ones(size(matchedPoints2, 1), 1)];

    % figure;
    % showMatchedFeatures(im1, im2, x1, x2, "montage", PlotOptions=["ro","go","y--"]);
    % title("Putative Point Matches");

end

function main1
    
    folder = 'Newkuba/';
    im1=imread(strcat(folder, 'im1.png'));
    im2=imread(strcat(folder, 'im2.png'));

    % Estimate the fundamental matrix
    [matchedPoints1, matchedPoints2] = find_matching_points(im1, im2);

    [~, inliers] = estimateFundamentalMatrix(matchedPoints1(:,1:2), matchedPoints2(:,1:2), Method="MSAC", NumTrials=2000);

    inlierPts1 = matchedPoints1(inliers,:);
    inlierPts2 = matchedPoints2(inliers,:);

    F = estimate_fundamental_matrix(inlierPts1, inlierPts2);

    figure('Name', 'Point Matches After Outliers Are Removed');
    showMatchedFeatures(im1, im2, inlierPts1(:,1:2), inlierPts2(:,1:2), "montage",PlotOptions=["ro","go","y--"]);
    title("Point Matches After Outliers Are Removed");

    % Display the epipolar lines
    figure('Name', 'Epipolar Lines in Image 1');
    imshow(im1);
    hold on;
    plot(inlierPts1(:, 1), inlierPts1(:, 2), 'ro', 'MarkerSize', 5);
    title('Epipolar Lines in Image 1');
    for i = 1:size(inlierPts1, 1)
        epipolarLine = F * [inlierPts2(i, 1:2), 1]';
        x = 1:size(im1, 2);
        y = (-epipolarLine(3) - epipolarLine(1) * x) / epipolarLine(2);
        line(x, y, 'Color', 'g');
    end

    % estimate the camera projection matrices 𝐏 and 𝐏, given the intrinsic calibration matrix 𝐊
    % K is fixed
    cam1=[1037.6 0 642.2316;
         0 1043.3 387.8358; 
         0 0 1];
    cam2=[998.834 0 368.275;
         0 998.834 245.534; 
         0 0 1];
    doffs=0;
    baseline=200;
    width=1280;
    height=720;

    % Compute the camera projection matrices, using a built-in function called relativeCameraPose
    intrinsics1 = cameraIntrinsics([1037.6, 1043.3], [642.2316, 387.8358], [1280, 720]);
    intrinsics2 = cameraIntrinsics([998.834, 998.834], [368.275, 245.534], [1280, 720]);
    [R, t] = relativeCameraPose(F, intrinsics1, intrinsics2, inlierPts1(:,1:2), inlierPts2(:,1:2)); 
    P1 = cam1 * [eye(3), zeros(3, 1)];
    P2 = cam2 * [R, t'];


    % Linear triangulation
    X = linbackproj(inlierPts1, inlierPts2, P1, P2);

    % generat 3D point cloud  by a built-in Matlab function called triangulate.
    X1 = triangulate(inlierPts1(:,1:2), inlierPts2(:,1:2), P1, P2);

    % Compare the 3D points side by side
    figure('Name', 'Compare My Implemation with Built-in Function');

    % First subplot
    subplot(1, 2, 1);
    scatter3(X(:,1), X(:,2), X(:,3), 'filled');
    title('My Implementation');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    axis equal;

    % Second subplot
    subplot(1, 2, 2);
    scatter3(X1(:,1), X1(:,2), X1(:,3), 'filled');
    title('Built-in Triangulate');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    axis equal;

    % Compute the signal-to-noise ratio of my 3D point cloud by taking the one by Matlab as the ground-truth
    % Compute the mean squared error between the two point clouds
    mse = mean((X(:,1:3) - X1).^2, 'all');
    snr = 10 * log10(sum(X1.^2, 'all') / mse);
    fprintf('The mean squared error between the two point clouds is %.2f\n', mse);
    fprintf('The signal-to-noise ratio of the 3D point cloud is %.2f dB\n', snr);

end

main1;