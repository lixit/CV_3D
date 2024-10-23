function main1
    % Specify the folder path
    folderPath = 'mov2';

    [images, ~] = list_files_and_median(folderPath);

    show_all(images);

    tforms = get_tforms(images);

    [panorama, panoramaView] = get_panorama(tforms, images);

    blender = vision.AlphaBlender('Operation', 'Binary mask', ...
        'MaskSource', 'Input port'); 

    for i = 1:numel(tforms)
        I = imread(images{i});
        warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
        mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
        panorama = step(blender, panorama, warpedImage, mask);
    end

    figure;
    imshow(panorama);
end

main1;


function [normalized_points, T] = normalize_points(points)
    % Normalize a set of 2D points in homogeneous coordinates
    % points: A 3xN matrix representing N 2D points in homogeneous coordinates
    % normalized_points: A 3xN matrix of normalized points
    % T: The normalization transformation matrix
    
    % Ensure points are in homogeneous coordinates
    if size(points, 1) ~= 3
        error('Input points must be in homogeneous coordinates');
    end
    
    % Compute the centroid of the points
    centroid = mean(points(1:2, :), 2);
    
    % Shift the origin of the points to the centroid
    shifted_points = points;
    shifted_points(1, :) = points(1, :) - centroid(1);
    shifted_points(2, :) = points(2, :) - centroid(2);
    
    % Compute the average distance of the points from the origin
    dists = sqrt(sum(shifted_points(1:2, :).^2, 1));
    mean_dist = mean(dists);
    
    % Compute the scale factor
    scale = sqrt(2) / mean_dist;
    
    % Create the normalization transformation matrix
    T = [scale, 0, -scale * centroid(1);
         0, scale, -scale * centroid(2);
         0, 0, 1];
    
    % Normalize the points
    normalized_points = T * points;
end



% Direct Linear Transformation (DLT) algorithm   
% return the infinite homography between pairs of images.
function H = DLT_algorithm(x1, x2)
    % x1, x2: 2D points in homogeneous coordinates
    % x1: A 3xN matrix representing N 2D points in homogeneous coordinates from the first image.
    % x2: A 3xN matrix representing N 2D points in homogeneous coordinates from the second image.
    % H: A 3x3 homography matrix that maps points from the first image to the second image.
    
    % Number of points(which is the number of columns in x1)
    n = size(x1, 2);
    
    % Construct the matrix A
    % n pairs of points will give us 2n equations
    A = zeros(2*n, 9);
    for i = 1:n
        A(2*i-1, :) = [0, 0, 0, -x1(1, i), -x1(2, i), -1, x2(2, i)*x1(1, i), x2(2, i)*x1(2, i), x2(2, i)];
        A(2*i, :) = [x1(1, i), x1(2, i), 1, 0, 0, 0, -x2(1, i)*x1(1, i), -x2(1, i)*x1(2, i), -x2(1, i)];
    end
    
    % Solve the linear system A*h = 0
    [~, ~, V] = svd(A);
    h = V(:, end);
    
    % Reshape the vector h into a 3x3 matrix H
    H = reshape(h, 3, 3)';
end


function H = normalized_DLT(x1, x2)
    % x1, x2: 2D points in homogeneous coordinates
    % x1: A 3xN matrix representing N 2D points in homogeneous coordinates from the first image.
    % x2: A 3xN matrix representing N 2D points in homogeneous coordinates from the second image.
    % H: A 3x3 homography matrix that maps points from the first image to the second image.
    
    % Normalize the points
    [x1, T1] = normalize_points(x1);
    [x2, T2] = normalize_points(x2);
    
    % Compute the homography matrix using the DLT algorithm
    H = DLT_algorithm(x1, x2);

    % Denormalize the homography matrix
    H = inv(T2) * H * T1;
end


function [fileList, medianIndex] = list_files_and_median(folderPath)
    % List all files in the specified folder and return the median file according to file name
    % folderPath: Path to the folder
    % fileList: Cell array of file names
    % medianFile: The median file name according to numerical order

    % Get a list of all files in the folder
    files = dir(fullfile(folderPath, '*.jpg')); % Assuming files are .jpg
    
    % Filter out directories
    files = files(~[files.isdir]);
    
    % Get the file names
    fileList = {files.name};
    
    % Extract numerical part of the file names
    numList = zeros(1, length(fileList));
    for i = 1:length(fileList)
        % find any number of digits after an underscore
        numStr = regexp(fileList{i}, '_\d+', 'match');
        numList(i) = str2double(numStr{1}(2:end));
    end
    
    % Sort the file names based on the numerical part
    [~, sortedIndices] = sort(numList);
    sortedFileList = fileList(sortedIndices);
    
    % Find the median file
    numFiles = length(sortedFileList);
    if mod(numFiles, 2) == 1
        medianIndex = (numFiles + 1) / 2;
    else
        medianIndex = numFiles / 2;
    end
    medianFile = sortedFileList{medianIndex};
    
    % Return the sorted file list and the median file
    fileList = sortedFileList;

    % prefix the folder path to the file names
    fileList = cellfun(@(x) fullfile(folderPath, x), fileList, 'UniformOutput', false);
end



function show_all(images)
    % Read all images into an array
    imArray = cell(1, numel(images));
    for i = 1:numel(images)
        imArray{i} = imread(images{i});
    end
    
    % concatenate all images into a single image
    % and display it
    im = cat(2, imArray{:});
    figure;
    imshow(im);
end


function tform = images_2(img1, img2)
    im1 = imread(img1);
    im2 = imread(img2);
    %%
    im1 = im2double(im1);
    im1_gray = rgb2gray(im1);

    im2 = im2double(im2);
    im2_gray = rgb2gray(im2);
    %%
    pts1 = detectSURFFeatures(im1_gray);
    pts2 = detectSURFFeatures(im2_gray);

    [features1, valid_pt1] = extractFeatures(im1_gray, pts1);
    [features2, valid_pt2] = extractFeatures(im2_gray, pts2);

    indexPairs = matchFeatures(features1, features2);

    matchedPt1 = valid_pt1(indexPairs(:,1),:);
    matchedPt2 = valid_pt2(indexPairs(:,2),:);

    % figure;
    % showMatchedFeatures(im1_gray, im2_gray, matchedPt1, matchedPt2);

    [tform,inlierIdx] = estimateGeometricTransform2D(matchedPt2, matchedPt1,...
            'projective', 'Confidence', 99.9, 'MaxNumTrials', 1000);

    % % Get the inlier points
    % inlierPts1 = matchedPt1(inlierIdx, :);
    % inlierPts2 = matchedPt2(inlierIdx, :);

    % % transfer the location to homogeneous coordinates
    % inlierPts1 = inlierPts1.Location;
    % inlierPts2 = inlierPts2.Location;
    % inlierPts1 = [inlierPts1, ones(size(inlierPts1, 1), 1)]';
    % inlierPts2 = [inlierPts2, ones(size(inlierPts2, 1), 1)]';

    % H = DLT_algorithm(inlierPts1, inlierPts2);
    % tform = projective2d(H');
end


function tforms = get_tforms(images)

    n = numel(images);
    % all elements initialized to the identity matrix
    tforms(n) = projective2d(eye(3));
    
    % T(n)*T(n-1)*...*T(1)
    for i = 2:n

        tforms(i) = images_2(images{i-1}, images{i});

        % Compute T(n) * T(n-1) * ... * T(1)
        tforms(i).T = tforms(i).T * tforms(i-1).T;
    end
end


function [panorama, panoramaView] = get_panorama(tforms, images)
    n = numel(tforms);
    ImageSize = zeros(n, 2);
    for i = 1:numel(tforms)

        im = imread(images{i});
        [height,width,~] = size(im);
        ImageSize(i,:) = [height width];
    end

    for i = 1:numel(tforms)
            [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 ImageSize(i,2)], [1 ImageSize(i,1)]);
    end

    % return mean along dimension 2, aka a column vector containing the mean of each row.
    avgXLim = mean(xlim, 2);

    [~, idx] = sort(avgXLim);

    centerIdx = floor((numel(tforms)+1)/2);

    centerImageIdx = idx(centerIdx);
    Tinv = invert(tforms(centerImageIdx));

    for i = 1:numel(tforms)
        tforms(i).T = tforms(i).T * Tinv.T;
    end
    for i = 1:numel(tforms)           
        [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 ImageSize(i,2)], [1 ImageSize(i,1)]);
    end

    maxImageSize = max(ImageSize);
    
    % Find the minimum and maximum output limits
    xMin = min([1; xlim(:)]);
    xMax = max([maxImageSize(2); xlim(:)]);

    yMin = min([1; ylim(:)]);
    yMax = max([maxImageSize(1); ylim(:)]);
    
    % Width and height of panorama.
    width  = round(xMax - xMin);
    height = round(yMax - yMin);

    % Initialize the "empty" panorama.
    panorama = zeros([height width 3], 'like', imread(images{1}));
     

    % Create a 2-D spatial reference object defining the size of the panorama.
    xLimits = [xMin xMax];
    yLimits = [yMin yMax];
    panoramaView = imref2d([height width], xLimits, yLimits);
end