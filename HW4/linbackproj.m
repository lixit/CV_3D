% Linear triangulation method
% inputs: x1, x2, P1, P2
% outputs: X, which is the 3D coordinates of the points in the world frame
function X = linbackproj(x1, x2, P1, P2)
    % x1, x2: 2D points in non-homogeneous coordinates
    % x1: An n x 2 matrix representing n 2D points from the first image.
    % x2: An n x 2 matrix representing n 2D points from the second image.
    % P1, P2: 3x4 projection matrices for the first and second images.
    % X: A N*4 matrix representing N 3D points in homogeneous coordinates.
    
    num_points = size(x1, 1);
    X = zeros(num_points, 4);
    
    for i = 1:num_points
        A = [
            x1(i,1) * P1(3,:) - P1(1,:);
            x1(i,2) * P1(3,:) - P1(2,:);
            x2(i,1) * P2(3,:) - P2(1,:);
            x2(i,2) * P2(3,:) - P2(2,:);
        ];
        
        [~, ~, V] = svd(A);
        X(i,:) = V(:,end)';
    end
    
    % Check if any element in the fourth column of X is close to zero
    epsilon = 1e-10; % Small threshold value
    if any(abs(X(:,4)) < epsilon)
        error('Division by zero detected in the fourth column of X. Aborting operation.');
    else
        % Convert homogeneous coordinates to non-homogeneous coordinates
        X = X ./ X(:,4);
    end
end