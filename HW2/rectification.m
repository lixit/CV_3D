function main()
    % Read the image
    img_path = 'UCF SU.jpg';

    outim = affine_rect(img_path);

    % Save the transformed image
    affine_rect_filepath = strcat(img_path(1:end-4), '_affine_rect.jpg');
    imwrite(outim, affine_rect_filepath);

    outim2 = metric_rect_2_perpendicular_lines(affine_rect_filepath);
    % outim2 = metric_rect_ellipse(affine_rect_filepath, l5);

    % Save the metric rectified image
    metric_rect_filepath = strcat(img_path(1:end-4), '_metric_rect.jpg');
    imwrite(outim2, metric_rect_filepath);
end

main();


function outim = affine_rect(img_path)
    im = imread(img_path);

    points_8 = getPoints(im, 8, 'Please select 2 pairs of parallel lines on the image');
    l5 = findLineAtInfinity(points_8);

    % Let the point transformation be H
    H_A = [1 0 0; 0 1 0; 0 0 1];
    H = H_A * [1 0 0; 0 1 0; l5(1) l5(2) l5(3)];

    % Call the TransformImage function
    outim = Transform_Image2(im, H);

    show_2_im(im, outim);
end

function outim = metric_rect_2_perpendicular_lines(img_path)
    im2 = imread(img_path);

    points_8 = getPoints(im2, 8, 'Please select 2 pairs of perpendicular lines on the image');
    lines_4 = getLines(points_8);

    % Get the orthogonal lines and compute the vector s = KK'
    % s = [s11, s12, s22]
    s = getS(lines_4);
    
    % Construct the matrix S from the vector s
    S = [s(1), s(2); s(2), s(3)];
    
    % Perform Cholesky decomposition to get the upper triangular matrix K
    K = chol(S, 'upper');
    
    % Compute the inverse of K
    K_inv = inv(K);
    
    % Construct the homography matrix H
    H = [K_inv, [0; 0]; 0, 0, 1];
    
    % Apply the homography to the image
    outim = Transform_Image2(im2, H);
    
    show_2_im(im2, outim);
end


function outim = metric_rect_ellipse(path, l_inf)
    % Read the image
    im = imread('UCF SU.jpg');

    % Show a prompt on the screen
    h = msgbox('Please select 5 points on the image', 'Instructions');
    uiwait(h); % Wait for the user to acknowledge the message box

    points_5 = getPoints(im, 5);
    disp('Coordinates of the five points:');
    disp(points_5);

    H = ellipseRectification(points_5, l_inf);

    % Call the TransformImage function
    im2 = imread('UCF SU.jpg');
    outim = Transform_Image2(im2, H);

    show_2_im(im2, outim);
end

% Get n points from the user, output is a n x 3 matrix
function points = getPoints(img, n, prompt)
    imshow(img);
    title(sprintf('%s. Click on %d points on the image',prompt, n));
    
    % Initialize an array to store the coordinates
    points =  zeros(n, 3);
    
    % Loop to get four points from the user
    for i = 1:n
        [x, y] = ginput(1); % Get one point from the user
        points(i, :) = [x, y, 1]; % Store the coordinates
        hold on; % Keep the image displayed
        plot(x, y, 'r+', 'MarkerSize', 10, 'LineWidth', 2); % Mark the point
        % Connect the points on a pair basis
        if mod(i, 2) == 0
            plot([points(i-1, 1), points(i, 1)], [points(i-1, 2), points(i, 2)], 'g-', 'LineWidth', 2);
        end
    end
    
    hold off; % Release the hold on the image
    pause(1); % Pause for 1 second
    close; % Close the image window
end

% Get n lines from 2*n points
function lines = getLines(points)

    % Number of lines to be computed
    n = size(points, 1) / 2;
    
    % Initialize the lines matrix
    lines = zeros(n, 3);
    
    % Compute the lines from pairs of points
    for i = 1:n
        lines(i, :) = cross(points(2*i-1, :), points(2*i, :));
    end
end

% Given 8 points(that is 4 lines), find the line at infinity
function l5 = findLineAtInfinity(points)
    
    % Get the 4 lines from the 8 points
    lines = getLines(points);
    
    % 2 points at infinity
    m1 = cross(lines(1, :), lines(2, :));
    m2 = cross(lines(3, :), lines(4, :));

    % Normalize 2 points
    if abs(m1(3)) > eps
        m1 = m1 / m1(3);
    end
    if abs(m2(3)) > eps
        m2 = m2 / m2(3);
    end
    
    % the line at infinity
    l5 = cross(m1, m2);
end

% This function takes as input an image im and a 3x3 homography H to return
% the transformed image outim
function outim = Transform_Image(im, H)
    % MAKETFORM('PROJECTIVE',A) is not recommended. Use PROJECTIVE2D instead.
    tform = maketform('projective',H');
    % Next line returns the x and y coordinates of the bounding box of the 
    % transformed image through H
    [boxx, boxy]=tformfwd(tform, [1 1 size(im,2) size(im,2)], [1 size(im,1) 1 size(im,1)]);
    % Find the minimum and maximum x and y coordinates of the bounding box
    minx=min(boxx); maxx=max(boxx);
    miny=min(boxy); maxy=max(boxy);
    % 'imtransform' is not recommended. With appropriate code changes, use 'imwarp' instead.
    outim =imtransform(im,tform,'XData',[minx maxx],'YData',[miny maxy],'Size',[size(im,1),round(size(im,1)*(maxx-minx)/(maxy-miny))]);

end

% This function takes as input an image im and a 3x3 homography H to return
% the transformed image outim
function outim = Transform_Image2(im, H)
    % Use projective2d instead of maketform
    tform = projective2d(H');
    
    % Calculate the output limits for the transformed image
    [xLimits, yLimits] = outputLimits(tform, [1 size(im, 2)], [1 size(im, 1)]);
    
    % Define the spatial referencing object for the output image
    outputRef = imref2d(size(im), xLimits, yLimits);
    
    % Use imwarp instead of imtransform
    outim = imwarp(im, tform, 'OutputView', outputRef);
end


% Give 2 pairs of orthogonal lines, return a 3 vector s = KK'
function s = getS(lines)

    % Normalize the lines if the third element is not close to 0
    for i = 1:size(lines, 1)
        if abs(lines(i, 3)) > eps
            lines(i, :) = lines(i, :) / lines(i, 3);
        else
            % raise an error if the third element is close to 0
            error('The third element of the line is close to 0');
        end
    end

    % Suppose the first pair of imaged orthogonal lines is l1 and l2
    l1 = lines(1, :);
    l2 = lines(2, :);
    
    % Suppose the second pair of imaged orthogonal lines is l3 and l4
    l3 = lines(3, :);
    l4 = lines(4, :);
    
    % Formulate the constraints for orthogonality
    % (l1 * m1, l1 * m2 + l2 * m1, l2 * m2) * S = 0
    % For the first pair (l1, l2)
    A1 = [l1(1) * l2(1), l1(1) * l2(2) + l1(2) * l2(1), l1(2) * l2(2)];
    
    % For the second pair (l3, l4)
    A2 = [l3(1) * l4(1), l3(1) * l4(2) + l3(2) * l4(1), l3(2) * l4(2)];
    
    % Stack these constraints to form a 2x3 matrix
    A = [A1; A2];
    
    % Determine the null vector s
    [~, ~, V] = svd(A);
    s = V(:, end);
end

% Input are 5 points of an ellipse in the affinely rectified image
% Output is the homography H that rectifies the ellipse to a circle
function H = ellipseRectification(points, l_inf)
    % Get the conic coefficients matrix
    C = conicfit(points');
    
    % generate 2 points on the line l_inf_prime
    % point m lies on the line if (m)T * l = 0
    % m1 * l1 + m2 * l2 + m3 * l3 = 0
    l1 = l_inf(1);
    l2 = l_inf(2);
    l3 = l_inf(3);

    m1 = [0; -l3/l2; 1];
    m2 = [-l3/l1; 0; 1];

    % Compute the coefficients of the quadratic equation in λ
    A = m2' * C * m2;
    B = 2 * (m1' * C * m2);
    C_ = m1' * C * m1;

    % Solve for λ
    lambda = roots([A, B, C_]);

    % Compute the intersection points
    p1 = m1 + lambda(1) * m2;
    p2 = m1 + lambda(2) * m2;

    % conic dual to the circular points: c = p1 * p2' + p2 * p1'
    conic_dual = p1 * p2' + p2 * p1';

    disp('Conic Dual:');
    disp(conic_dual);
    
    % do a svd and
    [U, ~, ~] = svd(conic_dual);
    
    H = inv(U);

    disp('Homography Matrix:');
    disp(H);
end

% Show 2 images side by side
function show_2_im(im1, im2)
    fig = figure(units='normalized', OuterPosition = [0 0 1 1]); % Create a full-screen figure
    [ha, pos] = tight_subplot(1, 2, [0.001, 0.001], [0.001 0.001],[0.001 0.001]);
    axes(ha(1));
    imshow(im1);
    title('im1');
    axis off; % Remove axis

    axes(ha(2));
    imshow(im2);
    title('im2');
    axis off; % Remove axis

    % Wait for the user to close the figure window
    waitfor(fig);
end
