function C = conicfit(pnts)        
    %
    % This function returns a conic fit to 5 points
    %
    % pnts should be in the following format
    %           [x1  y1  1]
    %           [x2  y2  1]
    %           [x3  y3  1]
    %           [x4  y4  1]
    %           [x5  y5  1]
    %
    
    % make sure pnts is a 5x3 matrix
    if size(pnts, 1) ~= 5 || size(pnts, 2) ~= 3
        error('The input matrix should be 5x3');
    end
    
    % Extract x, y, and 1 from the input matrix
    x = pnts(:, 1);
    y = pnts(:, 2);
    ones_col = pnts(:, 3);
    
    % Construct matrix A
    A = [x.^2, x.*y, y.^2, x, y, ones_col];
    
    % Compute the null space of A
    CC = null(A);
    
    % Construct the conic matrix C
    C = [CC(1), CC(2)/2, CC(4)/2; CC(2)/2, CC(3), CC(5)/2; CC(4)/2, CC(5)/2, CC(6)];
end