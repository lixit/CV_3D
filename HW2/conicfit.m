function C = conicfit(pnts)        
    %
    % This function returns a conic fit to 5 points
    %
    % pnts should be in the following format
    %           [x1  x2  x3  x4  x5]
    %  pnts =   [y1  y2  y3  y4  y5]
    %           [1   1   1   1   1 ]
    %
    
    % make sure pnts is a 3x5 matrix
    if size(pnts, 1) ~= 3 || size(pnts, 2) ~= 5
        error('The input matrix should be 3x5');
    end
    A=[(pnts(1,:).^2)',(pnts(1,:).*pnts(2,:))',(pnts(2,:).^2)',pnts(1,:)',pnts(2,:)',pnts(3,:)'];
    CC=null(A);
    C=[CC(1),CC(2)/2,CC(4)/2;CC(2)/2,CC(3),CC(5)/2;CC(4)/2,CC(5)/2,CC(6)];
end