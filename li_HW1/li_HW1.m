function intersection_points = circle_intersections(a, d, e, f)
    % Define the conic matrix C for the circle
    C = [a, 0, d/2;
         0, a, e/2;
         d/2, e/2, f];

    % Define the line at infinity
    m1 = [1; 0; 0];
    m2 = [0; 1; 0];

    % Solve the quadratic equation m^T * C * m = 0
    % Parametric representation: m = m1 + λ * m2
    % => (m1 + λ * m2)^T * C * (m1 + λ * m2) = 0
    
    % Compute the coefficients of the quadratic equation in λ
    A = m2' * C * m2;
    B = 2 * (m1' * C * m2);
    C_ = m1' * C * m1;

    % Solve for λ
    lambda = roots([A, B, C_]);

    % Compute the intersection points
    intersection_points = [m1 + lambda(1) * m2, m1 + lambda(2) * m2];
end

a = -2;
d = 4;
e = -1;
f = 2;

intersection_points = circle_intersections(a, d, e, f);
disp(intersection_points);