import numpy as np
import matplotlib.pyplot as plt

def circle_intersections(a, d, e, f):
    # Define the conic matrix C for the circle
    C = np.array([
        [a, 0, d/2],
        [0, a, e/2],
        [d/2, e/2, f]
    ])

    # Define the points m1 and m2 on the line at infinity
    m1 = np.array([1, 0, 0])
    m2 = np.array([0, 1, 0])

    # Compute the coefficients of the quadratic equation in λ
    print(C)
    
    A = np.dot(m2.T, np.dot(C, m2))
    print(np.dot(C, m2))
    B = 2 * np.dot(m1.T, np.dot(C, m2))
    B1 = 2 * np.dot(m2.T, np.dot(C, m1))
    if B != B1:
        print("B and B1 are not equal")
    C_ = np.dot(m1.T, np.dot(C, m1))

    # Solve the quadratic equation for λ
    lambda_vals = np.roots([A, B, C_])

    # Compute the intersection points
    intersection_points = [m1 + lam * m2 for lam in lambda_vals]

    return intersection_points


def plot_conic(a, b, c, d, e, f, x_range=(-10, 10), y_range=(-10, 10), resolution=500):
    # Create a grid of points
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Compute the conic equation at each point
    Z = a * X**2 + b * X * Y + c * Y**2 + d * X + e * Y + f

    # Plot the contour where the conic equation equals zero
    plt.contour(X, Y, Z, levels=[0], colors='blue')
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Conic Section: {a}x^2 + {b}xy + {c}y^2 + {d}x + {e}y + {f} = 0')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Example usage
a = 1
d = -10
e = -10
f = 30

# Example usage
plot_conic(a=a, b=0, c=a, d=d, e=e, f=f)



intersection_points = circle_intersections(a, d, e, f)
print("Intersection points:")
for point in intersection_points:
    print(point )
    
for point in intersection_points:
    print(point / point[0] ) 
    