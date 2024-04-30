import numpy as np
import matplotlib.pyplot as plt


def plot_hyperplane(point, hyperplane_coefficients):
    """
    Plot a hyperplane and a point in 3D space.

    Parameters:
        point: numpy array representing the coordinates of the point.
        hyperplane_coefficients: list or numpy array representing the coefficients of the hyperplane equation.
                                 The last element of the array is the bias term.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the grid for plotting
    x_min, x_max = -10, 10
    y_min, y_max = -10, 10
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))

    # Calculate corresponding z values for the hyperplane
    normal_vector = hyperplane_coefficients[:-1]
    bias = hyperplane_coefficients[-1]
    z = (-normal_vector[0] * xx - normal_vector[1] * yy - bias) / normal_vector[2]

    # Plot the hyperplane
    ax.plot_surface(xx, yy, z, alpha=0.5)

    # Plot the point
    ax.scatter(point[0], point[1], point[2], color='red', label='Point')

    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()

def distance_point_hyperplane(point, hyperplane_coefficients):
    """
    Calculate the distance between a point and a hyperplane in n-dimensional space.

    Parameters:
        point: numpy array representing the coordinates of the point.
        hyperplane_coefficients: list or numpy array representing the coefficients of the hyperplane equation.
                                 The last element of the array is the bias term.

    Returns:
        distance: Distance between the point and the hyperplane.
    """
    # Extract coefficients of the hyperplane equation
    normal_vector = hyperplane_coefficients[:-1]  # Coefficients of the normal vector
    bias = hyperplane_coefficients[-1]            # Bias term

    # Calculate the distance between the point and the hyperplane
    numerator = np.dot(normal_vector, point) + bias  # |A*x + B*y + C*z + ... + D|
    denominator = np.linalg.norm(normal_vector)             # Length of the normal vector (|A, B, C, ...|)
    distance = numerator / denominator

    return distance


if __name__ == '__main__':
    # Example usage:
    hyperplane_coefficients = np.array([1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -8])  # Example coefficients for 3D: x + 2y + 3z + 4 = 0
    #plot_hyperplane(point, hyperplane_coefficients) only works for 3d
    nlb = [4.961934217532005, -16.699169072448374, -2.81757855263168, -5.354210196980052, -11.770294001693696, -4.707831895418187, -5.148231507762388, -6.906109022636763, -6.816170646620341, -5.6059638710706965]
    nub = [8.907420808515031, -7.111515249554054, 1.046756686364746, 0.37541851950607796, -3.8524064350472056, 1.5702912247165945, 0.31570019921467174, -0.37800614319192727, -0.5451312101554124, -1.0311242370200313]
    point = np.array([nlb[0], nub[1], nub[2], nub[3], nub[4], nub[5], nub[6], nub[7], nub[8], nub[9]])
    #point = np.array([8, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # Example point in 3D: (1, 1, 1)
    distance = distance_point_hyperplane(point, hyperplane_coefficients)
    print("Distance between the point and the hyperplane:", distance)

'''
from scipy.spatial import distance
from skspatial.objects import Plane
if __name__ == '__main__':


    H = np.array([[-1, 1]])
    h = np.array([-8])
    A = pp.H_polytope(H, h)


    plt = pp.visualize([A], title=r'$A$')
    plt.show()


    # Define the hyperplane equation coefficients (Ax + By + Cz + D = 0)
    confidence_threshold = 8
    target_logit = 0
    total_output_logits = 3
    initial_point = [0 for i in range(total_output_logits)]
    initial_point[target_logit] = confidence_threshold

    points = [initial_point]
    for i in range(total_output_logits):
        if not i == target_logit:
            
            print(i)


    plane = Plane([0, 0, 0], [0, 0, 1])

    # Define a point to check
    print(plane.distance_point_signed([5, 2, 0]))
    print(plane.distance_point([5, 2, -4]))
    print(plane.distance_point_signed([5, 2, -4]))


'''