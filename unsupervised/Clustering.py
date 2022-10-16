import matplotlib.pyplot as plt

import numpy as np

# Dataset with coordinates of stores
points = np.array([[5, 3], [10, 15], [15, 12], [24, 10], [30, 45], [85, 70], [71, 80], [60, 78], [55, 52], [80, 91]])

# plt.scatter(points[:, 0], points[:, 1])
# plt.title('Stores` coordinates')

# plt.show()

# Choosing centroids for the groups
bp1 = points[0]
bp2 = points[1]


def assigns_points_to_groups(p1, p2):

    # Calculating Euclidean distances between points and the centroids
    points_in_g1 = []
    points_in_g2 = []
    group = []

    for p in points:
        x1, y1 = p[0], p[1]
        euclidean_distance_g1 = np.sqrt((p1[0] - x1) ** 2 + (p1[1] - y1) ** 2)
        euclidean_distance_g2 = np.sqrt((p2[0] - x1) ** 2 + (p2[1] - y1) ** 2)

        if euclidean_distance_g1 < euclidean_distance_g2:
            points_in_g1.append(p)
            group.append('1')
        else:
            points_in_g2.append(p)
            group.append('2')

    print('Group 1 points: %s \nGroup 2 points: %s \nGroup: %s \n' % (points_in_g1, points_in_g2, group))

    plt.scatter(points[:, 0], points[:, 1], c=[group])
    plt.title('Dividing points into groups')
    plt.show()

    # Calculating the mean for the centroids
    g1_center = np.array(points_in_g1)[:, 0].mean(), np.array(points_in_g1)[:, 1].mean()
    g2_center = np.array(points_in_g2)[:, 0].mean(), np.array(points_in_g2)[:, 1].mean()

    print('Mean values for basis points: %s %s\n' % (g1_center, g2_center))


assigns_points_to_groups(bp1, bp2)
assigns_points_to_groups([5, 3], [47.7, 50.3])
assigns_points_to_groups([13.5, 10.0], [63.5, 69.33])