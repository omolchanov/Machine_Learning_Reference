import math
import sys
import copy

import numpy as np

import matplotlib.pyplot as plt


# Configuration
np.set_printoptions(threshold=np.inf, suppress=True)
np.set_printoptions(threshold=sys.maxsize, suppress=True)


def build_terrain(x):
    """
    The function builds the terrain with sum of gaussian terrain model
    https://en.wikipedia.org/wiki/Gaussian_surface
    """

    # center, height, sigma
    peaks = [
         ((0.3, 0.4), 2, 0.1),
         ((1.0, 0.0), 1, 0.1),
         ((0.5, 0.5), 1, 0.1),
         ((0.0, 1.0), 1, 0.1),
         ((0.6, 1.0), 2, 0.1),
         ((0.6, 0.8), 2, 0.1),
         ((0.8, 0.4), 2, 0.1),
         ((0.8, 0.8), 2, 0.1),
         ((0.9, 0.4), 1, 0.1),
         ((0.4, 0.6), 1, 0.1),
         ((0.6, 0.4), 1, 0.1),
         ((1.0, 0.4), 1, 0.1),
         ((0.4, 0.2), 3, 0.1),
         ((0.0, 1.0), 1, 0.1),
    ]

    t = 0
    x = np.array(x)

    for peak in peaks:
        xp = peak[0]
        h = peak[1]
        sig = peak[2]

        d = np.array(x) - np.array(xp)
        t += h*math.exp(-d.dot(d)/sig**2)

    return t


n = 30

# linspace() function generates an array of evenly spaced numbers over a defined interval.
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)

nx = len(x)
ny = len(y)

z = np.zeros([nx, ny])
for i in range(nx):
    for j in range(ny):
        z[j, i] = build_terrain([x[i], y[j]])


X, Y = np.meshgrid(x, y)


def position(i_x):
    """
    Defines the position of a particular point
    :param i_x:
    :return:
    """
    return np.array([x[i_x[0]], y[i_x[1]]])


# Setting initial and final positions
i_x0 = (int(0.3*n), int(0.3*n))
i_xf = (int(0.7*n), int(0.9*n))
x0 = position(i_x0)
xf = position(i_xf)

print('Initial position: {0} \nFinal position: {1}'.format(x0, xf))


def plot_terrain():
    """
    Plots the terrain surface
    """

    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.contour(X, Y, z, 20)

    plt.plot(x0[0], x0[1], 'go', markersize=20)
    plt.plot(xf[0], xf[1], 'ro', markersize=20)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Terrain contour')

    ax = plt.subplot(122, projection='3d')
    ax.plot_surface(X, Y, z, cstride=1, rstride=1)

    h_start = ax.plot3D([x0[0]], [x0[1]], [build_terrain(x0)], 'go')[0]
    h_end = ax.plot3D([xf[0]], [xf[1]], [build_terrain(xf)], 'ro')[0]

    ax.view_init(elev=60, azim=-150)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title('terrain')

    ax.legend(handles=[h_start, h_end], labels=['start', 'end'])

    plt.grid()
    plt.show()


def calculate_path_cost(x0, x1):
    """
    Estimated path length
    """
    s = math.sqrt((x0[0] - x1[0])**2 + (x0[1] - x1[1])**2 + (build_terrain(x0) - build_terrain(x1))**2)
    return s


# initialize at the final point
i_x_prev = i_x = i_xf
V = np.zeros((n, n))
live = [i_x]


def in_bounds(i_x):
    """
    Bound all points to within the defined terrain
    """
    return i_x[0] >= 0 and i_x[0] < n and i_x[1] >= 0 and i_x[1] < n


# define all possible actions
possible_actions = [
    (1, 0), (-1, 0),  # left/right
    (0, 1), (0, -1),  # up/down
    (1, 1), (1, -1), (-1, -1), (-1, 1)  # NE, SE, SW, NW
]

# process all remaining moves (live nodes)
# Calculate V-scores
V_data = []
while len(live) > 0:
    new = set()
    for p in live:
        for a in possible_actions:
            pa = (p[0] - a[0], p[1] - a[1])  # move backwards so action negative
            if in_bounds(pa):
                V_new = calculate_path_cost(position(p), position(pa)) + V[p[0], p[1]]
                V_old = V[pa[0], pa[1]]
                if V_old == 0 or V_new < V_old:
                    V[pa[0], pa[1]] = V_new
                    new.add(pa)
    live = new
    V_data.append(copy.copy(V))

# Find optimal path, start at initial conditions and choose the lowest cost moving
# Gorward using Bellman's principle of optimality
p = i_x0
p_hist_opt = [p]
count = 0
while p != i_xf:
    count += 1
    pa_opt = None
    V_old = V[p[0], p[1]]
    for a in possible_actions:
        pa = (p[0] + a[0], p[1] + a[1])
        if in_bounds(pa):
            V_new = V[pa[0], pa[1]]
            if pa_opt is None or pa == i_xf or (V_new < V_opt and V_new < V_old):
                V_opt = V_new
                pa_opt = pa
        if pa == i_xf:
            break
    if count > 1000:
        break
    p = pa_opt
    p_hist_opt.append(p)

pos_hist_opt = np.array([position(p) for p in p_hist_opt])
print('\nOptimal path: ', pos_hist_opt)


def plot_optimal_path():

    # 2D Plot
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.pcolor(X, Y, V.T)

    plt.plot(pos_hist_opt[:, 0], pos_hist_opt[:, 1], '-k', linewidth=5)
    plt.plot([x0[0]], [x0[1]], 'go')
    plt.plot([xf[0]], [xf[1]], 'ro')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('optimal path and cost to go')

    # 3D Plot
    ax = plt.subplot(122, projection='3d')
    ax.plot_surface(X, Y, z, cstride=1, rstride=1, alpha=0.3)

    ax.plot3D([x0[0]], [x0[1]], [build_terrain(x0)], 'go')
    ax.plot3D([xf[0]], [xf[1]], [build_terrain(xf)], 'ro')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Optimal path, terrain')

    ax.view_init(elev=60, azim=-150)
    ax.legend(['terrain', 'start', 'end'])

    terrain_opt = [build_terrain(pos_hist_opt[i, :]) for i in range(len(pos_hist_opt[:, 0]))]
    ax.plot3D(pos_hist_opt[:, 0], pos_hist_opt[:, 1], terrain_opt, '-y', linewidth=5)

    plt.show()


plot_terrain()
plot_optimal_path()
