import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

points = np.array([[-1, -1, -1],
                      [1, -1, -1 ],
                      [1, 1, -1],
                      [-1, 1, -1],
                      [-1, -1, 1],
                      [1, -1, 1 ],
                      [1, 1, 1],
                      [-1, 1, 1]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
r = [-1,1]
X, Y = np.meshgrid(r, r)
ax.plot_surface(X,Y,np.atleast_2d(1), alpha=0.5)
ax.plot_surface(X,Y,np.atleast_2d(-1), alpha=0.5)
ax.plot_surface(X,np.atleast_2d(-1),Y, alpha=0.5)
ax.plot_surface(X,np.atleast_2d(1),Y, alpha=0.5)
ax.plot_surface(np.atleast_2d(1),X,Y, alpha=0.5)
ax.plot_surface(np.atleast_2d(-1),X,Y, alpha=0.5)
ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()