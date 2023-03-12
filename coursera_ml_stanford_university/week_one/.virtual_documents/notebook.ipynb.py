import pandas as pd

data = pd.DataFrame(data=[[2104, 460], [1416, 232], [1538, 315], [852, 178]], columns=['size in feet**2(x)', 'price ($) in 1000\'s(y)'])
display(data)














import numpy as np
import matplotlib.pyplot as plt

f = lambda x, y: np.sin(x) * np.cos(y)
x = np.linspace(-2, 2, 40)
y = np.linspace(-2, 2, 40)

X, Y = np.meshgrid(x, y)

F = f(X, Y)

plt.contour(X, Y, F, 15)
plt.show()




from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection='3d')


# ax.contour3D(X, Y, F)
ax.plot_surface(X, Y, F, cmap=cm.coolwarm)
plt.show()


def plotter(E, A):
    fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111, projection='3d')
    ax = plt.axes(projection='3d')


    # ax.contour3D(X, Y, F)
    ax.plot_surface(X, Y, F, cmap=cm.Blues)
    ax.view_init(elev=E, azim=A)
    plt.show()

from ipywidgets import interactive
iplot= interactive(plotter, E=(-90, 90, 5), A=(-90,90, 5))
iplot



