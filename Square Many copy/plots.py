import numpy as np
from matplotlib import animation, cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def plot_iteration(arr, min_it=0, max_it=None):
    plt.figure()
    plt.plot(np.arange(len(arr[min_it:max_it])), np.array(arr[min_it:max_it]))
    plt.show()


def plot_fun(fun, d_x=0.01, title=None, return_ax=False, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(0, 1.0+d_x, d_x)
    X, Y = np.meshgrid(x, y)

    zs = np.array([fun(x,y,**kwargs) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm_r)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.title.set_text(title)

    if return_ax:
        return ax
    else:
        plt.show()



def animate_p(p, output_file):
    fig, ax = plt.subplots()

    lines = []
    for HN in range(len(p[0][1])):
        ln, = plt.plot([], [])
        lines.append(ln)

    def init():
        ax.plot([0,0,1,1,0], [0,1,1,0,0], linewidth=3)
        for HN in range(len(p[0][1])):
            lines[HN].set_data([], [])

        ax.set_xlim(-1, 2)
        ax.set_ylim(-1, 2)
        return lines[0],

    def update(itr):
        for HN in range(len(p[itr][1])):
            x = np.linspace(-1,2,500)
            y = -(p[itr][1][HN] + p[itr][0][0][HN]*x) / p[itr][0][1][HN]
            lines[HN].set_data(x, y)

        return lines[0],

    ani = FuncAnimation(fig, update, frames=np.repeat(np.arange(len(p)), 2),
                        init_func=init, blit=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, bitrate=1800)
    ani.save(output_file, writer=writer)