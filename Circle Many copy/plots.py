import numpy as np
from matplotlib import animation, cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


RADIUS=10


def plot_iteration(arr, min_it=0, max_it=None):
    plt.figure()
    plt.plot(np.arange(len(arr[min_it:max_it])), np.array(arr[min_it:max_it]))
    plt.show()


def plot_fun(fun, dx=0.01, title=None, return_ax=False, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    r = np.arange(0, RADIUS+dx, dx)
    th = np.arange(-np.pi, np.pi+dx, dx)
    R, TH = np.meshgrid(r,th)
    X = R*np.cos(TH)
    Y = R*np.sin(TH)

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
        r_bound = np.array([RADIUS])
        th_bound = np.arange(-np.pi, np.pi+0.01, 0.01)
        R_bound, TH_bound = np.meshgrid(r_bound,th_bound)

        x_bound = R_bound*np.cos(TH_bound)
        y_bound = R_bound*np.sin(TH_bound)

        ax.plot(x_bound, y_bound, linewidth=2, c='black')
        
        for HN in range(len(p[0][1])):
            lines[HN].set_data([], [])

        ax.set_xlim(-13, 13)
        ax.set_ylim(-13, 13)
        return lines[0],

    def update(itr):
        for HN in range(len(p[itr][1])):
            x = np.linspace(-13,13,500)
            y = -(p[itr][1][HN] + p[itr][0][0][HN]*x) / p[itr][0][1][HN]
            lines[HN].set_data(x, y)

        return lines[0],

    ani = FuncAnimation(fig, update, frames=np.repeat(np.arange(len(p)), 2),
                        init_func=init, blit=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, bitrate=1800)
    ani.save(output_file, writer=writer)