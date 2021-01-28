import numpy as np
from matplotlib import animation, cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def plot_iteration(arr, min_it=0, max_it=None):
    plt.figure()
    plt.plot(np.arange(len(arr[min_it:max_it])), np.array(arr[min_it:max_it]))
    plt.show()


def plot_fun(fun, dx_r=1.5, dx_th1=0.2, dx_th2=0.3, title=None, return_ax=False, **kwargs):
    RADIUS = 10
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    r = np.arange(dx_r/2, RADIUS+dx_r, dx_r)
    th1 = np.arange(-np.pi, np.pi+dx_th1, dx_th1)
    th2 = np.arange(-np.pi, np.pi+dx_th2, dx_th2)
    R, TH1, TH2 = np.meshgrid(r,th1,th2)

    x = np.ravel(R*np.sin(TH1)*np.cos(TH2))
    y = np.ravel(R*np.sin(TH1)*np.sin(TH2))
    z = np.ravel(R*np.cos(TH1))

    cs = np.array([fun(x,y,z,**kwargs) for x,y,z in zip(x, y, z)])
    c = cs.reshape(x.shape)

#     ax.plot_surface(X, Y, Z, cmap=cm.coolwarm_r)
    img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
    fig.colorbar(img)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
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