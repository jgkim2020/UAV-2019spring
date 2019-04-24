import matplotlib.pyplot as plt
import numpy as np

from dubins_path_planning import dubins_path_planning, plot_arrow

def main():
    # RSR
    # start_x = 0.0
    # start_y = 0.0
    # start_yaw = np.deg2rad(60.0)
    # end_x = 5.0
    # end_y = 0.0
    # end_yaw = np.deg2rad(-120.0)

    # RSL
    start_x = 0.0
    start_y = 0.0
    start_yaw = np.deg2rad(60.0)
    end_x = 5.0
    end_y = 0.0
    end_yaw = np.deg2rad(120.0)

    curvature = 1.0

    px, py, pyaw, mode, clen = dubins_path_planning(start_x, start_y, start_yaw,
                                                    end_x, end_y, end_yaw, curvature)

    plt.plot(px, py, label="final course " + "".join(mode))

    # plotting
    plot_arrow(start_x, start_y, start_yaw)
    plot_arrow(end_x, end_y, end_yaw)

    #  for (ix, iy, iyaw) in zip(px, py, pyaw):
    #  plot_arrow(ix, iy, iyaw, fc="b")

    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

if __name__ == '__main__':
    main()
