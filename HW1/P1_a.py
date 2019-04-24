"""

Potential Field based path planner

author: Atsushi Sakai (@Atsushi_twi)

modified by Jigang Kim

Ref:
https://www.cs.cmu.edu/~motionplanning/lecture/Chap4-Potential-Field_howie.pdf

"""

import numpy as np
import matplotlib.pyplot as plt

import copy
import time

# Parameters (potential function)
KP = 5.0  # attractive potential gain
ETA = 10000.0  # repulsive potential gain

# Parameters (map)
MINX = 0.0
MAXX = 22.0
MINY = 0.0
MAXY = 15.0

# Parameters (control rate)
RATE = 2.0

show_animation = True


def calc_potential_field(gx, gy, obstacles, reso, rr):
    xw = int(round((MAXX - MINX) / reso))
    yw = int(round((MAXY - MINY) / reso))

    # calc each potential
    pmap = [[0.0 for i in range(yw)] for i in range(xw)]

    for ix in range(xw):
        x = ix * reso + MINX

        for iy in range(yw):
            y = iy * reso + MINY
            ug = calc_attractive_potential(x, y, gx, gy)
            uo, _ = calc_repulsive_potential(x, y, obstacles, rr)
            uf = ug + uo
            pmap[ix][iy] = uf

    return pmap


def calc_attractive_potential(x, y, gx, gy):
    return 0.5 * KP * np.hypot(x - gx, y - gy)


def calc_repulsive_potential(x, y, obstacles, rr):
    # search nearest obstacle
    collision = False
    dmin, _ = mindist_to_obstacles(x, y, obstacles)

    # calc repulsive potential
    if dmin <= 1.15*rr: # within the sphere of influence
        if dmin <= 0.1*rr: # clearance less than 10% of robot size
            dmin = 0.1*rr
        if dmin < rr:
            collision = True
        
        return 0.5 * ETA * (1.0 / dmin - 1.0 / 1.15 / rr) ** 2, collision
    else:
        return 0.0, collision


def mindist_to_obstacles(x, y, obstacles): # only for rectangular obstacles
    minid = -1
    dmin = float("inf")
    for i, obstacle in enumerate(obstacles):
        xcond = obstacle['x'] < x and x < obstacle['x'] + obstacle['xlen']
        ycond = obstacle['y'] < y and y < obstacle['y'] + obstacle['ylen']
        dcand = []
        if xcond and ycond: # inside the obstacle
            dmin = 0.0
            minid = i
            return dmin, minid
        elif xcond: # perpendicular to top/bottom sides
            dcand.append(min(abs(y - obstacle['y']), abs(y - obstacle['y'] - obstacle['ylen'])))
        elif ycond: # perpendicular to left/right sides
            dcand.append(min(abs(x - obstacle['x']), abs(x - obstacle['x'] - obstacle['xlen'])))
        # four corners
        dcand.append(np.hypot(x - obstacle['x'], y - obstacle['y']))
        dcand.append(np.hypot(x - obstacle['x'] - obstacle['xlen'], y - obstacle['y']))
        dcand.append(np.hypot(x - obstacle['x'], y - obstacle['y'] - obstacle['ylen']))
        dcand.append(np.hypot(x - obstacle['x'] - obstacle['xlen'], y - obstacle['y'] - obstacle['ylen']))
        if min(dcand) < dmin:
            dmin = min(dcand)
            minid = i
    
    return dmin, minid


def get_motion_model(dubins_car):
    # dx, dy, dtheta
    theta = dubins_car.theta
    w_max = np.tan(dubins_car.max_steering)
    w_mid = w_max/2.0
    motion = [[np.cos(theta)/RATE, np.sin(theta)/RATE, 0.0],
              [(np.sin(theta + w_max/RATE) - np.sin(theta))/w_max, (np.cos(theta) - np.cos(theta + w_max/RATE))/w_max, w_max/RATE],
              [(np.sin(theta + w_mid/RATE) - np.sin(theta))/w_mid, (np.cos(theta) - np.cos(theta + w_mid/RATE))/w_mid, w_mid/RATE],
              [-(np.sin(theta - w_mid/RATE) - np.sin(theta))/w_mid, -(np.cos(theta) - np.cos(theta - w_mid/RATE))/w_mid, -w_mid/RATE],
              [-(np.sin(theta - w_max/RATE) - np.sin(theta))/w_max, -(np.cos(theta) - np.cos(theta - w_max/RATE))/w_max, -w_max/RATE]
             ]

    return motion


def potential_field_planning(dubins_car, waypoints, obstacles, reso, rr):
    pmap = calc_potential_field(waypoints[-1]['x'], waypoints[-1]['y'], obstacles, reso, rr) # for visualization purposes only!
    for n, waypoint in enumerate(waypoints):
        # calc potential field
        gx = waypoint['x']
        gy = waypoint['y']

        sx = dubins_car.x
        sy = dubins_car.y

        # search path
        d = np.hypot(sx - gx, sy - gy)
        ix = round((sx - MINX) / reso)
        iy = round((sy - MINY) / reso)
        gix = round((gx - MINX) / reso)
        giy = round((gy - MINY) / reso)

        if show_animation:
            draw_heatmap(pmap)
            plt.plot(gix, giy, "*m")
            if n == 0:
                plt.plot(ix, iy, "*k")

        rx, ry = [sx], [sy]
        while d >= 0.5*rr:
            minp = float("inf")
            motion = get_motion_model(dubins_car)

            for i, _ in enumerate(motion):
                x = dubins_car.x + motion[i][0]
                y = dubins_car.y + motion[i][1]
                theta = dubins_car.theta + motion[i][2]
                ug = calc_attractive_potential(x, y, gx, gy)
                uo, collision = calc_repulsive_potential(x, y, obstacles, rr)
                p = ug + uo
                if minp > p:
                    minp = p
                    minx = x
                    miny = y
                    mintheta = theta
                    mincollision = collision

            # update dubins_car;
            dubins_car.x = minx
            dubins_car.y = miny
            dubins_car.theta = mintheta

            d = np.hypot(minx - gx, miny - gy)
            ix = round((minx - MINX) / reso)
            iy = round((miny - MINY) / reso)
            rx.append(minx)
            ry.append(miny)

            print(minp, mincollision)

            if show_animation:
                plt.plot(ix, iy, ".r")
                plt.pause(0.1/RATE)

        print("Waypoint %d!!"%(n+1))

    return rx, ry


def draw_heatmap(data):
    data = np.array(data).T
    plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)


class DubinsCar():

    def __init__(self, x, y, theta, max_steering=40.0*np.pi/180):
        self.x = x
        self.y = y
        self.theta = theta
        self.max_steering = max_steering


def main():
    print("potential_field_planning start")
    
    waypoints = []
    waypoints.append({'x': 3.0, 'y': 5.5, 'theta': -np.pi/4})
    waypoints.append({'x': 8.5, 'y': 8.0, 'theta': np.pi/4})
    waypoints.append({'x': 11.5, 'y': 13.0, 'theta': 0})
    waypoints.append({'x': 19.5, 'y': 0.5, 'theta': -np.pi/2})
    grid_size = 0.1  # potential grid size [m]
    robot_radius = np.sqrt(2)/2.0  # robot radius [m]

    obstacles = [] # rectangular obstacles
    obstacles.append({'x': 0.0, 'y': 4.0, 'xlen': 1.0, 'ylen': 7.0}) # 1
    obstacles.append({'x': 3.0, 'y': 0.0, 'xlen': 1.0, 'ylen': 4.0}) # 2
    obstacles.append({'x': 3.0, 'y': 7.0, 'xlen': 3.0, 'ylen': 3.0}) # 3
    obstacles.append({'x': 3.0, 'y': 14.0, 'xlen': 13.0, 'ylen': 1.0}) # 4
    obstacles.append({'x': 4.0, 'y': 10.0, 'xlen': 1.0, 'ylen': 4.0}) # 5
    obstacles.append({'x': 7.0, 'y': 0.0, 'xlen': 11.0, 'ylen': 1.0}) # 6
    obstacles.append({'x': 7.0, 'y': 3.0, 'xlen': 2.0, 'ylen': 2.0}) # 7
    obstacles.append({'x': 8.0, 'y': 12.0, 'xlen': 1.0, 'ylen': 2.0}) # 8 
    obstacles.append({'x': 11.0, 'y': 1.0, 'xlen': 1.0, 'ylen': 11.0}) # 9
    obstacles.append({'x': 15.0, 'y': 4.0, 'xlen': 2.0, 'ylen': 8.0}) # 10
    obstacles.append({'x': 17.0, 'y': 7.0, 'xlen': 4.0, 'ylen': 1.0}) # 11
    obstacles.append({'x': 19.0, 'y': 14.0, 'xlen': 3.0, 'ylen': 1.0}) # 12  
    obstacles.append({'x': 21.0, 'y': 0.0, 'xlen': 1.0, 'ylen': 14.0}) # 13

    if show_animation:
        plt.grid(True)
        plt.axis("equal")

    dubins_car = DubinsCar(x=1.5, y=14.5, theta=-np.pi/2)

    # path generation
    _, _ = potential_field_planning(
        dubins_car, waypoints, obstacles, grid_size, robot_radius)

    if show_animation:
        plt.show()


if __name__ == '__main__':
    print(__file__ + " start!!")
    main()
    print(__file__ + " Done!!")
