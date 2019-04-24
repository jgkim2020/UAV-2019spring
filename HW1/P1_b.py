"""
Path Planning Sample Code with RRT and Dubins path

author: AtsushiSakai(@Atsushi_twi)

modified by Jigang Kim

"""

import random
import math
import copy
import numpy as np
import dubins_path_planning
import matplotlib.pyplot as plt
import matplotlib.patches as patches

show_animation = True


class RRT():
    """
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList, randArea,
                 goalSampleRate=10, maxIter=100):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Ramdom Samping Area (rectangular)

        """
        self.start = Node(start[0], start[1], start[2])
        self.end = Node(goal[0], goal[1], goal[2])
        self.xrange = randArea['x']
        self.yrange = randArea['y']
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter
        self.obstacleList = obstacleList

    def Planning(self, animation=True):
        """
        Pathplanning

        animation: flag for animation on or off
        """

        self.nodeList = [self.start]
        for i in range(self.maxIter):
            done = False
            rnd, isgoal = self.get_random_point()
            nind = self.GetNearestListIndex(self.nodeList, rnd)

            newNode = self.steer(rnd, nind)
            #  print(newNode.cost)

            if self.CollisionCheck(newNode, self.obstacleList):
                if isgoal is True:
                    done = True
                nearinds = self.find_near_nodes(newNode)
                newNode = self.choose_parent(newNode, nearinds)
                self.nodeList.append(newNode)
                self.rewire(newNode, nearinds)
                # add intermediate nodes
                if len(newNode.path_x) > 6:
                    newNode_ = copy.deepcopy(newNode)
                    idx = int(len(newNode.path_x)/2)
                    newNode_.path_x = newNode_.path_x[0:idx+1]
                    newNode_.path_y = newNode_.path_y[0:idx+1]
                    newNode_.path_yaw = newNode_.path_yaw[0:idx+1]
                    newNode_.x = newNode_.path_x[-1]
                    newNode_.y = newNode_.path_y[-1]
                    newNode_.yaw = newNode_.path_yaw[-1]
                    deltax, deltay = [], []
                    for j in range(len(newNode_.path_x) - 1):
                        deltax.append(newNode_.path_x[j+1] - newNode_.path_x[j])
                        deltay.append(newNode_.path_y[j+1] - newNode_.path_y[j])
                    cost_ = np.sum(np.hypot(deltax, deltay))
                    newNode_.cost = cost_
                    self.nodeList.append(newNode_)

            if animation and i % 5 == 0:
                self.DrawGraph(rnd=rnd)

            print('iteration: %d, nodes: %d'%(i, len(self.nodeList)))

            # newNode = self.nodeList[-1]
            if done is True:
                print('Path to goal found!')
                break

        # generate coruse
        lastIndex = self.get_best_last_index()
        #  print(lastIndex)

        if lastIndex is None:
            return None

        path = self.gen_final_course(lastIndex)
        return path

    def choose_parent(self, newNode, nearinds):
        if not nearinds:
            return newNode

        dlist = []
        for i in nearinds:
            tNode = self.steer(newNode, i)
            if self.CollisionCheck(tNode, self.obstacleList):
                dlist.append(tNode.cost)
            else:
                dlist.append(float("inf"))

        mincost = min(dlist)
        minind = nearinds[dlist.index(mincost)]

        if mincost == float("inf"):
            print("mincost is inf")
            return newNode

        newNode = self.steer(newNode, minind)

        return newNode

    def pi_2_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def steer(self, rnd, nind):
        #  print(rnd)
        curvature = 1.0

        nearestNode = self.nodeList[nind]

        px, py, pyaw, mode, clen = dubins_path_planning.dubins_path_planning(
            nearestNode.x, nearestNode.y, nearestNode.yaw, rnd.x, rnd.y, rnd.yaw, curvature)

        newNode = copy.deepcopy(nearestNode)
        newNode.x = px[-1]
        newNode.y = py[-1]
        newNode.yaw = pyaw[-1]

        newNode.path_x = px
        newNode.path_y = py
        newNode.path_yaw = pyaw
        newNode.cost += clen
        newNode.parent = nind

        return newNode

    def get_random_point(self):

        if random.randint(0, 100) > self.goalSampleRate:
            # rnd = [random.uniform(self.xrange[0], self.xrange[1]),
            #        random.uniform(self.yrange[0], self.yrange[1]),
            #        random.uniform(-math.pi, math.pi)
            #        ]
            
            if random.randint(0,1) == 0:
                costs = [node.cost for node in self.nodeList]
                maxind = costs.index(max(costs))
                ref_node = self.nodeList[maxind]
            else:
                ref_node = self.nodeList[random.randint(0,len(self.nodeList)-1)]

            minx = max(self.xrange[0], ref_node.x - 0.50*(self.xrange[1] - self.xrange[0]))
            maxx = min(self.xrange[1], ref_node.x + 0.50*(self.xrange[1] - self.xrange[0]))
            miny = max(self.yrange[0], ref_node.y - 0.50*(self.yrange[1] - self.yrange[0]))
            maxy = min(self.yrange[1], ref_node.y + 0.50*(self.yrange[1] - self.yrange[0]))
            minyaw = ref_node.yaw - math.pi/4
            maxyaw = ref_node.yaw + math.pi/4

            isobstacle = True
            while isobstacle:
                rnd = [random.uniform(minx, maxx), random.uniform(miny, maxy), self.pi_2_pi(random.uniform(minyaw, maxyaw))]
                rnd_node = Node(rnd[0], rnd[1], rnd[2])
                rnd_node.path_x = [rnd[0]]
                rnd_node.path_y = [rnd[1]]
                rnd_node.path_yaw = [rnd[2]]
                isobstacle = not self.CollisionCheck(rnd_node, self.obstacleList)
            isgoal = False
            
        else:  # goal point sampling
            rnd = [self.end.x, self.end.y, self.end.yaw]
            isgoal = True

        node = Node(rnd[0], rnd[1], rnd[2])

        return node, isgoal

    def get_best_last_index(self):
        #  print("get_best_last_index")

        YAWTH = np.deg2rad(1.0)
        XYTH = 0.5

        goalinds = []
        for (i, node) in enumerate(self.nodeList):
            if self.calc_dist_to_goal(node.x, node.y) <= XYTH:
                goalinds.append(i)

        # angle check
        fgoalinds = []
        for i in goalinds:
            if abs(self.nodeList[i].yaw - self.end.yaw) <= YAWTH:
                fgoalinds.append(i)

        if not fgoalinds:
            return None

        mincost = min([self.nodeList[i].cost for i in fgoalinds])
        for i in fgoalinds:
            if self.nodeList[i].cost == mincost:
                return i

        return None

    def gen_final_course(self, goalind):
        path = [[self.end.x, self.end.y]]
        while self.nodeList[goalind].parent is not None:
            node = self.nodeList[goalind]
            for (ix, iy) in zip(reversed(node.path_x), reversed(node.path_y)):
                path.append([ix, iy])
            #  path.append([node.x, node.y])
            goalind = node.parent
        path.append([self.start.x, self.start.y])
        return path

    def calc_dist_to_goal(self, x, y):
        return np.linalg.norm([x - self.end.x, y - self.end.y])

    def find_near_nodes(self, newNode):
        nnode = len(self.nodeList)
        r = 50.0 * math.sqrt((math.log(nnode) / nnode))
        #  r = self.expandDis * 5.0
        dlist = [(node.x - newNode.x) ** 2 +
                 (node.y - newNode.y) ** 2 +
                 (node.yaw - newNode.yaw) ** 2
                 for node in self.nodeList]
        nearinds = [dlist.index(i) for i in dlist if i <= r ** 2]
        return nearinds

    def rewire(self, newNode, nearinds):

        nnode = len(self.nodeList)

        for i in nearinds:
            nearNode = self.nodeList[i]
            tNode = self.steer(nearNode, nnode - 1)

            obstacleOK = self.CollisionCheck(tNode, self.obstacleList)
            imporveCost = nearNode.cost > tNode.cost

            if obstacleOK and imporveCost:
                #  print("rewire")
                self.nodeList[i] = tNode

    def DrawGraph(self, rnd=None):  # pragma: no cover
        """
        Draw Graph
        """
        plt.clf()
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot(node.path_x, node.path_y, "-g")
                #  plt.plot([node.x, self.nodeList[node.parent].x], [
                #  node.y, self.nodeList[node.parent].y], "-g")

        for obstacle in pad_obstacles(self.obstacleList, padding=0.0):
            ax = plt.gca()
            rect = patches.Rectangle((obstacle['x'], obstacle['y']), obstacle['xlen'], obstacle['ylen'], 
                linewidth=None, edgecolor='b', facecolor='b'
            )
            ax.add_patch(rect)
            # plt.plot(ox, oy, "ok", ms=30 * size)

        dubins_path_planning.plot_arrow(
            self.start.x, self.start.y, self.start.yaw)
        dubins_path_planning.plot_arrow(
            self.end.x, self.end.y, self.end.yaw)

        # plt.axis([-2, 15, -2, 15])
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.01)

        #  plt.show()
        #  input()

    def GetNearestListIndex(self, nodeList, rnd):
        # dlist = [(node.x - rnd.x) ** 2 +
        #          (node.y - rnd.y) ** 2 +
        #          (node.yaw - rnd.yaw) ** 2 for node in nodeList]
        # minind = dlist.index(min(dlist))   
        clenlist = [self.steer(rnd, i).cost for i, _ in enumerate(nodeList)]
        minind = clenlist.index(min(clenlist))
        
        return minind

    def mindist_to_obstacles(self, x, y, obstacleList): # only for rectangular obstacles
        minid = -1
        dmin = float("inf")
        for i, obstacle in enumerate(obstacleList):
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

    def CollisionCheck(self, node, obstacleList, dmin=0.5):

        for (ix, iy) in zip(node.path_x, node.path_y):
            d, _ = self.mindist_to_obstacles(ix, iy, obstacleList)
            xcond = ix < self.xrange[0] or ix > self.xrange[1]
            ycond = iy < self.yrange[0] or iy > self.yrange[1]
            if d < dmin or xcond or ycond:
                return False # collision

        return True  # safe


class Node():
    """
    RRT Node
    """

    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.path_x = []
        self.path_y = []
        self.path_yaw = []
        self.cost = 0.0
        self.parent = None


def pad_obstacles(obstacles, padding=0.0):
    padded_obstacles = []
    for obstacle in obstacles:
        padded_obstacle = obstacle.copy()
        padded_obstacle['x'] -= padding
        padded_obstacle['y'] -= padding
        padded_obstacle['xlen'] += 2*padding
        padded_obstacle['ylen'] += 2*padding
        padded_obstacles.append(padded_obstacle)

    return padded_obstacles


def main():
    print("Start rrt star with dubins planning")

    # ====Search Path with RRT====
    # obstacleList = [
    #     (5, 5, 1),
    #     (3, 6, 2),
    #     (3, 8, 2),
    #     (3, 10, 2),
    #     (7, 5, 2),
    #     (9, 5, 2)
    # ]  # [x,y,size(radius)]
    obstacleList = [
        {'x': 0.0, 'y': 4.0, 'xlen': 1.0, 'ylen': 7.0},
        {'x': 3.0, 'y': 0.0, 'xlen': 1.0, 'ylen': 4.0},
        {'x': 3.0, 'y': 7.0, 'xlen': 3.0, 'ylen': 3.0},
        {'x': 3.0, 'y': 14.0, 'xlen': 13.0, 'ylen': 1.0},
        {'x': 4.0, 'y': 10.0, 'xlen': 1.0, 'ylen': 4.0},
        {'x': 7.0, 'y': 0.0, 'xlen': 11.0, 'ylen': 1.0},
        {'x': 7.0, 'y': 3.0, 'xlen': 2.0, 'ylen': 2.0},
        {'x': 8.0, 'y': 12.0, 'xlen': 1.0, 'ylen': 2.0},
        {'x': 11.0, 'y': 1.0, 'xlen': 1.0, 'ylen': 11.0},
        {'x': 15.0, 'y': 4.0, 'xlen': 2.0, 'ylen': 8.0},
        {'x': 17.0, 'y': 7.0, 'xlen': 4.0, 'ylen': 1.0},
        {'x': 19.0, 'y': 14.0, 'xlen': 3.0, 'ylen': 1.0}, 
        {'x': 21.0, 'y': 0.0, 'xlen': 1.0, 'ylen': 14.0}
    ] # rectangular obstacle

    padded_obstacleList = pad_obstacles(obstacleList, padding=np.sqrt(2)/2.0)

    # Set Initial parameters
    start = [1.5, 14.5, np.deg2rad(-90.0)]
    goal = [19.5, 0.5, np.deg2rad(-90.0)]

    rrt = RRT(start, goal, randArea={'x': [0.0, 22.0], 'y': [0.0, 15.0]}, obstacleList=obstacleList, maxIter=5000)
    path = rrt.Planning(animation=show_animation)

    # Draw final path
    if show_animation:  # pragma: no cover
        rrt.DrawGraph()
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
        plt.grid(True)
        plt.pause(0.001)

        plt.show()


if __name__ == '__main__':
    main()
