import copy
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time


def q_u_to_ksi_v(q, u=None):
    """
     - Description: coordinate transformation from q, u -> ksi, v
    
     - input
    :param q: standard states (x, y, theta, phi1, phi2)
    :type q: list of floats
    :param u: standard inputs (u1, u2)
    :type u: list of floats
     - output
    :param ksi: transformed states (ksi1, ksi2, ksi3, ksi4, ksi5)
    :type ksi: list of floats
    :param v: transformed inputs (v1, v2)
    :type v: list of floats
    """
    ksi = []
    ksi.append(q[0])
    ksi.append(q[3] - q[4])
    ksi.append(q[4])
    ksi.append(q[2] - q[3] - q[4])
    ksi.append(q[1] - 2*q[2] + 2*q[3] + q[4])
    if u is None:
        return ksi
    else:
        v = []
        v.append(u[0])
        v.append((-2*q[3] + q[4])*u[0] + u[1])
        return ksi, v


def ksi_v_to_q_u(ksi, v=None):
    """
     - Description: coordinate transformation from ksi, v -> q, u
    
     - input
    :param ksi: transformed states (ksi1, ksi2, ksi3, ksi4, ksi5)
    :type ksi: list of floats
    :param v: transformed inputs (v1, v2)
    :type v: list of floats
     - output
    :param q: standard states (x, y, theta, phi1, phi2)
    :type q: list of floats
    :param u: standard inputs (u1, u2)
    :type u: list of floats
    """
    q = []
    q.append(ksi[0])
    q.append(ksi[2] + 2*ksi[3] + ksi[4])
    q.append(ksi[1] + 2*ksi[2] + ksi[3])
    q.append(ksi[1] + ksi[2])
    q.append(ksi[2])
    if v is None:
        return q
    else:
        u = []
        u.append(v[0])
        u.append((2*q[1] + q[2])*v[0] + v[1])
        return q, u


class Carwith2Trailers():

    def __init__(self, q, j=10, omegas=[5/6, 6/7, 1], omega_scaling=0.2*np.pi):
        """
         - Description: initializer

         - input
        :param q: standard states (x, y, theta, phi1, phi2)
        :type q: list of floats
        :param j: parameter from Sussmann and Liu
        :type j: integer
        :param omegas: frequency for sinusoids
        :type omegas: list of floats
        :param omega_scaling: scaling factor for frequencies
        :type omega_scaling: float
        """
        self.q = np.array(q)
        self.gamma = np.array(q)
        self.j = j
        self.omegas = np.array(omegas)*omega_scaling
        self.time = 0.0


    def run_transformed(self, gamma, T, dt=0.01):
        """
         - Description: find ordinary (non-extended) control inputs u

         - input
        :param gamma_dot: time derivative of the desired path
        :type gamma_dot: 1darray (numpy)
         - output
        :param us: ordinary (non-extended) control inputs
        :type us: list of floats
        """
        ksi = q_u_to_ksi_v(self.q)
        ksi_des = q_u_to_ksi_v(gamma)
        
        t0 = self.time
        omega = 2*np.pi/(T/4)
        us = []

        # phase 1
        alpha = (ksi_des[0] - ksi[0])*4/T
        beta = (ksi_des[0] - ksi[0])*4/T
        while self.time < t0 + T/4:
            v1 = alpha
            v2 = beta
            q, u = ksi_v_to_q_u(ksi, [v1, v2])
            us += u
            g1 = np.array([np.cos(q[2]), np.sin(q[2]), 0.0, -np.sin(q[3]), np.sin(q[3]) - np.cos(q[3])*np.sin(q[4])])
            g2 = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
            qdot = g1*u[0] + g2*u[1]
            self.q += qdot*dt
            self.time += dt
            ksi = q_u_to_ksi_v(self.q)
    
        # phase 2
        del_ksi3 = ksi_des[2] - ksi[2]
        alpha = (abs(64*np.pi*del_ksi3/T**2))**(1/2)
        beta = -alpha if del_ksi3 < 0 else alpha
        while self.time < t0 + 2*T/4:
            t_ = self.time - t0 - T/4
            v1 = alpha*np.sin(omega*t_)
            v2 = beta*np.cos(omega*t_)
            q, u = ksi_v_to_q_u(ksi, [v1, v2])
            us += u
            g1 = np.array([np.cos(q[2]), np.sin(q[2]), 0.0, -np.sin(q[3]), np.sin(q[3]) - np.cos(q[3])*np.sin(q[4])])
            g2 = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
            qdot = g1*u[0] + g2*u[1]
            self.q += qdot*dt
            self.time += dt
            ksi = q_u_to_ksi_v(self.q)

        # phase 3
        del_ksi4 = ksi_des[3] - ksi[3]
        alpha = (abs(2048*np.pi**2/T**3*del_ksi3))**(1/3)
        beta = -alpha if del_ksi4 < 0 else alpha
        while self.time < t0 + 3*T/4:
            t_ = self.time - t0 - 2*T/4
            v1 = alpha*np.sin(omega*t_)
            v2 = beta*np.cos(2*omega*t_)
            q, u = ksi_v_to_q_u(ksi, [v1, v2])
            us += u
            g1 = np.array([np.cos(q[2]), np.sin(q[2]), 0.0, -np.sin(q[3]), np.sin(q[3]) - np.cos(q[3])*np.sin(q[4])])
            g2 = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
            qdot = g1*u[0] + g2*u[1]
            self.q += qdot*dt
            self.time += dt
            ksi = q_u_to_ksi_v(self.q)

        # phase 4
        del_ksi5 = ksi_des[4] - ksi[4]
        alpha = (abs(98304*np.pi**3/T**4*del_ksi5))**(1/4)
        beta = -alpha if del_ksi5 < 0 else alpha

        ksi_debug = ksi

        while self.time < t0 + T:
            t_ = self.time - t0 - 3*T/4
            v1 = alpha*np.sin(omega*t_)
            v2 = beta*np.cos(3*omega*t_)
            q, u = ksi_v_to_q_u(ksi, [v1, v2])
            us += u
            g1 = np.array([np.cos(q[2]), np.sin(q[2]), 0.0, -np.sin(q[3]), np.sin(q[3]) - np.cos(q[3])*np.sin(q[4])])
            g2 = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
            qdot = g1*u[0] + g2*u[1]

            # debug
            h1 = np.array([1.0, 0.0, ksi_debug[1], ksi_debug[2], ksi_debug[3]])
            h2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
            ksidot = h1*v1 + h2*v2
            ksi_debug += ksidot*dt

            self.q += qdot*dt
            self.time += dt
            ksi = q_u_to_ksi_v(self.q)

        print('debug')


    def get_inputs(self, gamma_dot):
        """
         - Description: find ordinary (non-extended) control inputs u

         - input
        :param gamma_dot: time derivative of the desired path
        :type gamma_dot: 1darray (numpy)
         - output
        :param us: ordinary (non-extended) control inputs
        :type us: list of floats
        """
        # find extended inputs vs
        q = self.gamma # desired trajectory instead of actual trajectory
        g_hats = []
        g_hats.append([np.cos(q[2]), np.sin(q[2]), 0.0, -np.sin(q[3]), np.sin(q[3]) - np.cos(q[3])*np.sin(q[4])])
        g_hats.append([0.0, 0.0, 1.0, -1.0, 0.0])
        g_hats.append([np.sin(q[2]), -np.cos(q[2]), 0.0, -np.cos(q[3]), np.cos(q[3]) + np.sin(q[3])*np.sin(q[4])]) # g3
        # g_hats.append([-np.sin(q[2]), np.cos(q[2]), 0.0, np.cos(q[3]), -np.cos(q[3]) - np.sin(q[3])*np.sin(q[4])]) # g3_hat
        g_hats.append([0.0, 0.0, 0.0, -1.0, 1.0 + np.cos(q[4])])
        g_hats.append([0.0, 0.0, 0.0, -np.cos(q[3]), 2*np.cos(q[3]) + np.cos(q[3])*np.cos(q[4])]) # g5
        # g_hats.append([0.0, 0.0, 0.0, np.cos(q[3]), -2*np.cos(q[3]) - np.cos(q[3])*np.cos(q[4])]) # g5_hat
        G_hat = np.array(g_hats).T
        vs, _, rank, _ = np.linalg.lstsq(G_hat, gamma_dot)
        if rank < 5:
            print('G_hat rank is NOT maximal!')
        # vs = np.linalg.solve(G_hat, gamma_dot)

        # find coefficients etas for inputs
        etas = np.zeros([3, 4])
        ws = self.omegas
        etas[1,0] = vs[0]
        etas[2,0] = vs[1]
        etas[1,1] = (abs(2*ws[0]*vs[2]))**(1/2)
        etas[2,1] = etas[1,1] if vs[2] < 0 else -etas[1,1]
        etas[1,2] = (abs(8*ws[1]**2*vs[3]))**(1/3)
        etas[2,2] = etas[1,2] if vs[3] > 0 else -etas[1,2]
        etas[1,3] = (abs(48*ws[2]**3*vs[4]))**(1/4)
        etas[2,3] = etas[1,3] if vs[4] < 0 else -etas[1,3]

        # find inputs us
        us = []
        j = self.j
        t = self.time
        us.append(etas[1,0] + j**(1/2)*etas[1,1]*np.sin(j*ws[0]*t) + 
            j**(2/3)*etas[1,2]*np.sin(j*ws[1]*t) + j**(3/4)*etas[1,3]*np.sin(j*ws[2]*t)
        )
        us.append(etas[2,0] + j**(1/2)*etas[2,1]*np.cos(j*ws[0]*t) + 
            j**(2/3)*etas[2,2]*np.cos(2*j*ws[1]*t) + j**(3/4)*etas[2,3]*np.cos(3*j*ws[2]*t)
        )

        return us


    def step(self, gamma_dot, us=[0.0, 0.0], dt=0.01, verbose=False):
        """
         - Description: progress the system by dt seconds

         - input
        :param us: time derivative of the desired path
        :type us: list of floats
        :param dt: elapsed time
        :type dt: float
        """
        q = self.q
        g1 = np.array([np.cos(q[2]), np.sin(q[2]), 0.0, -np.sin(q[3]), np.sin(q[3]) - np.cos(q[3])*np.sin(q[4])])
        g2 = np.array([0.0, 0.0, 1.0, -1.0, 0.0])
        qdot = g1*us[0] + g2*us[1]
        self.q += qdot*dt
        self.gamma += gamma_dot*dt
        self.time += dt

        if verbose:
            print('time: %f, state:'%(self.time), self.q)


def plot_graph(pickle_name, draw_param):
    with open(pickle_name, 'rb') as f:
        data = pickle.load(f)
    
    str_to_idx = {'x':0, 'y':1, 'theta':2, 'phi1':3, 'phi2':4}
    ts = np.array(data['timestamp'])
    qs = np.array(data['states'])
    plt.figure(1)
    plt.plot(ts, qs[:,str_to_idx[draw_param]])
    plt.xlim(0, 100)
    if draw_param is 'y':
        plt.ylim(-0.5, 1.5)
    elif draw_param is 'phi1':
        plt.ylim(-2.0, 2.0)
    plt.xlabel('time (s)')
    plt.ylabel('%s'%draw_param)
    plt.grid(True)
    plt.show()


def main_transformed_coords():
    q0 = [0.0, 1.0, 0.0, 0.0, 0.0]
    qf = [0.0]*5
    T = 10
    num_steps = 1000

    agent = Carwith2Trailers(q=q0, j=10)

    for i in range(num_steps):
        gamma = np.array(q0) + (i + 1)/num_steps*(np.array(qf) - np.array(q0))
        agent.run_transformed(gamma, T, dt=0.001)
        print(agent.q)


def main_standard_coords(j=500):
    q0 = [0.0, 1.0, 0.0, 0.0, 0.0]
    qf = [0.0]*5
    T = 100.0
    gamma_dot = (np.array(qf) - np.array(q0))/T

    agent = Carwith2Trailers(q=q0, j=j)

    ts, qs = [], []
    tick = 0
    # flag_time = 0.0
    while agent.time < T:
        # if agent.time > flag_time + 10.0:
        # if True:
        #     flag_time = agent.time
        #     gamma_dot = (np.array(qf) - np.array(agent.q))/(T - agent.time)
            # print('gamma_dot at time %f'%(agent.time), gamma_dot)
            # time.sleep(5.0)

        if tick%100 == 0:
            ts.append(agent.time)
            qs.append(agent.q.copy())
        tick += 1

        inputs = agent.get_inputs(gamma_dot)
        agent.step(gamma_dot, inputs, dt=0.0001, verbose=False)

    with open('P3_b_j%s.pickle'%(str(agent.j)), 'wb') as f:
        pickle.dump({'timestamp': ts, 'states': qs}, f)
    print('done!')

    return agent.j


if __name__ == '__main__':
    j = 500
    main_standard_coords(j)
    plot_graph('P3_b_j%s.pickle'%(str(j)), 'phi1')
    # main_transformed_coords()
