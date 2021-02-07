import time
from itertools import product

import numpy as np

import nn_functions as nn


def get_p(nntrain):
    p = np.array([np.zeros((2, nntrain.hidden_nodes)), 
                  np.zeros(nntrain.hidden_nodes),
                  np.zeros(nntrain.hidden_nodes)], dtype=object)
    p[0][0] = nntrain.p00
    p[0][1] = nntrain.p01
    p[1] = nntrain.p1
    p[2] = nntrain.p2
    
    return p


def get_mini_batches(points, boundary_points, batch_size):
    np.random.shuffle(points)
    np.random.shuffle(boundary_points)
    
    no_of_splits = np.ceil( (len(points) + len(boundary_points)) / batch_size)

    mini_batch_points = np.array_split(points, no_of_splits)
    mini_batch_boundary_points = np.array_split(boundary_points, no_of_splits)
    
    return mini_batch_points, mini_batch_boundary_points


def get_points(dx_r, dx_th, bx):
    r = np.arange(dx_r/2, nn.RADIUS+dx_r, dx_r)
    th = np.arange(-np.pi, np.pi+dx_th, dx_th)
    R, TH = np.meshgrid(r,th)

    r_bound = np.array([nn.RADIUS])
    th_bound = np.arange(-np.pi, np.pi+bx, bx)
    R_bound, TH_bound = np.meshgrid(r_bound,th_bound)

    x = np.ravel(R*np.cos(TH))
    y = np.ravel(R*np.sin(TH))

    x_bound = np.ravel(R_bound*np.cos(TH_bound))
    y_bound = np.ravel(R_bound*np.sin(TH_bound))

    boundary_points = list(zip(x_bound, y_bound))
    points = [(a,b) for a, b in zip(x,y) if (a,b) not in boundary_points]

    return np.array(points), np.array(boundary_points)


class NNTrain:
    def __init__(self, dx_r=0.3, dx_th=0.7, bx=0.08, hidden_nodes=10, alpha=0.01, batch_size=50,
                 beta=0.9, bc=1, if_rel_err=True):
        
        self.hidden_nodes = hidden_nodes
        self.alpha = alpha
        self.batch_size = batch_size
        self.beta = beta
        self.points, self.boundary_points = get_points(dx_r,dx_th,bx)
        self.cost_rate = []
        self.if_rel_err = if_rel_err
        if self.if_rel_err:
            self.rel_err = []
        self.p00 = np.random.randn(hidden_nodes)
        self.p01 = np.random.randn(hidden_nodes)
        self.p1 = np.random.randn(hidden_nodes)
        self.p2 = np.random.randn(hidden_nodes)
        self.m_t = np.array([np.zeros(hidden_nodes),
                             np.zeros(hidden_nodes),
                             np.zeros(hidden_nodes),
                             np.zeros(hidden_nodes)])
        self.bc = bc


    def sgd_mt(self, w, g_t, theta_0):
        #gradient descent with momentum
        self.m_t[w] = self.beta * self.m_t[w] + (1-self.beta) * g_t
        theta_0 = theta_0 - (self.alpha*self.m_t[w])
        
        return theta_0
        

    def train(self, itr=1000):

        start=len(self.cost_rate)-1
        if start<1:
            start+=1
            self.cost_rate.append(nn.cost(self.points,self.boundary_points,self.p00, self.p01, self.p1, self.p2, self.bc))
            if self.if_rel_err:
                self.rel_err.append(nn.relative_err(self.p00, self.p01, self.p1, self.p2, 
                                                 all_points=np.vstack([self.points, self.boundary_points])))

        i = start
        while i < start+itr:
            mini_batch_points, mini_batch_boundary = get_mini_batches(self.points, 
                                                                      self.boundary_points, self.batch_size)

            for mini_point, mini_boundary in zip(mini_batch_points, mini_batch_boundary):

                g_w0 = nn.dE_dwj(mini_point, mini_boundary, self.p00, self.p01, self.p1, self.p2, 0, self.bc)
                g_w1 = nn.dE_dwj(mini_point, mini_boundary, self.p00, self.p01, self.p1, self.p2, 1, self.bc)
                g_u = nn.dE_du(mini_point, mini_boundary, self.p00, self.p01, self.p1, self.p2, self.bc)
                g_v = nn.dE_dv(mini_point, mini_boundary, self.p00, self.p01, self.p1, self.p2, self.bc)

                self.p00 = self.sgd_mt(0, g_w0, self.p00)
                self.p01 = self.sgd_mt(1, g_w1, self.p01)
                self.p1 = self.sgd_mt(2, g_u, self.p1)
                self.p2 = self.sgd_mt(3, g_v, self.p2)

            self.cost_rate.append(nn.cost(self.points,self.boundary_points,self.p00, self.p01, self.p1, self.p2, self.bc))
            if self.if_rel_err:
                self.rel_err.append(nn.relative_err(self.p00, self.p01, self.p1, self.p2, 
                                                 all_points=np.vstack([self.points, self.boundary_points])))

            i+=1
                
                
    def save_result(self, output_name=''):
        p = np.array([np.zeros((2, self.hidden_nodes)), 
                      np.zeros(self.hidden_nodes),
                      np.zeros(self.hidden_nodes)], dtype=object)
        p[0][0] = self.p00
        p[0][1] = self.p01
        p[1] = self.p1
        p[2] = self.p2

        timestr = time.strftime("%Y%m%d-%H%M")
        np.savez('output/'+ timestr + '_' + output_name +'_nn_params.npz', p)
        np.savez('output/'+ timestr + '_' + output_name +'_cost_rate.npz', self.cost_rate)
        np.savez('output/'+ timestr + '_' + output_name + '_momentum.npz', self.m_t)
        if self.if_rel_err:
            np.savez('output/'+ timestr + '_' + output_name +'_rel_err.npz', self.rel_err)
