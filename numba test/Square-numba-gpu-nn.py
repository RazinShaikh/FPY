
# coding: utf-8

# # Imports

# In[1]:


import numpy as np
import itertools
import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# import pixiedust
from numba import jit, njit, cuda, vectorize, float64, float32, guvectorize
import numba
import os
import cProfile as profile
# import cupy as cp


# # Introduction

# Consider the following equation:
# $$\Delta_p \Psi(x,y)=f(x,y)$$
# 
# $x \in [0,1],\ y \in [0,1]$ with *Dirichlet* BC: $\Psi(0,y) = 0$, $\Psi(1,y) = 0$, $\Psi(x,0) = 0$ and $\Psi(x,1) = 0$.
# 
# For this first attempt, we will take $p=2$.

# # Defining functions

# In[2]:


HN_CONST=10


# ## Arith

# In[3]:


@cuda.jit(device=True)
def g_mul_num_arr(x, arr, result):
    for i in range(arr.shape[0]):
        result[i] = x * arr[i]
    return result

@cuda.jit(device=True)
def g_mul_2(arr1, arr2, result):
    for i in range(arr1.shape[0]):
        result[i] = arr1[i] * arr2[i]
    return result

@cuda.jit(device=True)
def g_add_2(arr1, arr2, result):
    for i in range(arr1.shape[0]):
        result[i] = arr1[i] + arr2[i]
    return result

@cuda.jit(device=True)
def g_add_3(arr1, arr2, arr3, result):
    for i in range(arr1.shape[0]):
        result[i] = arr1[i] + arr2[i] + arr3[i]
    return result

#returns x-arr
@cuda.jit(device=True)
def g_sub_num_arr(x, arr, result):
    for i in range(arr.shape[0]):
        result[i] = x - arr[i]
    return result

#returns arr1-arr2
@cuda.jit(device=True)
def g_sub_2(arr1, arr2, result):
    for i in range(arr1.shape[0]):
        result[i] = arr1[i] - arr2[i]
    return result

@cuda.jit(device=True)
def g_sum(arr):
    s = 0
    for i in range(arr.shape[0]):
        s += arr[i]
    return s

@cuda.jit(device=True)
def g_pow(arr, n, result):
    for i in range(arr.shape[0]):
        result[i] = arr[i]**n
    return result


# ## Sigmoid

# Sigmoid $\sigma(x) = \frac{1}{1+e^{-x}}$ and its derrivatives.

# Sigmoid with parameter $t$
# $$ \sigma_t(x) = \frac{1}{1+e^{-tx}}$$

# In[4]:


@cuda.jit(device=True)
def sig(x, result):
    for i in range(x.shape[0]):
        result[i] = 1 / (1 + np.e**(-x[i]))
    return result

@njit
def sig_orig(x):
    return 1 / (1 + np.exp(-x))

@cuda.jit(device=True)
def sig1(x, result):
    sig_x = cuda.local.array(shape=HN_CONST, dtype=float64)
    sig_minus = cuda.local.array(shape=HN_CONST, dtype=float64)
    
    sig_x = sig(x, sig_x)
    sig_minus = g_sub_num_arr(1, sig_x, sig_minus)
    
    for i in range(HN_CONST):
        result[i] = sig_x[i] * sig_minus[i]
    return result

@njit
def sig1_orig(x):
    return sig_orig(x) * (1 - sig_orig(x))

@cuda.jit(device=True)
def sig2(x, result):
    sig_x = cuda.local.array(shape=HN_CONST, dtype=float64)
    sig_1_x = cuda.local.array(shape=HN_CONST, dtype=float64)
    
    sig_x = sig(x, sig_x)
    sig_1_x = sig1(x, sig_1_x)
    
    for i in range(x.shape[0]):
        result[i] = sig_1_x[i] - 2*sig_x[i]*sig_1_x[i]
    
    return result

@njit
def sig2_orig(x):
    return (sig1_orig(x) - 2*sig_orig(x)*sig1_orig(x))

@cuda.jit(device=True)
def sig3(x, result):
    sig_x = cuda.local.array(shape=HN_CONST, dtype=float64)
    sig_1_x = cuda.local.array(shape=HN_CONST, dtype=float64)
    sig_2_x = cuda.local.array(shape=HN_CONST, dtype=float64)

    sig_x = sig(x, sig_x)
    sig_1_x = sig1(x, sig_1_x)
    sig_2_x = sig2(x, sig_2_x)
    
    for i in range(x.shape[0]):
        result[i] = sig_2_x[i] - 2 * (sig_1_x[i]**2 + sig_x[i] * sig_2_x[i])
    
    return result

@njit
def sig3_orig(x):
    return (sig2_orig(x) - 2 * (sig1_orig(x)**2 + sig_orig(x) * sig2_orig(x)))

@cuda.jit(device=True)
def sig_pr(x, k, result):
    if k==0:
        result = sig(x, result)
    if k==1:
        result = sig1(x, result)
    if k==2:
        result = sig2(x, result)
    if k==3:
        result = sig3(x, result)
#     return result

@njit
def sig_pr_orig(x, k):
    if k==0:
        return sig_orig(x)
    if k==1:
        return sig1_orig(x)
    if k==2:
        return sig2_orig(x)
    if k==3:
        return sig3_orig(x)


# ## RHS

# The right side of the equation:  $\Delta\Psi(x,y) = 6 x (x-1) (1-2 y) + 2 y (y-1) (1-2 y)$

# In[5]:


K=10
@njit
def f(x, y):
    return K * (6*(x-1)*x*(1-2*y) + 2*(y-1)*y*(1-2*y))


# The analytic solution is given by: $\ \Psi_a(x,y) = x (1-x) y (1-y) (1-2 y)$

# In[6]:


@njit
def psi_a(x, y):
    return K * x*(1-x)*y*(1-y)*(1-2*y)


# $$\frac{\partial}{\partial x} \Psi_a = (1-2x) y (1-y) (1-2 y)$$

# In[7]:


@njit
def psi_a_dx(x,y):
    return K * (1-2*x)*y*(1-y)*(1-2*y)


# $$\frac{\partial}{\partial y} \Psi_a = x (1-x) (1-6y + 6y^2) $$

# In[8]:


@njit
def psi_a_dy(x,y):
    return K * x*(1-x)*(1-6*y+6*(y**2))


# ## Neural Network

# The output of neural network $N(x,y,\vec{p})$, where $\vec{p} = [w, u, v]$:
# $$N = \sum_i^H v_i \sigma(z_i) \text{, where } z_i = w_{i0} x + w_{i1} y + u_i$$

# In[9]:


@cuda.jit(device=True)
def z(x, y, p00, p01, p1, result):
    z_x = cuda.local.array(shape=HN_CONST, dtype=float64)
    z_y = cuda.local.array(shape=HN_CONST, dtype=float64)

    z_x = g_mul_num_arr(x, p00, z_x)
    z_y = g_mul_num_arr(y, p01, z_y)
    g_add_3(z_x, z_y, p1, result)

    return result

@njit
def z_orig(x, y, p00, p01, p1):
    z_x = x * p00
    z_y = y * p01
    z_ = z_x + z_y + p1

    return z_


@cuda.jit(device=True)
def N(x, y, p00, p01, p1, p2):
    n_res = cuda.local.array(shape=HN_CONST, dtype=float64)
    n_res = z(x, y, p00, p01, p1, n_res)
    n_res = sig(n_res, n_res)
    n_res = g_mul_2(n_res, p2, n_res)
    return g_sum(n_res)

@njit
def N_orig(x, y, p00, p01, p1, p2):
    return np.sum(sig_orig(z_orig(x, y, p00, p01, p1)) * p2)


# $$\frac{\partial^k N}{\partial x_j^k} = \sum_{i=1}^H v_i w_{ij}^k \sigma^{(k)}$$

# In[10]:


@cuda.jit(device=True)
def dN_dxj_k(x, y, p00, p01, p1, p2, j, k):
    wj = p00 if j==0 else p01
    z_ = cuda.local.array(shape=HN_CONST, dtype=float64)
    z_ = z(x, y, p00, p01, p1, z_)
    sig_pr_ = cuda.local.array(shape=HN_CONST, dtype=float64)
    sig_pr(z_, k, sig_pr_)
    
    s = 0
    for i in range(HN_CONST):
        s += p2[i] * (wj[i]**k) * sig_pr_[i]
    
    return s

@njit
def dN_dxj_k_orig(x, y, p00, p01, p1, p2, j, k):
    wj = p00 if j==0 else p01
    z_ = z_orig(x, y, p00, p01, p1)
    
    return np.sum(p2 * (wj**k) * sig_pr_orig(z_, k))


# $$\frac{\partial N}{\partial w_j} = x_j v \sigma '$$

# In[11]:


@cuda.jit(device=True)
def dN_dwj(x, y, p00, p01, p1, p2, j, result):
    xj = x if j==0 else y
    sig1_z = cuda.local.array(shape=HN_CONST, dtype=float64)
    sig1_z = z(x, y, p00, p01, p1, sig1_z)
    sig1_z = sig1(sig1_z, sig1_z)
    for i in range(HN_CONST):
        result[i] = xj * p2[i] * sig1_z[i]
    return result

@njit
def dN_dwj_orig(x, y, p00, p01, p1, p2, j):
    xj = x if j==0 else y
    z_ = z_orig(x, y, p00, p01, p1)
    return xj * p2 * sig1_orig(z_)


# $$ \frac{\partial}{\partial w_j} \frac{\partial N}{\partial x_k} = x_j v w_k \sigma'' + v_i \sigma' \quad\text{ if } j = k$$
# 
# $$ \frac{\partial}{\partial w_j} \frac{\partial N}{\partial x_k} = x_j v w_k \sigma'' \quad\text{ if } j \neq k$$

# In[12]:


@cuda.jit(device=True)
def d_dwj_dN_dxk(x, y, p00, p01, p1, p2, j, k):
    xj = x if j==0 else y
    wk = p00 if k==0 else p01
    jk = 1 if j==k else 0
    z_ = z(x, y, p00, p01, p1)
    return xj * p2 * wk * sig2(z_) + jk * p2 * sig1(z_)


# $$ \frac{\partial}{\partial w_j} \frac{\partial^2 N}{\partial x_k^2} = x_j v w_k^2 \sigma^{(3)} + 2 v w_k \sigma'' \quad\text{ if } j = k $$
# 
# $$ \frac{\partial}{\partial w_j} \frac{\partial^2 N}{\partial x_k^2} = x_j v w_k^2 \sigma^{(3)} \quad\text{ if } j \neq k $$

# In[13]:


@cuda.jit(device=True)
def d_dwj_dN2_dxk2(x, y, p00, p01, p1, p2, j, k, result):
    xj = x if j==0 else y
    wk = p00 if k==0 else p01
    jk = 1 if j==k else 0
    
    z_ = cuda.local.array(shape=HN_CONST, dtype=float64)
    z_ = z(x, y, p00, p01, p1, z_)
    sig_2 = cuda.local.array(shape=HN_CONST, dtype=float64)
    sig_2 = sig2(z_, sig_2)
    sig_3 = cuda.local.array(shape=HN_CONST, dtype=float64)
    sig_3 = sig3(z_, sig_3)
    
    for i in range(HN_CONST):
        result[i] = xj * p2[i] * (wk[i]**2) * sig_3[i] +                     jk * 2 * p2[i] * wk[i] * sig_2[i]
    
    return result

@njit
def d_dwj_dN2_dxk2_orig(x, y, p00, p01, p1, p2, j, k):
    xj = x if j==0 else y
    wk = p00 if k==0 else p01
    jk = 1 if j==k else 0
    z_ = z_orig(x, y, p00, p01, p1)
    return xj * p2 * (wk**2) * sig3_orig(z_) + jk * 2 * p2 * wk * sig2_orig(z_)


# $$ \frac{\partial}{\partial u} \frac{\partial^k}{\partial x_j^k} N = v w_j^k \sigma^{(k+1)} $$

# In[14]:


@cuda.jit(device=True)
def d_du_dkN(x, y, p00, p01, p1, p2, j, k, result):
    wj = p00 if j==0 else p01
    z_ = cuda.local.array(shape=HN_CONST, dtype=float64)
    z_ = z(x, y, p00, p01, p1, z_)
    
    sig_pr_ = cuda.local.array(shape=HN_CONST, dtype=float64)
    sig_pr(z_, k+1, sig_pr_)
    
    for i in range(HN_CONST):
        result[i] = p2[i] * (wj[i]**k) * sig_pr_[i]

    return result

@njit
def d_du_dkN_orig(x, y, p00, p01, p1, p2, j, k):
    wj = p00 if j==0 else p01
    z_ = z_orig(x, y, p00, p01, p1)
    return p2 * (wj**k) * sig_pr_orig(z_, k+1)


# $$ \frac{\partial}{\partial v} \frac{\partial^k}{\partial x_j^k} N = w_j^k \sigma^{(k)} $$

# In[15]:


@cuda.jit(device=True)
def d_dv_dkN(x, y, p00, p01, p1, p2, j, k, result):
    wj = p00 if j==0 else p01
    z_ = cuda.local.array(shape=HN_CONST, dtype=float64)
    z_ = z(x, y, p00, p01, p1, z_)
    
    sig_pr_ = cuda.local.array(shape=HN_CONST, dtype=float64)
    sig_pr(z_, k, sig_pr_)
    
    for i in range(HN_CONST):
        result[i] = (wj[i]**k) * sig_pr_[i]

    return result

@njit
def d_dv_dkN_orig(x, y, p00, p01, p1, p2, j, k):
    wj = p00 if j==0 else p01
    z_ = z_orig(x, y, p00, p01, p1)
    return (wj**k) * sig_pr_orig(z_, k)


# ## Cost function

# $$E[\vec{p}] = \sum_{i \in \hat{D}} \left\{ \frac{\partial^2 N}{\partial x^2} + \frac{\partial^2 N}{\partial y^2} - f(x,y) \right\}^2 
#            +  \sum_{i \in \partial \hat{D}} N^2$$

# In[16]:


BC=1


# In[17]:


@cuda.jit(device=True)
def error_term1(x, y, p00, p01, p1, p2):
    ans = dN_dxj_k(x, y, p00, p01, p1, p2, 0, 2) +             dN_dxj_k(x, y, p00, p01, p1, p2, 1, 2)  -  f(x, y)
    return ans

@njit
def error_term1_orig(x, y, p00, p01, p1, p2):
    ans = dN_dxj_k_orig(x, y, p00, p01, p1, p2, 0, 2) +             dN_dxj_k_orig(x, y, p00, p01, p1, p2, 1, 2)  -  f(x, y)
    return ans


# In[44]:


@njit
def cost(points, boundary_points, p00, p01, p1, p2):
    et1 = np.zeros(points.shape[0])
    et2 = np.zeros(boundary_points.shape[0])
    
    for i, pnt in enumerate(points):
        et1[i] = error_term1_orig(pnt[0], pnt[1], p00, p01, p1, p2)**2
    
    for i, pnt in enumerate(boundary_points):
        et2[i] = N_orig(pnt[0], pnt[1], p00, p01, p1, p2)**2
    
    cost = np.sum(et1) + BC*np.sum(et2)
    
    return cost


# In[19]:


# @cuda.jit
# def cost_et1(points, p00, p01, p1, p2, result):
#     pos = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
#     if pos >= points.shape[0]:
#         return
    
#     result[pos] = error_term1(points[pos][0], points[pos][1], p00, p01, p1, p2)**2

# @cuda.jit
# def cost_et2(points, p00, p01, p1, p2, result):
#     pos = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
#     if pos >= points.shape[0]:
#         return
    
#     result[pos] = N(points[pos][0], points[pos][1], p00, p01, p1, p2)**2
    
# def cost(points, boundary_points, p00, p01, p1, p2):
#     et1 = np.zeros(points.shape[0])
#     et2 = np.zeros(boundary_points.shape[0])
#     threadsperblock = 64
    
#     blockspergrid_et1 = (points.shape[0] + (threadsperblock - 1)) // threadsperblock
#     cost_et1[blockspergrid_et1, threadsperblock](points, p00, p01, p1, p2, et1)
    
#     blockspergrid_et2 = (points.shape[0] + (threadsperblock - 1)) // threadsperblock
#     cost_et2[blockspergrid_et1, threadsperblock](boundary_points, p00, p01, p1, p2, et2)
    
#     cost = np.sum(et1) + BC*np.sum(et2)
    
#     return cost


# In[20]:


def relative_err_without_points(p00, p01, p1, p2, nx=100):
    all_points = np.array(list(itertools.product(np.linspace(0, 1, nx), np.linspace(0, 1, nx))))
    return relative_err(p00, p01, p1, p2, all_points)

@njit
def relative_err(p00, p01, p1, p2, all_points):
    dOmega = 1. / len(all_points)
    
    tr1 = np.zeros(all_points.shape[0])
    tr2 = np.zeros(all_points.shape[0])
    ana1 = np.zeros(all_points.shape[0])
    ana2 = np.zeros(all_points.shape[0])
    
    for i, pnt in enumerate(all_points):
        tr1[i] = dOmega * (np.abs(N_orig(pnt[0], pnt[1], p00, p01, p1, p2) - psi_a(pnt[0],pnt[1]))**2)

    for i, pnt in enumerate(all_points):
        tr2[i] = dOmega * ((dN_dxj_k_orig(pnt[0],pnt[1],p00,p01,p1,p2,0,1)-psi_a_dx(pnt[0],pnt[1]))**2 +
                           (dN_dxj_k_orig(pnt[0],pnt[1],p00,p01,p1,p2,1,1)-psi_a_dy(pnt[0],pnt[1]))**2)

    for i, pnt in enumerate(all_points):
        ana1[i] = dOmega * (np.abs(psi_a(pnt[0],pnt[1]))**2)
        
    for i, pnt in enumerate(all_points):
        ana2[i] = dOmega* (psi_a_dx(pnt[0],pnt[1])**2 + psi_a_dy(pnt[0],pnt[1])**2)
    
    rel_err = (np.sum(tr1) + np.sum(tr2))**(1/2) / (np.sum(ana1) + np.sum(ana2))**(1/2)

    return rel_err


# # Gradients

# In[21]:


@njit
def grad_ax0_sum(t1,t2):
    return np.sum(t1, axis=0) + BC*np.sum(t2, axis=0)


# $$ \frac{\partial E[\vec{p}]}{\partial w_j} = \sum_{i \in \hat{D}} \left\{ 2 \text{ (error_term1) } \left( \frac{\partial}{\partial w_j} \frac{\partial^2 N}{\partial x^2} + \frac{\partial}{\partial w_j} \frac{\partial^2 N}{\partial y^2} \right) \right\}  +  \sum_{i \in \partial \hat{D}} 2 N \frac{\partial N}{\partial w_j}$$

# In[22]:


@njit
def dE_dwj_orig(points, boundary_points, p00, p01, p1, p2, j):
    t1 = np.zeros((points.shape[0], p00.shape[0]))
    t2 = np.zeros((boundary_points.shape[0], p00.shape[0]))
    
    for i, pnt in enumerate(points):
        t1[i] = 2 * error_term1_orig(pnt[0],pnt[1],p00, p01, p1, p2) * (
                d_dwj_dN2_dxk2_orig(pnt[0],pnt[1],p00, p01, p1, p2,j,0) + 
                d_dwj_dN2_dxk2_orig(pnt[0],pnt[1],p00, p01, p1, p2,j,1))
        
    for i, pnt in enumerate(boundary_points):
        t2[i] = 2 * N_orig(pnt[0],pnt[1],p00, p01, p1, p2) * dN_dwj_orig(pnt[0],pnt[1],p00, p01, p1, p2,j)
    
    grad = np.sum(t1, axis=0) + BC*np.sum(t2, axis=0)
    
    return grad


# In[23]:


@cuda.jit
def dE_dwj_t1(points, p00, p01, p1, p2, j, result):
    pos = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    
    if pos >= points.shape[0]:
        return
    dnx = cuda.local.array(shape=HN_CONST, dtype=float64)
    dnx = d_dwj_dN2_dxk2(points[pos][0], points[pos][1], p00, p01, p1, p2, j, 0, dnx)
    dny = cuda.local.array(shape=HN_CONST, dtype=float64)
    dny = d_dwj_dN2_dxk2(points[pos][0], points[pos][1], p00, p01, p1, p2, j, 1, dny)
    et1 = error_term1(points[pos][0], points[pos][1], p00, p01, p1, p2)

    t1 = g_add_2(dnx, dny, dnx)
    t1 = g_mul_num_arr(2*et1, dnx, t1)
    for i in range(HN_CONST):
        result[pos][i] = t1[i]


# In[24]:


@cuda.jit
def dE_dwj_t2(points, p00, p01, p1, p2, j, result):
    pos = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    
    if pos >= points.shape[0]:
        return
    n_ = N(points[pos][0], points[pos][1], p00, p01, p1, p2)
    dn_dwj_ = cuda.local.array(shape=HN_CONST, dtype=float64)
    dn_dwj_ = dN_dwj(points[pos][0], points[pos][1], p00, p01, p1, p2, j, dn_dwj_)
    for i in range(HN_CONST):
        result[pos][i] = 2 * n_ * dn_dwj_[i]


# In[25]:


def dE_dwj(points, boundary_points, p00, p01, p1, p2, j):
    t1 = np.zeros((points.shape[0], p00.shape[0]))
    t2 = np.zeros((boundary_points.shape[0], p00.shape[0]))
    threadsperblock = 64
    
    blockspergrid_t1 = (points.shape[0] + (threadsperblock - 1)) // threadsperblock
    dE_dwj_t1[blockspergrid_t1, threadsperblock](points, p00, p01, p1, p2, j, t1)
    
    blockspergrid_t2 = (points.shape[0] + (threadsperblock - 1)) // threadsperblock
    dE_dwj_t2[blockspergrid_t2, threadsperblock](boundary_points, p00, p01, p1, p2, j, t2)
    
    grad = grad_ax0_sum(t1,t2)
    
    return grad


# $$ \frac{\partial E[\vec{p}]}{\partial u} = \sum_{i \in \hat{D}} \left\{ 2 \text{ (error_term1) } \left( \frac{\partial}{\partial u} \frac{\partial^2 N}{\partial x^2} + \frac{\partial}{\partial u} \frac{\partial^2 N}{\partial y^2} \right) \right\} +  \sum_{i \in \partial \hat{D}} 2 N \frac{\partial N}{\partial u}$$

# In[26]:


@njit
def dE_du_orig(points, boundary_points, p00, p01, p1, p2):
    t1 = np.zeros((points.shape[0], p1.shape[0]))
    t2 = np.zeros((boundary_points.shape[0], p1.shape[0]))
    
    for i, pnt in enumerate(points):
        t1[i] = 2 * error_term1_orig(pnt[0],pnt[1],p00, p01, p1, p2) *                 (d_du_dkN_orig(pnt[0],pnt[1],p00, p01, p1, p2,0,2) +                  d_du_dkN_orig(pnt[0],pnt[1],p00, p01, p1, p2,1,2))
        
    for i, pnt in enumerate(boundary_points):
        t2[i] = 2 * N_orig(pnt[0],pnt[1],p00, p01, p1, p2) * d_du_dkN_orig(pnt[0],pnt[1],p00, p01, p1, p2,0,0)
    
    grad = np.sum(t1, axis=0) + BC*np.sum(t2, axis=0)
    
    return grad


# In[27]:


@cuda.jit
def dE_du_t1(points, p00, p01, p1, p2, result):
    pos = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    
    if pos >= points.shape[0]:
        return
    dnx = cuda.local.array(shape=HN_CONST, dtype=float64)
    dnx = d_du_dkN(points[pos][0], points[pos][1], p00, p01, p1, p2, 0, 2, dnx)
    dny = cuda.local.array(shape=HN_CONST, dtype=float64)
    dny = d_du_dkN(points[pos][0], points[pos][1], p00, p01, p1, p2, 1, 2, dny)
    et1 = error_term1(points[pos][0], points[pos][1], p00, p01, p1, p2)

    t1 = g_add_2(dnx, dny, dnx)
    t1 = g_mul_num_arr(2*et1, dnx, t1)
    for i in range(HN_CONST):
        result[pos][i] = t1[i]


# In[28]:


@cuda.jit
def dE_du_t2(points, p00, p01, p1, p2, result):
    pos = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    
    if pos >= points.shape[0]:
        return
    
    n_ = N(points[pos][0], points[pos][1], p00, p01, p1, p2)
    dn_du_ = cuda.local.array(shape=HN_CONST, dtype=float64)
    dn_du_ = d_du_dkN(points[pos][0], points[pos][1], p00, p01, p1, p2, 0, 0, dn_du_)
    for i in range(HN_CONST):
        result[pos][i] = 2 * n_ * dn_du_[i]


# In[29]:


def dE_du(points, boundary_points, p00, p01, p1, p2):
    t1 = np.zeros((points.shape[0], p00.shape[0]))
    t2 = np.zeros((boundary_points.shape[0], p00.shape[0]))
    threadsperblock = 64
    
    blockspergrid_t1 = (points.shape[0] + (threadsperblock - 1)) // threadsperblock
    dE_du_t1[blockspergrid_t1, threadsperblock](points, p00, p01, p1, p2, t1)
    
    blockspergrid_t2 = (points.shape[0] + (threadsperblock - 1)) // threadsperblock
    dE_du_t2[blockspergrid_t2, threadsperblock](boundary_points, p00, p01, p1, p2, t2)
    
    grad = grad_ax0_sum(t1,t2)
    
    return grad


# $$ \frac{\partial E[\vec{p}]}{\partial v} = \sum_{i \in \hat{D}} \left\{ 2 \text{ (error_term1) } \left( \frac{\partial}{\partial v} \frac{\partial^2 N}{\partial x^2} + \frac{\partial}{\partial v} \frac{\partial^2 N}{\partial y^2} \right) \right\}  +  \sum_{i \in \partial \hat{D}} 2 N \frac{\partial N}{\partial v}$$

# In[30]:


@njit
def dE_dv_orig(points, boundary_points, p00, p01, p1, p2):
    t1 = np.zeros((points.shape[0], p2.shape[0]))
    t2 = np.zeros((boundary_points.shape[0], p2.shape[0]))
    
    for i, pnt in enumerate(points):
        t1[i] = 2 * error_term1_orig(pnt[0],pnt[1],p00, p01, p1, p2) *                 (d_dv_dkN_orig(pnt[0],pnt[1],p00, p01, p1, p2,0,2) +                  d_dv_dkN_orig(pnt[0],pnt[1],p00, p01, p1, p2,1,2))
        
    for i, pnt in enumerate(boundary_points):
        t2[i] = 2 * N_orig(pnt[0],pnt[1],p00, p01, p1, p2) * d_dv_dkN_orig(pnt[0],pnt[1],p00, p01, p1, p2,0,0)
    
    grad = np.sum(t1, axis=0) + BC*np.sum(t2, axis=0)
    
    return grad


# In[31]:


@cuda.jit
def dE_dv_t1(points, p00, p01, p1, p2, result):
    pos = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    
    if pos >= points.shape[0]:
        return
    dnx = cuda.local.array(shape=HN_CONST, dtype=float64)
    dnx = d_dv_dkN(points[pos][0], points[pos][1], p00, p01, p1, p2, 0, 2, dnx)
    dny = cuda.local.array(shape=HN_CONST, dtype=float64)
    dny = d_dv_dkN(points[pos][0], points[pos][1], p00, p01, p1, p2, 1, 2, dny)
    et1 = error_term1(points[pos][0], points[pos][1], p00, p01, p1, p2)

    t1 = g_add_2(dnx, dny, dnx)
    t1 = g_mul_num_arr(2*et1, dnx, t1)
    for i in range(HN_CONST):
        result[pos][i] = t1[i]


# In[32]:


@cuda.jit
def dE_dv_t2(points, p00, p01, p1, p2, result):
    pos = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    
    if pos >= points.shape[0]:
        return
    
    n_ = N(points[pos][0], points[pos][1], p00, p01, p1, p2)
    dn_dv_ = cuda.local.array(shape=HN_CONST, dtype=float64)
    dn_dv_ = d_dv_dkN(points[pos][0], points[pos][1], p00, p01, p1, p2, 0, 0, dn_dv_)
    for i in range(HN_CONST):
        result[pos][i] = 2 * n_ * dn_dv_[i]


# In[33]:


def dE_dv(points, boundary_points, p00, p01, p1, p2):
    t1 = np.zeros((points.shape[0], p00.shape[0]))
    t2 = np.zeros((boundary_points.shape[0], p00.shape[0]))
    threadsperblock = 64
    
    blockspergrid_t1 = (points.shape[0] + (threadsperblock - 1)) // threadsperblock
    dE_dv_t1[blockspergrid_t1, threadsperblock](points, p00, p01, p1, p2, t1)
    
    blockspergrid_t2 = (points.shape[0] + (threadsperblock - 1)) // threadsperblock
    dE_dv_t2[blockspergrid_t2, threadsperblock](boundary_points, p00, p01, p1, p2, t2)
    
    grad = grad_ax0_sum(t1,t2)
    
    return grad


# # NN class

# In[34]:


# @njit
def get_mini_batches(points, boundary_points, batch_size):
    np.random.shuffle(points)
    np.random.shuffle(boundary_points)
    
    no_of_splits = np.ceil( (len(points) + len(boundary_points)) / batch_size)

    mini_batch_points = np.array_split(points, no_of_splits)
    mini_batch_boundary_points = np.array_split(boundary_points, no_of_splits)
    
    return mini_batch_points, mini_batch_boundary_points


# In[35]:


class NNTrain:
    def __init__(self, nx=10, bx=10, hidden_nodes=10, alpha=0.01, batch_size=50,
                 beta=0.9, update_interval=50, if_rel_err=False, 
                 output_file='output/output.csv'):
        
        self.output_file = output_file
        self.training_started = False
        self.nx = nx
        self.hidden_nodes = hidden_nodes
        self.alpha = alpha
        self.batch_size = batch_size
        self.beta = beta
        self.update_interval = update_interval
        self.boundary_points = np.array(list(set(list(itertools.product([0, 1], np.linspace(0,1,bx))) +
                                        list(itertools.product(np.linspace(0,1,bx), [0, 1])))))
        pnts = list(itertools.product(np.linspace(0, 1, nx), np.linspace(0, 1, nx)))
        self.points = np.array([(x,y) for x,y in pnts if (x not in [0,1] and y not in [0,1])])
        self.cost_rate = []
        self.if_rel_err = if_rel_err
        if self.if_rel_err:
            self.rel_err = []
#         self.p = np.array([np.random.randn(2,hidden_nodes),
#                            np.random.randn(hidden_nodes),
#                            np.random.randn(hidden_nodes)])
        self.p00 = np.random.randn(hidden_nodes)
        self.p01 = np.random.randn(hidden_nodes)
        self.p1 = np.random.randn(hidden_nodes)
        self.p2 = np.random.randn(hidden_nodes)
        self.m_t = np.array([np.zeros(hidden_nodes),
                             np.zeros(hidden_nodes),
                             np.zeros(hidden_nodes),
                             np.zeros(hidden_nodes)])


    def sgd_mt(self, w, g_t, theta_0):
        #gradient descent with momentum
        self.m_t[w] = self.beta * self.m_t[w] + (1-self.beta) * g_t
        theta_0 = theta_0 - (self.alpha*self.m_t[w])
            
        return theta_0
        

    def train(self, itr=1000):
        self.training_started=True
        
        start=len(self.cost_rate)-1
        if start<1:
            start+=1
            self.cost_rate.append(cost(self.points,self.boundary_points,self.p00, self.p01, self.p1, self.p2))
            if self.if_rel_err:
                self.rel_err.append(relative_err(self.p00, self.p01, self.p1, self.p2, 
                                                 all_points=np.vstack([self.points, self.boundary_points])))

        i = start
        while i < start+itr:
            mini_batch_points, mini_batch_boundary = get_mini_batches(self.points, 
                                                                      self.boundary_points, self.batch_size)

            for mini_point, mini_boundary in zip(mini_batch_points, mini_batch_boundary):
#                 mini_point = list(mini_point)
#                 mini_boundary = list(mini_boundary)

                g_w0 = dE_dwj(mini_point, mini_boundary, self.p00, self.p01, self.p1, self.p2, 0)
                g_w1 = dE_dwj(mini_point, mini_boundary, self.p00, self.p01, self.p1, self.p2, 1)
                g_u = dE_du(mini_point, mini_boundary, self.p00, self.p01, self.p1, self.p2)
                g_v = dE_dv(mini_point, mini_boundary, self.p00, self.p01, self.p1, self.p2)

                self.pp00 = self.sgd_mt(0, g_w0, self.p00)
                self.p01 = self.sgd_mt(1, g_w1, self.p01)
                self.p1 = self.sgd_mt(2, g_u, self.p1)
                self.p2 = self.sgd_mt(3, g_v, self.p2)

            self.cost_rate.append(cost(self.points,self.boundary_points,self.p00, self.p01, self.p1, self.p2))
            cost_diff = self.cost_rate[i]-self.cost_rate[i+1]
            if self.if_rel_err:
                self.rel_err.append(relative_err(self.p00, self.p01, self.p1, self.p2, 
                                                 all_points=np.vstack([self.points, self.boundary_points])))
                rel_diff = self.rel_err[i]-self.rel_err[i+1]

            i+=1
                
                
    def save_result(self, output_name=''):
        timestr = time.strftime("%Y%m%d-%H%M")
        np.savez('output/'+ timestr + '_' + output_name +'_nn_params.npz', self.p)
        np.savez('output/'+ timestr + '_' + output_name +'_cost_rate.npz', self.cost_rate)
        if self.if_rel_err:
            np.savez('output/'+ timestr + '_' + output_name +'_rel_err.npz', self.rel_err)


K=10
a = NNTrain(nx=60, bx=300, hidden_nodes=HN_CONST, alpha=1e-4, batch_size=50, if_rel_err=True)


a.train(10)

