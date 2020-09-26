import numpy as np
import itertools
from numba import njit


#constants
T = 1


@njit
def sig(x):
    return 1 / (1 + np.exp(-T*x))

@njit
def sig1(x):
    return T * sig(x) * (1 - sig(x))

@njit
def sig2(x):
    return T * (sig1(x) - 2*sig(x)*sig1(x))

@njit
def sig3(x):
    return T * (sig2(x) - 2 * (sig1(x)**2 + sig(x) * sig2(x)))

@njit
def sig_pr(x, k):
    if k==0:
        return sig(x)
    if k==1:
        return sig1(x)
    if k==2:
        return sig2(x)
    if k==3:
        return sig3(x)


# %%
@njit
def f(x, y):
    return 2*y*(y-1) + 2*x*(x-1)


@njit
def psi_a(x, y):
    return x*(1-x)*y*(1-y)


@njit
def psi_a_dx(x,y):
    return (2*x-1)*(y-1)*y


@njit
def psi_a_dy(x,y):
    return (2*y-1)*(x-1)*x


# The output of neural network $N(x,y,\vec{p})$, where $\vec{p} = [w, u, v]$:
# $$N = \sum_i^H v_i \sigma(z_i) \text{, where } z_i = w_{i0} x + w_{i1} y + u_i$$
@njit
def z(x, y, p00, p01, p1):
    z_x = x * p00
    z_y = y * p01
    z_ = z_x + z_y + p1

    return z_

@njit
def N(x, y, p00, p01, p1, p2):
    return np.sum(sig(z(x, y, p00, p01, p1)) * p2)


# $$\frac{\partial^k N}{\partial x_j^k} = \sum_{i=1}^H v_i w_{ij}^k \sigma^{(k)}$$
@njit
def dN_dxj_k(x, y, p00, p01, p1, p2, j, k):
    wj = p00 if j==0 else p01
    z_ = z(x, y, p00, p01, p1)
    
    return np.sum(p2 * (wj**k) * sig_pr(z_, k))

# $$\frac{\partial N}{\partial w_j} = x_j v \sigma '$$
@njit
def dN_dwj(x, y, p00, p01, p1, p2, j):
    xj = x if j==0 else y
    z_ = z(x, y, p00, p01, p1)
    return xj * p2 * sig1(z_)


# $$ \frac{\partial}{\partial w_j} \frac{\partial N}{\partial x_k} = x_j v w_k \sigma'' + v_i \sigma' \quad\text{ if } j = k$$
# $$ \frac{\partial}{\partial w_j} \frac{\partial N}{\partial x_k} = x_j v w_k \sigma'' \quad\text{ if } j \neq k$$

@njit
def d_dwj_dN_dxk(x, y, p00, p01, p1, p2, j, k):
    xj = x if j==0 else y
    wk = p00 if k==0 else p01
    jk = 1 if j==k else 0
    z_ = z(x, y, p00, p01, p1)
    return xj * p2 * wk * sig2(z_) + jk * p2 * sig1(z_)


# $$ \frac{\partial}{\partial w_j} \frac{\partial^2 N}{\partial x_k^2} = x_j v w_k^2 \sigma^{(3)} + 2 v w_k \sigma'' \quad\text{ if } j = k $$
# $$ \frac{\partial}{\partial w_j} \frac{\partial^2 N}{\partial x_k^2} = x_j v w_k^2 \sigma^{(3)} \quad\text{ if } j \neq k $$

@njit
def d_dwj_dN2_dxk2(x, y, p00, p01, p1, p2, j, k):
    xj = x if j==0 else y
    wk = p00 if k==0 else p01
    jk = 1 if j==k else 0
    z_ = z(x, y, p00, p01, p1)
    return xj * p2 * (wk**2) * sig3(z_) + jk * 2 * p2 * wk * sig2(z_)


# $$ \frac{\partial}{\partial u} \frac{\partial^k}{\partial x_j^k} N = v w_j^k \sigma^{(k+1)} $$
@njit
def d_du_dkN(x, y, p00, p01, p1, p2, j, k):
    wj = p00 if j==0 else p01
    z_ = z(x, y, p00, p01, p1)
    return p2 * (wj**k) * sig_pr(z_, k+1)

# $$ \frac{\partial}{\partial v} \frac{\partial^k}{\partial x_j^k} N = w_j^k \sigma^{(k)} $$
@njit
def d_dv_dkN(x, y, p00, p01, p1, p2, j, k):
    wj = p00 if j==0 else p01
    z_ = z(x, y, p00, p01, p1)
    return (wj**k) * sig_pr(z_, k)


# ## Cost function
# $$E[\vec{p}] = \sum_{i \in \hat{D}} \left\{ \frac{\partial^2 N}{\partial x^2} + \frac{\partial^2 N}{\partial y^2} - f(x,y) \right\}^2 
#            +  \sum_{i \in \partial \hat{D}} N^2$$


# %%
@njit# ('float64(float64, float64, float64[:], float64[:], float64[:], float64[:])')
def error_term1(x, y, p00, p01, p1, p2):
    ans = dN_dxj_k(x, y, p00, p01, p1, p2, 0, 2) +  dN_dxj_k(x, y, p00, p01, p1, p2, 1, 2)  -  f(x, y)
    return ans


@njit
def cost(points, boundary_points, p00, p01, p1, p2, BC):
    et1 = np.zeros(points.shape[0])
    et2 = np.zeros(boundary_points.shape[0])
    
    for i, pnt in enumerate(points):
        et1[i] = error_term1(pnt[0], pnt[1], p00, p01, p1, p2)**2
    
    for i, pnt in enumerate(boundary_points):
        et2[i] = N(pnt[0], pnt[1], p00, p01, p1, p2)**2
    
    cost = np.sum(et1) + BC*np.sum(et2)
    
    return cost

@njit
def relative_err(p00, p01, p1, p2, all_points):
    #area/num_points
    dOmega = 1. / len(all_points)
    
    tr1 = np.zeros(all_points.shape[0])
    tr2 = np.zeros(all_points.shape[0])
    ana1 = np.zeros(all_points.shape[0])
    ana2 = np.zeros(all_points.shape[0])
    
    for i, pnt in enumerate(all_points):
        tr1[i] = dOmega * (np.abs(N(pnt[0], pnt[1], p00, p01, p1, p2) - psi_a(pnt[0],pnt[1]))**2)

    for i, pnt in enumerate(all_points):
        tr2[i] = dOmega * ((dN_dxj_k(pnt[0],pnt[1],p00,p01,p1,p2,0,1)-psi_a_dx(pnt[0],pnt[1]))**2 +
                           (dN_dxj_k(pnt[0],pnt[1],p00,p01,p1,p2,1,1)-psi_a_dy(pnt[0],pnt[1]))**2)

    for i, pnt in enumerate(all_points):
        ana1[i] = dOmega * (np.abs(psi_a(pnt[0],pnt[1]))**2)
        
    for i, pnt in enumerate(all_points):
        ana2[i] = dOmega* (psi_a_dx(pnt[0],pnt[1])**2 + psi_a_dy(pnt[0],pnt[1])**2)
    
    rel_err = (np.sum(tr1) + np.sum(tr2))**(1/2) / (np.sum(ana1) + np.sum(ana2))**(1/2)

    return rel_err

def relative_err_without_points(p00, p01, p1, p2, nx=100):
    all_points = np.array(list(itertools.product(np.linspace(0, 1, nx), np.linspace(0, 1, nx))))
    return relative_err(p00, p01, p1, p2, all_points)




# $$ \frac{\partial E[\vec{p}]}{\partial w_j} = \sum_{i \in \hat{D}} \left\{ 2 \text{ (error_term1) } \left( \frac{\partial}{\partial w_j} \frac{\partial^2 N}{\partial x^2} + \frac{\partial}{\partial w_j} \frac{\partial^2 N}{\partial y^2} \right) \right\}  +  \sum_{i \in \partial \hat{D}} 2 N \frac{\partial N}{\partial w_j}$$

# %%
def dE_dwj(points, boundary_points, p00, p01, p1, p2, j, BC):
    t1 = np.zeros((points.shape[0], p00.shape[0]))
    t2 = np.zeros((boundary_points.shape[0], p00.shape[0]))
    
    for i, pnt in enumerate(points):
        t1[i] = 2 * error_term1(pnt[0],pnt[1],p00, p01, p1, p2) * (
                d_dwj_dN2_dxk2(pnt[0],pnt[1],p00, p01, p1, p2,j,0) + 
                d_dwj_dN2_dxk2(pnt[0],pnt[1],p00, p01, p1, p2,j,1))
        
    for i, pnt in enumerate(boundary_points):
        t2[i] = 2 * N(pnt[0],pnt[1],p00, p01, p1, p2) * dN_dwj(pnt[0],pnt[1],p00, p01, p1, p2,j)
    
    grad = np.sum(t1, axis=0) + BC*np.sum(t2, axis=0)
    
    return grad


# $$ \frac{\partial E[\vec{p}]}{\partial u} = \sum_{i \in \hat{D}} \left\{ 2 \text{ (error_term1) } \left( \frac{\partial}{\partial u} \frac{\partial^2 N}{\partial x^2} + \frac{\partial}{\partial u} \frac{\partial^2 N}{\partial y^2} \right) \right\} +  \sum_{i \in \partial \hat{D}} 2 N \frac{\partial N}{\partial u}$$

@njit
def dE_du(points, boundary_points, p00, p01, p1, p2, BC):
    t1 = np.zeros((points.shape[0], p1.shape[0]))
    t2 = np.zeros((boundary_points.shape[0], p1.shape[0]))
    
    for i, pnt in enumerate(points):
        t1[i] = 2 * error_term1(pnt[0],pnt[1],p00, p01, p1, p2) *                 (d_du_dkN(pnt[0],pnt[1],p00, p01, p1, p2,0,2) +                  d_du_dkN(pnt[0],pnt[1],p00, p01, p1, p2,1,2))
        
    for i, pnt in enumerate(boundary_points):
        t2[i] = 2 * N(pnt[0],pnt[1],p00, p01, p1, p2) * d_du_dkN(pnt[0],pnt[1],p00, p01, p1, p2,0,0)
    
    grad = np.sum(t1, axis=0) + BC*np.sum(t2, axis=0)
    
    return grad


# $$ \frac{\partial E[\vec{p}]}{\partial v} = \sum_{i \in \hat{D}} \left\{ 2 \text{ (error_term1) } \left( \frac{\partial}{\partial v} \frac{\partial^2 N}{\partial x^2} + \frac{\partial}{\partial v} \frac{\partial^2 N}{\partial y^2} \right) \right\}  +  \sum_{i \in \partial \hat{D}} 2 N \frac{\partial N}{\partial v}$$

@njit
def dE_dv(points, boundary_points, p00, p01, p1, p2, BC):
    t1 = np.zeros((points.shape[0], p2.shape[0]))
    t2 = np.zeros((boundary_points.shape[0], p2.shape[0]))
    
    for i, pnt in enumerate(points):
        t1[i] = 2 * error_term1(pnt[0],pnt[1],p00, p01, p1, p2) *                 (d_dv_dkN(pnt[0],pnt[1],p00, p01, p1, p2,0,2) +                  d_dv_dkN(pnt[0],pnt[1],p00, p01, p1, p2,1,2))
        
    for i, pnt in enumerate(boundary_points):
        t2[i] = 2 * N(pnt[0],pnt[1],p00, p01, p1, p2) * d_dv_dkN(pnt[0],pnt[1],p00, p01, p1, p2,0,0)
    
    grad = np.sum(t1, axis=0) + BC*np.sum(t2, axis=0)
    
    return grad
