from time import time
import numpy as np
import itertools
from numba import jit, njit, cuda, float64, float32

HN_CONST=10
BC=1


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




K=10
@njit
def f(x, y):
    return K * (6*(x-1)*x*(1-2*y) + 2*(y-1)*y*(1-2*y))


@njit
def psi_a(x, y):
    return K * x*(1-x)*y*(1-y)*(1-2*y)


@njit
def psi_a_dx(x,y):
    return K * (1-2*x)*y*(1-y)*(1-2*y)



@cuda.jit(device=True)
def psi_a_dy(x,y):
    return K * x*(1-x)*(1-6*y+6*(y**2))


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


@cuda.jit(device=True)
def d_dwj_dN_dxk(x, y, p00, p01, p1, p2, j, k):
    xj = x if j==0 else y
    wk = p00 if k==0 else p01
    jk = 1 if j==k else 0
    z_ = z(x, y, p00, p01, p1)
    return xj * p2 * wk * sig2(z_) + jk * p2 * sig1(z_)


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
        result[i] = xj * p2[i] * (wk[i]**2) * sig_3[i] +  jk * 2 * p2[i] * wk[i] * sig_2[i]
    
    return result

@njit
def d_dwj_dN2_dxk2_orig(x, y, p00, p01, p1, p2, j, k):
    xj = x if j==0 else y
    wk = p00 if k==0 else p01
    jk = 1 if j==k else 0
    z_ = z_orig(x, y, p00, p01, p1)
    return xj * p2 * (wk**2) * sig3_orig(z_) + jk * 2 * p2 * wk * sig2_orig(z_)


@cuda.jit(device=True)
def d_du_dkN(x, y, p00, p01, p1, p2, j, k):
    wj = p00 if j==0 else p01
    z_ = z(x, y, p00, p01, p1)
    return p2 * (wj**k) * sig_pr(z_, k+1)


@cuda.jit(device=True)
def d_dv_dkN(x, y, p00, p01, p1, p2, j, k):
    wj = p00 if j==0 else p01
    z_ = z(x, y, p00, p01, p1)
    return (wj**k) * sig_pr(z_, k)



@cuda.jit(device=True)
def error_term1(x, y, p00, p01, p1, p2):
    ans = dN_dxj_k(x, y, p00, p01, p1, p2, 0, 2) + dN_dxj_k(x, y, p00, p01, p1, p2, 1, 2) - f(x, y)
    return ans

@njit
def error_term1_orig(x, y, p00, p01, p1, p2):
    ans = dN_dxj_k_orig(x, y, p00, p01, p1, p2, 0, 2) + dN_dxj_k_orig(x, y, p00, p01, p1, p2, 1, 2) - f(x, y)
    return ans



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


@cuda.jit
def dE_dwj_t1(points, p00, p01, p1, p2, j, result):
    pos = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    
    if pos < points.shape[0]:
        dnx = cuda.local.array(shape=HN_CONST, dtype=float64)
        dnx = d_dwj_dN2_dxk2(points[pos][0], points[pos][1], p00, p01, p1, p2, j, 0, dnx)
        dny = cuda.local.array(shape=HN_CONST, dtype=float64)
        dny = d_dwj_dN2_dxk2(points[pos][0], points[pos][1], p00, p01, p1, p2, j, 1, dny)
        et1 = error_term1(points[pos][0], points[pos][1], p00, p01, p1, p2)
        
        t1 = g_add_2(dnx, dny, dnx)
        t1 = g_mul_num_arr(2*et1, dnx, t1)
        for i in range(HN_CONST):
            result[pos][i] = t1[i]


@cuda.jit
def dE_dwj_t2(points, p00, p01, p1, p2, j, result):
    pos = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    
    if pos < points.shape[0]:
        n_ = N(points[pos][0], points[pos][1], p00, p01, p1, p2)
        dn_dwj_ = cuda.local.array(shape=HN_CONST, dtype=float64)
        dn_dwj_ = dN_dwj(points[pos][0], points[pos][1], p00, p01, p1, p2, j, dn_dwj_)
        for i in range(HN_CONST):
            result[pos][i] = 2 * n_ * dn_dwj_[i]


def dE_dwj(points, boundary_points, p00, p01, p1, p2, j):
    t1 = np.zeros((points.shape[0], p00.shape[0]))
    t2 = np.zeros((boundary_points.shape[0], p00.shape[0]))
    threadsperblock = 64
    
    blockspergrid_t1 = (points.shape[0] + (threadsperblock - 1)) // threadsperblock
    dE_dwj_t1[blockspergrid_t1, threadsperblock](points, p00, p01, p1, p2, j, t1)
    
    blockspergrid_t2 = (points.shape[0] + (threadsperblock - 1)) // threadsperblock
    dE_dwj_t2[blockspergrid_t2, threadsperblock](boundary_points, p00, p01, p1, p2, j, t2)
    
    grad = np.sum(t1, axis=0) + BC*np.sum(t2, axis=0)
    
    return grad



# In[324]:
nx=60
bx=300

boundary_points = np.array(list(set(list(itertools.product([0, 1], np.linspace(0,1,bx))) +
                                list(itertools.product(np.linspace(0,1,bx), [0, 1])))))
pnts = list(itertools.product(np.linspace(0, 1, nx), np.linspace(0, 1, nx)))
points = np.array([(x,y) for x,y in pnts if (x not in [0,1] and y not in [0,1])])

p00 = np.random.randn(HN_CONST)
p01 = np.random.randn(HN_CONST)
p1 = np.random.randn(HN_CONST)
p2 = np.random.randn(HN_CONST)

# dE_dwj(points, boundary_points, p00, p01, p1, p2, 0)

t1 = np.zeros((points.shape[0], p00.shape[0]))
threadsperblock = 64

blockspergrid_t1 = (points.shape[0] + (threadsperblock - 1)) // threadsperblock
loop = 100
dE_dwj_t1[blockspergrid_t1, threadsperblock](points, p00, p01, p1, p2, 0, t1)
start = time()
for _ in range(loop):
    dE_dwj_t1[blockspergrid_t1, threadsperblock](points, p00, p01, p1, p2, 0, t1)
end = time()

print(end-start, (end-start)/loop)
