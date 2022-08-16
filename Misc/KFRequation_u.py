import numpy as np
from math import erf as erf_, sin, exp as exp_
from numpy import pi, sqrt
import time
import os
import re
import array
import numba as nb
import h5py

@nb.vectorize(('f8(f8)'))
def erf(x):
    return erf_(x)

@nb.vectorize('f8(f8)')
def exp(x):
    return exp_(x)

Определим функцию $u_0(\xi)$ - некое переменное начальное состояние перед ударной волной. Аргумент $\xi$ здесь относится к лабораторной системе отсчета, т.е. $\xi=x + \int_0^t D(\tau) d\tau$.<br>
Из-за того, что в качестве масштаба выбрана величина $u_{0s} - u_a$, то выражение для $u_a$ надо поделить на $1-A$.

@nb.njit('f8(f8, f8, f8, i8)')
def u_a(x_lab, amp, wn, sign):
    '''Define the initial speed beyond the shock wave'''

    return amp / (1 - amp) * (1 + sign * sin(wn * x_lab))

@nb.njit('f8(f8, f8, optional(f8), optional(f8), i8)')
def Dshock(us, amp, x_lab, wn, sign):
    '''Define the speed of the shock wave, D'''

    if x_lab is not None:
        return (us + u_a(x_lab, amp, wn, sign)) * 0.5
    else:
        return (us + amp / (1 - amp)) * 0.5

@nb.njit('f8(f8, f8, f8)')
def xi(us, alpha, amp):
    '''Define the induction function dependence on the shock speed, us'''

    return -((1 - amp) * us) ** (-alpha)

@nb.njit('f8[:](f8[:], f8, f8, f8, f8)')
def s(x, us, alpha, beta, amp):
    '''Define the source u_t + F_x = S'''

    ainv = 8 * sqrt(pi * beta) * (1 + erf(1 / sqrt(4 * beta)))
    source = exp(-(x - xi(us, alpha, amp)) ** 2 / (4 * beta)) / ainv
    return source

# source = -xi(Dshock).*0.5*1/sqrt(4*pi*beta)*exp( -(x + 1).^2/(4*beta) );
# with the source on the previous line, there is instability but no period
# doubling or chaos.

@nb.njit('f8[:](f8[:], f8, f8, optional(f8), optional(f8), i8)')
def F(Y, us, amp, x_lab, wn, sign):
    '''Define the flux function in u_t + F_x = S'''

    D = Dshock(us, amp, x_lab, wn, sign)
    return 0.5 * Y ** 2. - D * Y

Функция задает граничные условия.
<code>Ym</code> - одномерный массив numpy.

@nb.jit('f8[:](f8[:], f8[:], i8[:], i8[:], i8[:])')
def set_bc(x, Ym, Nleft, Nint, Nright):
    '''
                                                      shock, x=0
                    x=-L                                :
      |     |     |  :  |                |     |     |  :
      |     |     |  :  |                |     |     |  :
   o-----o-----o-----X-----X--      --X-----X-----X-----X
   0     1     2     3     4                     N+2   N+3
the boundaries are shown as :
the physical domain [-L 0] is from x[3]=-L to x[N+3]=0.
the shock is located at x[N+3] = 0.
internal points are from i=3 to i=N+3.
the state at i=N+4:N+6 is given by parabolic extrapolation
'''

    Ym[Nleft] = Ym[Nint[0] * np.ones_like(Nleft, dtype=int)]
    order = 4  # right boundary extrapolation order
    Nextrap = np.arange(Nint[-1] - order, Nint[-1] + 1)  # index of points used for extrapolation
    p = np.polyfit(x[Nextrap], Ym[Nextrap], order)  # polynomial extrapolation
    Ym[Nright] = np.polyval(p, x[Nright])
    return Ym

<code>Ym</code>, <code>ii</code> - одномерные массивы numpy. Значения по умолчанию - нули, чтобы была совместимость типов при передаче параметра <code>x_lab=None</code> в функцию <code>Dshock</code>.

%run ./weno_matrices.py

@nb.njit('f8[:](f8[:], i8[:], i8, f8, optional(f8), optional(f8), i8)')
def weno5(Ym, ii, ishock, amp, x_lab, wn, sign):
    '''
ii = Nint. Returns the flux at the right cell edge: Flux = F_j+1/2
u = [f(j-2) f(j-1) f(j) f(j+1) f(j+2) f(i+3)]'.
shifted by -1, this flux gives the flux at i = 3+1/2 which is first cell
boundary ahead of the shock position, i=3.
'''

    eps = 1e-6

    u = Ym
    us = u[ishock]
    
    #Is it lax-fredrich flux? It is indeed
    
    alfa = np.absolute(u - Dshock(us, amp, x_lab, wn, sign)).max()
    fp = 0.5 * (F(Ym, us, amp, x_lab, wn, sign) + alfa * Ym)
    fm = 0.5 * (F(Ym, us, amp, x_lab, wn, sign) - alfa * Ym)

    Flux = np.zeros_like(Ym)

    for j in range(ii[0] - 1, ii[-1] + 1):  # at all internal-cell edges, 3:N+3

        Yplus = np.array([fp[j-2], fp[j-1], fp[j], fp[j+1], fp[j+2], fp[j+3]], dtype=np.float64) #Left Bias
        Yminus = np.array([fm[j-2], fm[j-1], fm[j], fm[j+1], fm[j+2], fm[j+3]], dtype=np.float64) #Right Bias

        # smoothness indicators (Beta) (But how?)
        ISplus = np.array([
            np.dot(np.dot(Yplus, WmAp0), Yplus),
            np.dot(np.dot(Yplus, WmAp1), Yplus),
            np.dot(np.dot(Yplus, WmAp2), Yplus)
        ])

        ISminus = np.array([
            np.dot(np.dot(Yminus, WmAm0), Yminus),
            np.dot(np.dot(Yminus, WmAm1), Yminus),
            np.dot(np.dot(Yminus, WmAm2), Yminus)
        ])

        # the weights (The linear weights gamma)
        alphaplus = 0.1 * np.array([1, 6, 3], dtype=np.float64) / (eps + ISplus) ** 2.
        alphaminus = 0.1 * np.array([1, 6, 3], dtype=np.float64) / (eps + ISminus) ** 2.

        omegaplus = alphaplus / alphaplus.sum()
        omegaminus = alphaminus / alphaminus.sum()
        
        #Monotone fluxes
        
        fhatplus = np.dot(omegaplus, np.dot(WmNplus, Yplus)) 
        fhatminus = np.dot(omegaminus, np.dot(WmNminus, Yminus))

        Flux[j] = fhatplus + fhatminus

    return Flux

Co = 0.8  # Courant number
epsilon = 10 ** -6  # magnitude of perturbation
%run ./result_handler.py

def kfr_equation(
        L=10,       # domain [-L,0]
        N=200,      # number of grid points on the domain
        tmax=500,   # computation time
        alpha=5.4,  # sensitivity to shock speed
        beta=0.1,   # smoothness of the delta approximation
        amp=0.,     # parameters of u_a
        wn=None,    # wave number
        sign = 1,
        store=True, # enables output to a file
        further=True,  # continue calculations from the last moment in file
        format='hdf5'  # format of file to save to
):
    """
-------------------------Aslan Kasimov------------------------------------
----------------------------KAUST-----------------------------------------
-------------------------Feb-9-2012---------------------------------------

Solve our special hyperbolic balance law (mini-detonation model)
u_t + f_x = s.
where f = u^2/2 - u*u(0,t)/2, s = s(x,u(0,t))
below we take s = 0.5*1/sqrt(4*pi*beta)*exp[ -(x - xi)^2/(4*beta) ]
where xi = [u(0,t)]^(-alpha)
                                                       shock, x=0
                     x=-L                                :
       |     |     |  :  |                |     |     |  :
       |     |     |  :  |                |     |     |  :
    o-----o-----o-----X-----X--      --X-----X-----X-----X
    0     1     2     3     4                     N+2   N+3
the boundaries are shown as :
the physical domain [-L 0] is from x(4)=-L to x(N+4)=0.
the shock is located at x(N+4) = 0.
internal points are from i=4 to i=N+4.

WENO matrices for conservation law
WM.Ap0 = zeros(6); WM.Ap0(1:3,1:3) = 1/6*[ 8 -19 11; -19 50 -31; 11 -31 20]
WM.Ap1 = zeros(6); WM.Ap1(2:4,2:4) = 1/6*[ 8 -13  5; -13 26 -13;  5 -13  8]
WM.Ap2 = zeros(6); WM.Ap2(3:5,3:5) = 1/6*[20 -31 11; -31 50 -19; 11 -19  8]
WM.Am0 = rot90(WM.Ap0,2);  WM.Am1 = rot90(WM.Ap1,2); WM.Am2 = rot90(WM.Ap2,2)
WM.Nplus = 1/6*[2 -7 11 0 0 0; 0 -1 5 2 0 0; 0 0 2 5 -1 0]
WM.Nminus = fliplr(WM.Nplus)
"""
    
    # set the computational parameters
    ishock = N + 3  # the shock state, i.e. the right boundary
    dx = L / N  # grid size

    x = np.empty(N + 7)
    x[0:3] = [(-L + dx * i) for i in [-3, -2, -1]]  # left ghost points
    x[3:N + 4] = [(-L + dx * i) for i in range(N + 1)]  # internal grid points on [-L,0]
    x[N + 4:N + 7] = np.arange(1, 4) * dx  # grid points ahead of the shock

    # initial profiles for steady solutions (could be other Cauchy data)
    us = 1. / (1 - amp)
    Source = s(x, us, alpha, beta, amp)
    var1 = 1 + erf((x + 1) / sqrt(4 * beta))
    var2 = 1 + erf((0 + 1) / sqrt(4 * beta))

    u = amp / (1 - amp) + 0.5 * (1 + sqrt(var1 / var2)) + epsilon * np.sin(x)
    Nleft = np.arange(0, 3, dtype=np.int64)  # index of left ghost points
    Nint = np.arange(3, N + 4, dtype=np.int64)  # index range for the internal points
    Nweno = np.arange(3, N + 4, dtype=np.int64)  # index range for the internal points where WENO is applied
    Nright = np.arange(N + 4, N + 7, dtype=np.int64)  # index of right ghost points

    # -------------------------Set the initial profile and BC----------------
    set_bc(x, u, Nleft, Nint, Nright)

    # --------------------------Initial state finished -------------------------
    cpu = np.absolute(u - 0.5 * (us + amp / (1 - amp))).max()
    dt = Co * dx / cpu
    t = 0.
    x_lab = 0. if wn else None

    # fill in the dict of problem parameters and initial conditions, needed
    # for saving
    par = {
        'alpha': alpha,
        'beta': beta,
        'L': L,
        'N': N,
        'amp': amp,
        'wn': wn,
        'sign': sign,
        'tmax': tmax,
    }
    
    # solve the equation du/dt = L by TVD RK3 (L = - F_x + S).
    start = time.perf_counter()
    Source = s(x, u[ishock], alpha, beta, amp)
    L1, Y2, L2, Y3, L3 = (np.zeros_like(Source) for _ in range(5))
    
    # setting output and further calculations
    if store:
        #_path = '../comp_res'
        #relpath = f'alpha_{alpha}beta_{beta}/L_{L}N_{N}'
        filename = f'amp_{amp}' + (f'wn_{wn}p' if wn else '') + f'alpha_{alpha}'
        #fullpath = os.path.join(_path)
        #os.makedirs(fullpath, exist_ok=True)
        print(os.path.join(filename), end=' ')
            
    @nb.jit
    def RK3(u, Source, x_lab, dt):
        Flux = weno5(u, Nweno, ishock, amp, x_lab, wn, sign)
        L1[Nweno] = (Flux[Nweno - 1] - Flux[Nweno]) / dx
        L1[Nint] += Source[Nint]
        Y2[Nint] = u[Nint] + dt * L1[Nint]
        set_bc(x, Y2, Nleft, Nint, Nright)

        Source = s(x, Y2[ishock], alpha, beta, amp)
        Flux = weno5(Y2, Nweno, ishock, amp, x_lab, wn, sign)
        L2[Nweno] = (Flux[Nweno - 1] - Flux[Nweno]) / dx
        L2[Nint] += Source[Nint]
        Y3[Nint] = Y2[Nint] + dt / 4 * (-3 * L1[Nint] + L2[Nint])
        set_bc(x, Y3, Nleft, Nint, Nright)

        Source = s(x, Y3[ishock], alpha, beta, amp)
        Flux = weno5(Y3, Nweno, ishock, amp, x_lab, wn, sign)
        L3[Nweno] = (Flux[Nweno - 1] - Flux[Nweno]) / dx;
        L3[Nint] += Source[Nint]
        u[Nint] = Y3[Nint] + dt / 12 * (-L1[Nint] - L2[Nint] + 8 * L3[Nint])
        set_bc(x, u, Nleft, Nint, Nright)
        return u

    res_t, res_u = array.array('d'), array.array('d')
    res_t.append(0)
    res_u.extend(u)
    while t < tmax:
        u = RK3(u, Source, x_lab, dt)
        us = u[ishock]
        if x_lab is not None:
            x_lab += Dshock(us, amp, x_lab, wn, sign) * dt  # current \xi_s
        Source = s(x, us, alpha, beta, amp)
        D = Dshock(us, amp, x_lab, wn, sign)
        cpu = np.absolute(u - D).max()
        dt = Co * dx / cpu
        t += dt  # current time
        res_t.append(t)
        res_u.extend(u)

    if store:
        res_u = np.array(res_u).reshape((int(len(res_u) / len(x)), len(x)))
        #res_u = res_u[:,-1]
        np.savez(os.path.join(filename), t=res_t, u=res_u)
    comp_time = round(time.perf_counter() - start, 2)
    print(f'=-=-=-=-=-= {comp_time}s')
    par['further'] = further
    par.pop('tmax')
    return par



