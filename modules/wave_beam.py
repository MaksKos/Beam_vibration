#!/usr/bin/env python
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import solve, solve_banded

class WaveLong():

    rho = 7800
    E = 200e9
    vi = 0.25
    g = 9.81

    def __init__(self, geometry,  T, dt, T_st=0, fi=0, mass_on=False, seal_move=0) -> None:

        self.geom = geometry
        self.T = T
        self.dt = dt
        self.fi = fi
        self.T_st = T_st
        self.on_mass = mass_on
        self.seal = seal_move
        self._is_pressure = False
        self._xgrid = []
        self._tgrid = []
        self._u = []
        if mass_on and T_st < 1e-15:
            print("Reccomend att T_st for calming of fluctuations")
    
    def add_pressure(self, pressure, projectile=False):
        self._is_pressure = True
        self._pressure = pressure
        self._p = pressure.p
        self._dSdx = self.geom.dSdx
        self._S = self.geom.S
        self._dpdx = pressure.dpdx
        self._proj = pressure.proj
        if projectile:
            self.f_p = lambda x, t: self.__f_pressure(x, t)-\
                                    self._proj(x, t)*np.sin(self.fi)
        else:
            self.f_p = lambda x, t: self.__f_pressure(x, t)

    def __f_pressure(self, x, t):
        return 2*self.vi*(self._p(x, t)*self._dSdx(x) + \
                 self._S(x)*self._dpdx(x, t)) - \
                self._p(x, t)*self._dSdx(x)

    def __f_mass(self, x, t):

        stabil = 0.8*self.T_st
        if stabil < 1e-10:
            coef = 1
        else:
            coef = t/stabil
        if coef > 1:
            coef=1
        return -self.rho*self.g*np.sin(self.fi)*\
                    coef*self.geom.F(x)

    def solver(self):
        
        print(f'\n T: {np.round(self.T, 3)}\n T_st: {self.T_st}\n',
              f'Mass: {self.on_mass}\n Pressure: {self._is_pressure}\n')
        # u_tt = a*(c**2*u_x)_x + a*f(x,t) on (0,L)
        # initial comditions
        I = lambda x: 0 # u(0, x)
        V = lambda x: 0 # du/dt (t=0)
        # boundaru conditions
        U_0 = lambda t: 0 # u(t, 0) if None - du/dx = 0 use - в начале 
        U_L = None # u(t, L) if None - du/dx = 0 use - на конце

        # функция f(x, t)
        # Если решаем на учатске F и S - const, то делим на rho и F
        if self._is_pressure:
            f = lambda x, t: self.on_mass*self.__f_mass(x+self.seal, t)+\
                                self.f_p(x+self.seal, t-self.T_st)
        else:
            f = lambda x, t: self.on_mass*self.__f_mass(x+self.seal, t)
        a = lambda x: 1/self.rho/self.geom.F(x+self.seal)
        c = lambda x: np.sqrt(self.E*self.geom.F(x+self.seal))
        # параметры
        L = self.geom.L - self.seal
        C = 0.75 # the Courant number (=max(c)*dt/dx).
        T_total = self.T + self.T_st
        
        # --- start calculate ---
        x, t, u = solver_long(I, V, f, c, U_0, U_L, L, self.dt, C, T_total, a, version='vectorized')
        self._xgrid = x
        self._tgrid = t
        self._u = u

        # make function u(x, t)
        self.u = RegularGridInterpolator((x, t), u.T, fill_value=0, bounds_error=False, method='linear')

    @property
    def grid_u(self):
        return self._u
    
    @property
    def grid_x(self):
        return self._xgrid

    @property
    def grid_t(self):
        return self._tgrid
    

class WaveTang():
        
    rho = 7800
    E = 200e9
    vi = 0.25
    g = 9.81

    def __init__(self, geometry,  T, dt, T_st=0, fi=0,  dx_user=None, mass_on=False, seal_move=0) -> None:

        self.geom = geometry
        self._T = T
        self.dt = dt
        self.fi = fi
        self.T_st = T_st
        self.T = T+T_st
        self.on_mass = mass_on
        self.seal = seal_move
        self._is_pressure = False
        self._is_init = False
        self.L = self.geom.L - self.seal
        self._f = lambda x, t: self.__f_mass(x, t)*mass_on
        if mass_on and T_st < 1e-15:
            print("Reccomend att T_st for calming of fluctuations")
        self.__generate_grid(dx_user)

    def __generate_grid(self, dx_user):

        C = 0.75
        # --- Compute time and space mesh ---
        self.Nt = int(round(self.T/self.dt))
        self._t = np.linspace(0, self.Nt*self.dt, self.Nt+1)      # Mesh points in time
        # --- Find dx ---
        if isinstance(dx_user, (float, int)):
            dx = dx_user
        else:
            JF_max = max([self.geom.J(x_+self.seal)/self.geom.F(x_+self.seal)\
                      for x_ in np.linspace(0, self.L, 101)])
            dx = self.dt*np.sqrt(self.E*JF_max/self.rho)/C

        self.Nx = int(round(self.L/dx))
        self._x = np.linspace(0, self.L, self.Nx+1)        # Mesh points in space
        # Add main geometry
        self._F = self.geom.F(self._x+self.seal)
        self._S = self.geom.S(self._x+self.seal)
        self._J = self.geom.J(self._x+self.seal)
        # Make sure dx and dt are compatible with x and t
        self.dx = self._x[1] - self._x[0]
        self.dt = self._t[1] - self._t[0]
        self._vgrid = []
    
    def __f_mass(self, x, t):

        stabil = 0.8*self.T_st
        if stabil < 1e-10:
            coef = 1
        else:
            coef = t/stabil
        if coef > 1:
            coef=1
        return -self.rho*self.g*np.cos(self.fi)*\
                    coef*self.geom.F(x)
       
    def add_pressure(self, pressure, u, projectile=False):

        self._is_pressure = True
        self._pressure = pressure
        self._u = u
        self._p = pressure.p
        self._proj = pressure.proj
        self._f_p = lambda x, t: np.zeros_like(x)
        if projectile:
            self._f_p = lambda x, t: -self._proj(x, t)*np.cos(self.fi)

    def init_condition(self, v_0 = None):
        
        if v_0 is None:
            v_0 = np.zeros(self.Nx+1) 
        elif v_0.shape[0] != self._x.shape[0]:
            raise ValueError(f'different shape u0={v_0.shape[0]} and x={self._x.shape[0]}')
        # --- Allocate memomry for solutions ---
        self.v     = np.zeros(self.Nx+1)   # Solution array at new time level
        self.v_n   = np.zeros(self.Nx+1)   # Solution at 1 time level back
        self.v_nm1 = np.zeros(self.Nx+1)   # Solution at 2 time levels back

        # --- Valid indices for space and time mesh ---
        self.Ix = self.Nx+1
        self.It = self.Nt+1

        # --- Load initial condition into v_n and v_nm1 ---
        self.v_nm1 = v_0 
        self.v_n = v_0 + self.dt*self._f(self._x+self.seal, self._t[0])\
                                        /2/self.rho/self._F
        if self._is_pressure:
            self.v_n += self.dt*self._f_p(self._x+self.seal, self._t[0]-self.T_st)\
                                        /2/self.rho/self._F                  
        self.v_n[:2] = 0
        self._is_init = True  
         
    def solver(self, max_iter=None):
        
        if not self._is_init:
            print('initialized grid')
            return None
        
        self._grid_v = []
         
        # add to matrix init condition
        self.save_layer(self.v_nm1)
        self.save_layer(self.v_n)
        
        dt2 = self.dt**2
        # --- Time loop ---
        It = self.It
        if isinstance(max_iter, (int)):
            It = min(max_iter, It)
        # --- start loop ---
        for n in range(1, It-1):

            if self._is_pressure:
                u = self._u(self._x, self._t[n] - self.T_st)
                p = self._p(self._x + self.seal, self._t[n] - self.T_st) 
                f_p = self._f_p(self._x + self.seal, self._t[n] - self.T_st)
                omega = self.E*self._F*self.du_dx(u, self.dx) - self._S*p
            else:
                omega = np.zeros_like(self._x)
                f_p = np.zeros_like(self._x)

            f = self._f(self._x + self.seal, self._t[n])*dt2 + f_p*dt2\
                    + self.rho*self._F*(2*self.v_n - self.v_nm1)
            # --- solv x layer ---
            self.v = self.solver_matrix_bann(self.dx, dt2, self.Ix, self._J, omega, self._F, f, self.E, self.rho)
            # save
            self.save_layer(self.v)
            # Update data structures for next step
            self.v_nm1 = self.v_n
            self.v_n = self.v
            self.v = self.v_nm1 # ??
        self._vgrid = np.array(self._grid_v)
           
    @staticmethod
    def solver_matrix(dx, dt2, Ix, J, Om, F, f, E, rho):
        
        dx2 = dx*dx
        dx4 = dx**4
        C1 = dt2/dx2
        C2 = dt2/dx4
        # empty layer
        v_x = np.zeros(Ix)
        # allocate memory for coefficients
        a = np.zeros(Ix)
        b = np.zeros(Ix)
        c = np.zeros(Ix)
        d = np.zeros(Ix)
        e = np.zeros(Ix)
        # calculate coefficients
        a[2:] = C2*E*J[1:-1]
        b[1:] = -C1*(2*E/dx2 * (J[1:]+J[:-1]) + 0.5*(Om[1:] + Om[:-1]))
        c[1:-1] = rho*F[1:-1] + C1*0.5*(Om[0:-2] + 2*Om[1:-1] + Om[2:]) +\
                    E*C2*(J[2:] + 4*J[1:-1] + J[0:-2])
        d[:-1] = -C1*(2*E/dx2 * (J[1:]+J[:-1]) + 0.5*(Om[:-1] + Om[1:]))
        e[:-2] = C2*E*J[1:-1]
        # boundary condition for c[0]=0, c[I]
        c[-1] = rho*F[-1] + C1*0.5*(Om[-2]+2*Om[-1]) + E*C2*(4*J[-1]+J[-2])
        # --- calculate v_x ---
        A = np.zeros((Ix-2, Ix))
        for i in range(2,Ix-2):
                A[i-2][i-2:i+3] = np.array([
                        a[i], b[i], c[i], d[i], e[i]
                    ])
        A = np.delete(A,[0,1],1)
        A[-2,-4:] = [-J[-2], (J[-1]+2*J[-2]), -(J[-2]+2*J[-1]), J[-1]]
        A[-1,-3:] = [1, -2, 1]
        f_new = f[2:]
        f_new[-2:] = 0 
        # solve equption
        v_new = solve(A, f_new.reshape((f_new.shape[0],1)))
        v_x[2:] = v_new.reshape(-1)
        return v_x
    
    @staticmethod
    def solver_matrix_bann(dx, dt2, Ix, J, Om, F, f, E, rho):
        
        dx2 = dx*dx
        dx4 = dx**4
        C1 = dt2/dx2
        C2 = dt2/dx4
        # empty layer
        v_x = np.zeros(Ix)
        # allocate memory for coefficients
        a = np.zeros(Ix)
        b = np.zeros(Ix)
        c = np.zeros(Ix)
        d = np.zeros(Ix)
        e = np.zeros(Ix)
        # calculate coefficients
        a[2:] = C2*E*J[1:-1]
        b[1:] = -C1*(2*E/dx2 * (J[1:]+J[:-1]) + 0.5*(Om[1:] + Om[:-1]))
        c[1:-1] = rho*F[1:-1] + C1*0.5*(Om[0:-2] + 2*Om[1:-1] + Om[2:]) +\
                    E*C2*(J[2:] + 4*J[1:-1] + J[0:-2])
        d[:-1] = -C1*(2*E/dx2 * (J[1:]+J[:-1]) + 0.5*(Om[:-1] + Om[1:]))
        e[:-2] = C2*E*J[1:-1]
        # boundary condition for c[0]=0, c[I]
        c[-1] = rho*F[-1] + C1*0.5*(Om[-2]+2*Om[-1]) + E*C2*(4*J[-1]+J[-2])
        # --- calculate v_x ---
        A = np.zeros((5, Ix-2))
        A[0, 2:] = e[2:Ix-2]
        A[1, 1:-1] = d[2:Ix-2]
        A[1, -1] = J[-1]
        A[2,:-2] = c[2:Ix-2]
        A[2,-2:] = [-(J[-2]+2*J[-1]), 1]
        A[3, :-3] =  b[3:Ix-2]
        A[3, -3:-1] = [(J[-1]+2*J[-2]), -2]
        A[4, :-4] = a[4:Ix-2]
        A[4, -4:-2] = [-J[-2], 1]
        f_new = f[2:]
        f_new[-2:] = 0 
        # solve equption
        v_new = solve_banded((2, 2), A, f_new)
        v_x[2:] = v_new.reshape(-1)
        return v_x
    
    @staticmethod
    def du_dx(u_x, dx):
        dudx = np.zeros(u_x.shape[0]+2)
        dudx[1:-1] = u_x
        dudx[-1] = u_x[-1]
        dudx = np.roll(dudx, -2) - dudx
        return dudx[:-2]/2/dx
    
    def save_layer(self, v):
        self._grid_v.append(v.copy())

    @property
    def grid_v(self):
        return self._vgrid
    
    @property
    def grid_x(self):
        return self._x

    @property
    def grid_t(self):
        return self._t
      
class WaveTang_0():
    
    rho = 7800
    E = 200e9
    vi = 0.25
    g = 9.81
    
    def __init__(self, L, T, dt, F, S, J, u, p, f, C=0.75, dx_user=None):
        
        # --- Compute time and space mesh ---
        self.Nt = int(round(T/dt))
        self.t = np.linspace(0, self.Nt*dt, self.Nt+1)      # Mesh points in time
        # --- Matrix for v(x, t) --
        self.grid_v = []
        # --- Find dx ---
        
        JF_max = max([J(x_)/F(x_) for x_ in np.linspace(0, L, 101)])
        dx = dt*np.sqrt(self.E*JF_max/self.rho)/C
        if isinstance(dx_user, (float, int)):
            dx = dx_user #min(dx, dx_user)
        self.Nx = int(round(L/dx))
        self.x = np.linspace(0, L, self.Nx+1)          # Mesh points in space
        # Make sure dx and dt are compatible with x and t
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t[1] - self.t[0]
        # Add geometry
        self.F = F(self.x)
        self.S = S(self.x)
        self.J = J(self.x)
        # Add function
        self.fun_u = u
        self.fun_p = p
        self.fun_f = f
    
    def save_layer(self, v):
        self.grid_v.append(v.copy())
        
    def init_condition(self, v_0 = None):
        
        if v_0 is None:
            v_0 = np.zeros(self.Nx+1) 
        elif v_0.shape[0] != self.x.shape[0]:
            raise ValueError(f'different shape u0={v_0.shape[0]} and x={self.x.shape[0]}')
        # --- Allocate memomry for solutions ---
        self.v     = np.zeros(self.Nx+1)   # Solution array at new time level
        self.v_n   = np.zeros(self.Nx+1)   # Solution at 1 time level back
        self.v_nm1 = np.zeros(self.Nx+1)   # Solution at 2 time levels back

        # --- Valid indices for space and time mesh ---
        self.Ix = self.Nx+1
        self.It = self.Nt+1

        # --- Load initial condition into v_n and v_nm1 ---
        self.v_n = v_0 + self.dt*self.fun_f(self.x, self.t[0])/2\
                            /self.rho/self.F               
        self.v_n[:2] = 0
        self.v_nm1 = v_0
        
    def solver(self, max_iter=None):
        
        if not self.grid_v: 
            self.grid_v = []
            
        # add to matrix init condition
        self.save_layer(self.v_nm1)
        self.save_layer(self.v_n)
        
        dx2, dt2 = self.dx**2, self.dt**2
        du_dx = np.zeros_like(self.x)
        # --- Time loop ---
        It = self.It
        if isinstance(max_iter, (int, float)):
            It = min(max_iter, It)
        for n in range(1, It-1):
            u = self.fun_u(self.x, self.t[n])
            p = self.fun_p(self.x, self.t[n]) 
            f = self.fun_f(self.x, self.t[n])*dt2 + self.rho*self.F*\
                    (2*self.v_n - self.v_nm1)
            omega = self.E*self.F*self.du_dx(u, self.dx) - self.S*p
            # --- solv x layer ---
            self.v = self.solver_matrix(self.dx, dt2, self.Ix, self.J, omega, self.F, f, self.E, self.rho)
            # save
            self.save_layer(self.v)
            # Update data structures for next step
            self.v_nm1 = self.v_n
            self.v_n = self.v
            self.v = self.v_nm1 # ??
           
    @staticmethod
    def solver_matrix(dx, dt2, Ix, J, Om, F, f, E, rho):
        
        dx2 = dx*dx
        dx4 = dx**4
        C1 = dt2/dx2
        C2 = dt2/dx4
        # empty layer
        v_x = np.zeros(Ix)
        # allocate memory for coefficients
        a = np.zeros(Ix)
        b = np.zeros(Ix)
        c = np.zeros(Ix)
        d = np.zeros(Ix)
        e = np.zeros(Ix)
        # calculate coefficients
        a[2:] = C2*E*J[1:-1]
        b[1:] = -C1*(2*E/dx2 * (J[1:]+J[:-1]) + 0.5*(Om[1:] + Om[:-1]))
        c[1:-1] = rho*F[1:-1] + C1*0.5*(Om[0:-2] + 2*Om[1:-1] + Om[2:]) +\
                    E*C2*(J[2:] + 4*J[1:-1] + J[0:-2])
        d[:-1] = -C1*(2*E/dx2 * (J[1:]+J[:-1]) + 0.5*(Om[:-1] + Om[1:]))
        e[:-2] = C2*E*J[1:-1]
        # boundary condition for c[0]=0, c[I]
        c[-1] = rho*F[-1] + C1*0.5*(Om[-2]+2*Om[-1]) + E*C2*(4*J[-1]+J[-2])
        # --- calculate v_x ---
        A = np.zeros((Ix-2, Ix))
        for i in range(2,Ix-2):
                A[i-2][i-2:i+3] = np.array([
                        a[i], b[i], c[i], d[i], e[i]
                    ])
        A = np.delete(A,[0,1],1)
        A[-2,-4:] = [-J[-2], (J[-1]+2*J[-2]), -(J[-2]+2*J[-1]), J[-1]]
        A[-1,-3:] = [1, -2, 1]
        f_new = f[2:]
        f_new[-2:] = 0 
        # solve equption
        v_new = solve(A, f_new.reshape((f_new.shape[0],1)))
        v_x[2:] = v_new.reshape(-1)
        return v_x
    
    @staticmethod
    def du_dx(u_x, dx):
        dudx = np.zeros(u_x.shape[0]+2)
        dudx[1:-1] = u_x
        dudx[-1] = u_x[-1]
        dudx = np.roll(dudx, -2) - dudx
        return dudx[:-2]/2/dx
    
    def get_grid(self):
        return np.array(self.grid_v)

def find_A(u, x): 
    arr_u = []
    arr_x = []
    add = False 
    b = u[-1]
    c = x[-1]
    for i in range(len(u)-1, -1, -1):
        if u[i] > b:
            add = True
            b = u[i]
            c = x[i]
        elif add:
            add = False
            arr_u.append(b)
            arr_x.append(c)
    return np.array(arr_u[::-1]), np.array(arr_x[::-1])

"""
1D wave equation with Dirichlet or Neumann conditions
and variable wave velocity::
 u, x, t, cpu = solver(I, V, f, c, U_0, U_L, L, dt, C, T,
                       user_action=None, version='scalar',
                       stability_safety_factor=1.0)
Solve the wave equation u_tt = (c**2*u_x)_x + f(x,t) on (0,L) with
u=U_0 or du/dn=0 on x=0, and u=u_L or du/dn=0
on x = L. If U_0 or U_L equals None, the du/dn=0 condition
is used, otherwise U_0(t) and/or U_L(t) are used for Dirichlet cond.
Initial conditions: u=I(x), u_t=V(x).
T is the stop time for the simulation.
dt is the desired time step.
C is the Courant number (=max(c)*dt/dx).
stability_safety_factor enters the stability criterion:
C <= stability_safety_factor (<=1).
I, f, U_0, U_L, and c are functions: I(x), f(x,t), U_0(t),
U_L(t), c(x).
U_0 and U_L can also be 0, or None, where None implies
du/dn=0 boundary condition. f and V can also be 0 or None
(equivalent to 0). c can be a number or a function c(x).
user_action is a function of (u, x, t, n) where the calling code
can add visualization, error computations, data analysis,
store solutions, etc.
"""
def solver_long(
    I, V, f, c, U_0, U_L, L, dt, C, T, a,
    user_action=None, version='scalar',
    stability_safety_factor=1.0):
    """Solve u_tt=a(x)(c^2*u_x)_x + f on (0,L)x(0,T]."""

    # --- Compute time and space mesh ---
    Nt = int(round(T/dt))
    t = np.linspace(0, Nt*dt, Nt+1)      # Mesh points in time
    # --- Matrix for u(x, t) --
    u_grid = []

    # Find max(c) using a fake mesh and adapt dx to C and dt
    if isinstance(c, (float,int)):
        c_max = c
    elif callable(c):
        c_max = max([c(x_) for x_ in np.linspace(0, L, 101)])
    # Find min(a) using a fake mesh and adapt dx to C and dt
    if isinstance(a, (float,int)):
        a_min = np.sqrt(a)
    elif callable(a):
        a_min = np.sqrt(min([a(x_) for x_ in np.linspace(0, L, 101)]))

    dx = dt*c_max*a_min/(stability_safety_factor*C)
    Nx = int(round(L/dx))
    print('N_x = ', Nx)
    x = np.linspace(0, L, Nx+1)          # Mesh points in space
    # Make sure dx and dt are compatible with x and t
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Make c(x) available as array
    if isinstance(c, (float,int)):
        c = np.zeros(x.shape) + c
    elif callable(c):
        # Call c(x) and fill array c
        c_ = np.zeros(x.shape)
        for i in range(Nx+1):
            c_[i] = c(x[i])
        c = c_

    # Make a(x) available as array
    if isinstance(a, (float,int)):
        a = np.zeros(x.shape) + a
    elif callable(a):
        # Call a(x) and fill array c
        a_ = np.zeros(x.shape)
        for i in range(Nx+1):
            a_[i] = a(x[i])
        a = a_

    q = c**2
    C2 = (dt/dx)**2; dt2 = dt*dt    # Help variables in the scheme

    # --- Wrap user-given f, I, V, U_0, U_L if None or 0 ---
    if f is None or f == 0:
        f = (lambda x, t: 0) if version == 'scalar' else \
            lambda x, t: np.zeros(x.shape)
    if I is None or I == 0:
        I = (lambda x: 0) if version == 'scalar' else \
            lambda x: np.zeros(x.shape)
    if V is None or V == 0:
        V = (lambda x: 0) if version == 'scalar' else \
            lambda x: np.zeros(x.shape)
    if U_0 is not None:
        if isinstance(U_0, (float,int)) and U_0 == 0:
            U_0 = lambda t: 0
    if U_L is not None:
        if isinstance(U_L, (float,int)) and U_L == 0:
            U_L = lambda t: 0

    # --- Allocate memomry for solutions ---
    u     = np.zeros(Nx+1)   # Solution array at new time level
    u_n   = np.zeros(Nx+1)   # Solution at 1 time level back
    u_nm1 = np.zeros(Nx+1)   # Solution at 2 time levels back

    # --- Valid indices for space and time mesh ---
    Ix = range(0, Nx+1)
    It = range(0, Nt+1)

    # --- Load initial condition into u_n ---
    for i in range(0,Nx+1):
        u_n[i] = I(x[i])

    u_grid.append(u_n.copy()) #add to matrix init condition

    if user_action is not None:
        user_action(u_n, x, t, 0)

    # --- Special formula for the first step ---
    for i in Ix[1:-1]:
        u[i] = u_n[i] + dt*V(x[i]) + \
            a[i]*0.5*C2*(0.5*(q[i] + q[i+1])*(u_n[i+1] - u_n[i]) - \
                0.5*(q[i] + q[i-1])*(u_n[i] - u_n[i-1])) + \
            a[i]*0.5*dt2*f(x[i], t[0])

    i = Ix[0]
    if U_0 is None:
        # Set boundary values (x=0: i-1 -> i+1 since u[i-1]=u[i+1]
        # when du/dn = 0, on x=L: i+1 -> i-1 since u[i+1]=u[i-1])
        ip1 = i+1
        im1 = ip1  # i-1 -> i+1
        u[i] = u_n[i] + dt*V(x[i]) + \
               a[i]*0.5*C2*(0.5*(q[i] + q[ip1])*(u_n[ip1] - u_n[i])  - \
                       0.5*(q[i] + q[im1])*(u_n[i] - u_n[im1])) + \
            a[i]*0.5*dt2*f(x[i], t[0])
    else:
        u[i] = U_0(dt)

    i = Ix[-1]
    if U_L is None:
        im1 = i-1
        ip1 = im1  # i+1 -> i-1
        u[i] = u_n[i] + dt*V(x[i]) + \
               a[i]*0.5*C2*(0.5*(q[i] + q[ip1])*(u_n[ip1] - u_n[i])  - \
                       0.5*(q[i] + q[im1])*(u_n[i] - u_n[im1])) + \
            a[i]*0.5*dt2*f(x[i], t[0])
    else:
        u[i] = U_L(dt)

    if user_action is not None:
        user_action(u, x, t, 1)

    u_grid.append(u.copy())

    # Update data structures for next step
    #u_nm1[:] = u_n;  u_n[:] = u  # safe, but slower
    u_nm1, u_n, u = u_n, u, u_nm1

    # --- Time loop ---
    for n in It[1:-1]:
        # Update all inner points
        if version == 'scalar':
            for i in Ix[1:-1]:
                u[i] = - u_nm1[i] + 2*u_n[i] + \
                    a[i]*C2*(0.5*(q[i] + q[i+1])*(u_n[i+1] - u_n[i])  - \
                        0.5*(q[i] + q[i-1])*(u_n[i] - u_n[i-1])) + \
                    a[i]*dt2*f(x[i], t[n])

        elif version == 'vectorized':
            u[1:-1] = - u_nm1[1:-1] + 2*u_n[1:-1] + \
                a[1:-1]*C2*(0.5*(q[1:-1] + q[2:])*(u_n[2:] - u_n[1:-1]) -
                    0.5*(q[1:-1] + q[:-2])*(u_n[1:-1] - u_n[:-2])) + \
                a[1:-1]*dt2*f(x[1:-1], t[n])
        else:
            raise ValueError('version=%s' % version)

        # Insert boundary conditions
        i = Ix[0]
        if U_0 is None:
            # Set boundary values
            # x=0: i-1 -> i+1 since u[i-1]=u[i+1] when du/dn=0
            # x=L: i+1 -> i-1 since u[i+1]=u[i-1] when du/dn=0
            ip1 = i+1
            im1 = ip1
            u[i] = - u_nm1[i] + 2*u_n[i] + \
                   a[i]*C2*(0.5*(q[i] + q[ip1])*(u_n[ip1] - u_n[i])  - \
                       0.5*(q[i] + q[im1])*(u_n[i] - u_n[im1])) + \
                    a[i]*dt2*f(x[i], t[n])
        else:
            u[i] = U_0(t[n+1])

        i = Ix[-1]
        if U_L is None:
            im1 = i-1
            ip1 = im1
            u[i] = - u_nm1[i] + 2*u_n[i] + \
                   a[i]*C2*(0.5*(q[i] + q[ip1])*(u_n[ip1] - u_n[i])  - \
                       0.5*(q[i] + q[im1])*(u_n[i] - u_n[im1])) + \
            a[i]*dt2*f(x[i], t[n])
        else:
            u[i] = U_L(t[n+1])

        if user_action is not None:
            if user_action(u, x, t, n+1):
                break
        u_grid.append(u.copy())
        # Update data structures for next step
        u_nm1, u_n, u = u_n, u, u_nm1
    return x, t, np.array(u_grid)
