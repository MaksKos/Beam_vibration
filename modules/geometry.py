import numpy as np
from scipy.interpolate import interp1d

class Cannon():

    n = 100 #cells for cannon strenght
    cone_k1 = 1/75
    cone_k2 = 1/2.5
    cone_k6 = 1/200
    cone_k7 = 1/30
    hi = 1.5
    bottle_capacity = 1.25
    n_safety = 1.1
    sigma_steel = 10e8
    ro = 7856
    W_sn = 6.85*1e-4 
    k_min_r_otside = 1.5

    def __init__(self) -> None:
        self.r_inside = None
        self.r_inside_coordinate = None
        self.r_outside = None
        self.r_inside_coordinate = None


    def cannon_geometry(self, diametr, l0, l_tube):
        """
        Geometry of combustion chamber and tube
        """
        self.l0 = l0
        self.l_tube = l_tube
        W_0 = diametr**2*np.pi/4 * np.abs(self.l0)
        l_2 = 0.55*diametr if np.sqrt(self.hi) <= 1.25 else 0.9*diametr
        l_6 = 2.5*diametr
        l_7 = 0.1*diametr
        d_4 = diametr + self.cone_k7*l_7
        d_3 = d_4 + self.cone_k6*l_6 
        d_2 = d_3 + self.cone_k2*l_2
        d_k = self.bottle_capacity*diametr
        W_6 = np.pi/12*l_6*(d_3**2 + d_3*d_4 + d_4**2)
        W_2 = np.pi/12*l_2*(d_3**2 + d_3*d_2 + d_2**2)
        W_7 = np.pi/12*l_7*(d_4**2 + d_4*diametr + diametr**2)
        W_1 = 1.1*W_0-W_2-W_6-W_7+self.W_sn
        l_1 = W_1*12/np.pi / (d_k**2 + d_k*d_2 + d_2**2)

        self.r_inside_coordinate = np.cumsum([0, l_1, l_2, l_6, l_7, self.l_tube])
        self.r_inside = np.array([d_k, d_2, d_3, d_4, diametr, diametr]) / 2
        self.l_kam = np.sum([0, l_1, l_2, l_6, l_7])
        self.L = np.sum([0, l_1, l_2, l_6, l_7, self.l_tube])
    
    def inside_geometry(self, x, r1):
        if x.shape != r1.shape:
            raise ValueError(f'No same shape {x.shape} and {r1.shape}')
        self.r_inside = r1
        self.r_inside_coordinate = x
        
    def outside_geometry(self, x, r2):
        if x.shape != r2.shape:
            raise ValueError(f'No same shape {x.shape} and {r2.shape}')
        if x[-1]-self.r_inside_coordinate[-1] > 1e-5:
            raise ValueError('Different end')
        self.r_outside = r2
        self.r_outside_coordinate = x

    def make_func(self):
        self.r1 = interp1d(self.r_inside_coordinate, self.r_inside, 
                           bounds_error=False, fill_value=(0, 0))
        self.r2 = interp1d(self.r_outside_coordinate, self.r_outside, 
                           bounds_error=False, fill_value=(0, 0))
        
        self.F = lambda x: np.pi*(self.r2(x)**2 - self.r1(x)**2)
        self.S = lambda x: np.pi*self.r1(x)**2
        self.J = lambda x: np.pi/4 * (self.r2(x)**4 - self.r1(x)**4)

    def make_dsdx(self, nx):
        x = np.linspace(0, self.L, nx)
        dx = x[1]-x[0]
        dS_dx = self.d_dx(self.S(x), dx)
        self.dSdx = interp1d(x, dS_dx, bounds_error=False, fill_value=(0, 0))

    @staticmethod
    def d_dx(u_x, dx):
        dudx = np.zeros(u_x.shape[0]+2)
        dudx[1:-1] = u_x
        dudx[-1] = u_x[-1]
        dudx[0] = u_x[0]
        dudx = np.roll(dudx, -2) - dudx
        return dudx[:-2]/2/dx
    
    def make_p_xt(self, matrix_p, matrix_x, n_x):
        if matrix_p.shape != matrix_x.shape:
            raise ValueError('shape not alignment')
        
        coordinate = np.linspace(0, self.r_inside_coordinate[-1], n_x)
        pressure_layers = np.zeros((matrix_p.shape[0], n_x))
        for i in range(matrix_p.shape[0]):
            pressure_layers[i] = np.interp(coordinate, matrix_x[i], matrix_p[i], left=0, right=0)
        return pressure_layers
