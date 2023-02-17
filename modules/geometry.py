import numpy as np

def get_parametr_matrix(result_dict, label):
    """
    Method make marix dimension [layers x cells] 
    result_dict: <dict> from ozvb
    label: name of parametr ('p', 'x', etc.)
    """
    if not label in ['x', 'p', 'T', 'u']:
         raise TypeError('undefine "label": ', label)
    matrix = list()
    for layer in result_dict['layers']:
        matrix.append(layer[label])
    return np.array(matrix)

def get_time(result_dict):
    matrix = list()
    for layer in result_dict['layers']:
        matrix.append(layer['t'])
    return np.array(matrix)

def get_parametr_1d(result_dict, label):
    """
    """
    if not label in ['x', 'p', 'u', 'T']:
        raise TypeError('undefine "label": ', label)
    matrix = list()
    for layer in result_dict['layers']:
        matrix.append(layer[label][-1])
    return np.array(matrix)



class Cannon():

    n = 100 # cells for cannon strenght
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

    def __init__(self, diametr, coordinate, pressure, l0) -> None:
        """_summary_

        Args:
            diametr (_type_): caliber of the gun
            coordinate (_type_): matrix_x
            pressure (_type_): matrix_p
            l0 (_type_): tube_lenght

        Raises:
            ValueError: _description_
        """
        if coordinate.shape != pressure.shape:
            raise ValueError('shape not alignment')
        self.l0 = l0
        self.diametr = diametr
        self.__matrix_x = coordinate
        self.__matrix_p = pressure
        self.coordinate = None
        self.pressure = None
        self.r_inside = None
        self.r_outside = None
        self.r_inside_coordinate = None
        self.pressure_tube = None
        self.n_real = None

    def __inside_geometry(self):
        """
        Geometry of combustion chamber and tube
        """
        diametr = self.diametr
        W_0 = diametr**2*np.math.pi/4 * np.abs(self.l0)
        l_2 = 0.55*diametr if np.sqrt(self.hi) <= 1.25 else 0.9*diametr
        l_6 = 2.5*diametr
        l_7 = 0.1*diametr
        d_4 = diametr + self.cone_k7*l_7
        d_3 = d_4 + self.cone_k6*l_6 
        d_2 = d_3 + self.cone_k2*l_2
        d_k = self.bottle_capacity*diametr
        W_6 = np.math.pi/12*l_6*(d_3**2 + d_3*d_4 + d_4**2)
        W_2 = np.math.pi/12*l_2*(d_3**2 + d_3*d_2 + d_2**2)
        W_7 = np.math.pi/12*l_7*(d_4**2 + d_4*diametr + diametr**2)
        W_1 = 1.1*W_0-W_2-W_6-W_7+self.W_sn
        l_1 = W_1*12/np.math.pi / (d_k**2 + d_k*d_2 + d_2**2)

        self.r_inside_coordinate = np.cumsum([0, l_1, l_2, l_6, l_7, self.__matrix_x[-1][-1]-self.l0])
        self.r_inside = np.array([d_k, d_2, d_3, d_4, diametr, diametr]) / 2
        self.coordinate = np.linspace(0, self.r_inside_coordinate[-1], Cannon.n)

    def __outside_geometry(self):
        """
        Geometry of outer shell
        """
        if self.r_inside_coordinate is None:
            raise ValueError("empty inside coordinate")
        if self.r_inside is None:
            raise ValueError("empty inside radius")
        if self.pressure is None:
            raise ValueError("empty pressure")
        if 0.75*self.pressure.max() >= self.sigma_steel:
            raise ValueError("pressure in tube more than 3/4 sigma (steel)")

        sqr = (3*self.sigma_steel+2*self.pressure*self.n_safety) / (3*self.sigma_steel-4*self.pressure*self.n_safety)
        
        if min(sqr) < 0:
            raise ValueError("pressure in tube destroy cannon")
        radius_inside = np.interp(self.coordinate, self.r_inside_coordinate, self.r_inside)
        radius_outside = radius_inside*np.sqrt(sqr)
        self.r_outside = np.array([max(radius_outside[i], self.k_min_r_otside*radius_inside[i]) for i in range(Cannon.n)])

    def cannon_geometry(self):
        """
        Method for construct cannon
        """   

        self.__inside_geometry()
        self.__pressure_on_tube()
        self.__outside_geometry()
        # find inside radius in each of coordinate
        r_inside = np.interp(self.coordinate, self.r_inside_coordinate, self.r_inside)
        a_21 = self.r_outside/r_inside
        self.pressure_tube = 3/2*self.sigma_steel*(a_21**2 - 1)/(2*a_21**2 + 1)
        self.n_real = self.pressure_tube/self.pressure
        if min(self.n_real) < 1:
            ValueError("check of real pressure fail")

    def get_volume(self):
        """
        Calculate volume of cannon 
        """
        if self.coordinate is None:
            self.cannon_geometry()
        x1 = self.coordinate
        x2 = self.r_inside_coordinate
        r1 = self.r_outside
        r2 = self.r_inside

        volume_1 = 1/3*np.math.pi*np.sum((x1[1:]-x1[:-1])*(r1[:-1]**2 + r1[:-1]*r1[1:] + r1[1:]**2)) 
        volume_2 = 1/3*np.math.pi*np.sum((x2[1:]-x2[:-1])*(r2[:-1]**2 + r2[:-1]*r2[1:] + r2[1:]**2)) 
        return volume_1-volume_2
 

    def get_mass(self):
        return self.get_volume()*self.ro

    def __pressure_on_tube(self):
        """
        Method for calculate pressure distribution along the cannon's tube
        """
        if self.__matrix_p.shape != self.__matrix_x.shape:
            raise ValueError('shape not alignment')

        pressure_layers = np.zeros((self.__matrix_p.shape[0], Cannon.n))
        for i in range(self.__matrix_p.shape[0]):
            pressure_layers[i] = np.interp(self.coordinate, self.__matrix_x[i], self.__matrix_p[i], left=0, right=0)
        self.pressure = np.max(pressure_layers, axis=0)

    def make_p_xt(self):
        """
        """
        if self.__matrix_p.shape != self.__matrix_x.shape:
            raise ValueError('shape not alignment')

        pressure_layers = np.zeros((self.__matrix_p.shape[0], Cannon.n))
        for i in range(self.__matrix_p.shape[0]):
            pressure_layers[i] = np.interp(self.coordinate, self.__matrix_x[i], self.__matrix_p[i], left=0, right=0)
        return pressure_layers
