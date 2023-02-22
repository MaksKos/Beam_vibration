import numpy as np
from pyballistics import ozvb_lagrange
from scipy.interpolate import interp1d

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

def get_parametr_pr(result_dict, label):
    """
    """
    if not label in ['x', 'p', 'u', 'T']:
        raise TypeError('undefine "label": ', label)
    matrix = list()
    for layer in result_dict['layers']:
        matrix.append(layer[label][-1])
    return np.array(matrix)

class Pressure:

    g = 9.81
    _smoothing = 0.3 #[0, 1]

    def __init__(self, init_dict) -> None:
        
        self.init_dict = init_dict
        self.q = init_dict['init_conditions']['q']
        self.wq = init_dict['powders'][0]['omega']/\
                        self.q
        self.l_tube = init_dict['stop_conditions']['x_p']

    def calculate(self):
        result = ozvb_lagrange(self.init_dict)
        if result['stop_reason'] != 'x_p':
            raise ValueError(f'Stop reason:{result["stop_reason"]}')
        self.arr_t = get_time(result)
        self.t_end = self.arr_t[-1]
        self.arr_x_pr = get_parametr_pr(result, 'x')
        self.arr_p_pr = get_parametr_pr(result, 'p')
        matrix_x = get_parametr_matrix(result, 'x')
        matrix_p = get_parametr_matrix(result, 'p')
        self.arr_p_m = np.mean(matrix_p, axis=1)
        #add layre to 'p' for shape alignment with 'x'
        self.matrix_p = np.row_stack((matrix_p.T, matrix_p.T[-1])).T
        #calculate coordinate from 0 point
        self.l0 = np.abs(matrix_x[0][0])
        self.matrix_x = matrix_x + self.l0

    def make_func(self, l_kam, l_pr):
        self._l_pr = l_pr
        self._l_kam = l_kam
        # function (t) [0, t_end]
        self.p_mean = interp1d(self.arr_t, self.arr_p_m, bounds_error=False, fill_value=(0, 0))
        self.p_pr = lambda t: self.p_mean(t) /(1 + 1/3 * self.wq)
        self.p_kn = lambda t: 1+0.5*self.wq*self.p_pr(t)
        self.x_pr = interp1d(self.arr_t, self.arr_x_pr+l_kam, bounds_error=False, fill_value=(0, 0))
        # function (x, t)
        self.p = np.vectorize(self._p)
        self.dpdx = np.vectorize(self._dp_dx)
        self.proj = np.vectorize(self._projectile)

    def _p (self, x: np.ndarray, t: float):
    
        if t < 0 or t > self.t_end:
            return 0
        x_pr = self.x_pr(t)
        if x > x_pr or x < 0:
            return 0
        return self.p_kn(t) - 0.5*self.wq*self.p_pr(t)* x**2/x_pr**2

    def _dp_dx(self, x: np.ndarray, t: float):

        if t < 0 or t > self.t_end:
            return 0
        x_pr = self.x_pr(t)
        if x > x_pr or x < 0:
            return 0
        return - self.wq*self.p_pr(t)*x/x_pr**2
    
    def _projectile(self, x: float, t: float):

        if t < 0 or t > self.t_end:
            return 0
        x_pr0 = self.x_pr(t)
        x_pr = min(self.l_tube+self._l_kam,
                    x_pr0+self._l_pr)
        if x > x_pr or x < x_pr0:
            return 0
        
        stabil = self._smoothing*self.t_end
        if stabil < 1e-10:
            coef = 1
        else:
            coef = t/stabil
        if coef > 1:
            coef=1
        return self.q*self.g/self._l_pr*coef
    
    def set_smoothing(self, coef):
        
        if coef > 1 or coef < 0:
            print("No correct value: should in [0, 1]")
            return 0
        self._smoothing = coef