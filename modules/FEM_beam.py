"""
FEM modul for static deflection 
of the beam research
"""
import numpy as np
import matplotlib.pyplot as plt

class FEMBeam():

    E = 200e9
    g = 9.81
    rho = 7850

    def __init__(self, coordination: np.ndarray, radius_1: np.ndarray, 
                 raius_2: np.ndarray, fi: float) -> None:
        # check dimensions
        # check r2 >= r1

        self.r_1 = radius_1.copy()
        self.r_2 = raius_2.copy()
        self.x = coordination.copy()
        self.n_fem = len(self.x) - 1
        self.fi = fi
        self.__gen_finite_elemants()
        pass

    def __gen_finite_elemants(self):
        self.R1 = (self.r_1[0:-1] + self.r_1[1:])/2
        self.R2 = (self.r_2[0:-1] + self.r_2[1:])/2
        self.l = self.x[1:] - self.x[:-1]
        self.square = np.pi*(self.R2**2 - self.R1**2)
        pass
    
    def init_y_deflect(self, extra_stress=0):
        #check extra_stress
        self.q_y = -self.rho*self.g*self.square*np.cos(self.fi)
        self.J_z = np.pi*(self.R2**4 - self.R1**4)/4
        self.Q = np.zeros(self.n_fem+1) + extra_stress
        self.M = np.zeros(self.n_fem+1)
        self.teta = np.zeros(self.n_fem+1)
        self.deflect_y = np.zeros(self.n_fem+1)

    def init_x_deflect(self, extra_stress=0):
        #check extra_stress
        self.q_x = -self.rho*self.g*self.square*np.sin(self.fi)
        self.P = np.zeros(self.n_fem+1) + extra_stress
        self.deflect_x = np.zeros(self.n_fem+1)
        pass

    def calc_y_deflect(self):
        # check for empty Q, M and etc 

        for i in range(1, self.n_fem+1):
            j = i-1 # index for FEM cell
            self.Q[i] = self.Q[i-1] + self.l[j]*self.q_y[j]
            self.M[i] = self.M[i-1] + self.Q[i-1]*self.l[j] + self.q_y[j]*self.l[j]**2/2
            #self.teta[i] = self.teta[i-1] + 1/(self.E * self.J_z[j]) * \
            #            (self.M[i-1]*self.l[j] + self.Q[i-1]*self.l[j]**2/2 + self.q_y[j]*self.l[j]**3/6)
            
        #self.teta_max = self.teta[-1]
        #self.teta = self.teta_max - self.teta
        self.__reverse()
        for i in range(1, self.n_fem+1):
            j = i-1 # index for FEM cel
            self.teta[i] = self.teta[i-1] + 1/(self.E * self.J_z[j]) * \
                        (self.M[i-1]*self.l[j] + self.Q[i-1]*self.l[j]**2/2 + self.q_y[j]*self.l[j]**3/6)
            self.deflect_y[i] = self.deflect_y[i-1] + self.teta[i-1]*self.l[j] + 1/(self.E * self.J_z[j]) * \
                        (self.M[i-1]*self.l[j]**2/2 + self.Q[i-1]*self.l[j]**3/6 + self.q_y[j]*self.l[j]**4/24)
        
        self.teta_max = self.teta[-1]
        self.deflect_y_max = self.deflect_y[-1]
        pass

    def calc_x_deflect(self):
        # check for empty P

        for i in range(1, self.n_fem+1):
            j = i-1 # index for FEM cell
            self.P[i] = self.P[i-1] + self.q_x[j]*self.l[j]
            self.deflect_x[i] = self.deflect_x[i-1] + 1/(self.square[j]*self.E)*\
                                (self.P[i-1]*self.l[j] + self.q_x[j]*self.l[j]**2/2)
        self.deflect_x_max = self.deflect_x[-1]
        self.P = self.P[::-1]
        self.deflect_x = self.deflect_x[::-1]
        pass

    def __reverse(self):
        #self.teta = self.teta[::-1]
        self.l = self.l[::-1]
        self.J_z =self.J_z[::-1]
        self.M = self.M[::-1]
        self.Q = -self.Q[::-1]
        self.q_y = self.q_y[::-1]
    
    @property
    def sigma_max(self):
        W_z = self.J_z/self.R2
        W_z = np.hstack((W_z, W_z[-1]))
        return self.M/W_z
    

def draw_geometry(beam: FEMBeam, axis: plt.Axes):
    fs = 16
    width = 0.75
    axis.plot(beam.x, beam.r_2, color='black', linewidth=width)
    axis.plot(beam.x, beam.r_1, color='black', linewidth=width)
    axis.vlines(beam.x[0], 0, beam.r_2[0], color='black', linewidth=width)
    axis.vlines(beam.x[-1], 0, beam.r_2[-1], color='black', linewidth=width)
    axis.fill_between(beam.x, beam.r_1, beam.r_2, facecolor="white", hatch="//")
    axis.set_xlabel(r'$x$, м',  fontsize=fs)
    axis.set_ylabel(r'$r$, м',  fontsize=fs)
    axis.grid(linewidth=0.5)
    return axis

def draw_epure(beam: FEMBeam, axis: plt.Axes, epure='Q'):
    fs = 16
    width = 0.75
    epure_type = {'Q':{'x_lab':r'$x$, м', 'y_lab':r'$Q$, H'},
                  'M':{'x_lab':r'$x$, м', 'y_lab':r'$M$, Hм'},
                  'teta':{'x_lab':r'$x$, м', 'y_lab':r'$\theta, \circ$'},
                  'deflect_y':{'x_lab':r'$x$, м', 'y_lab':r'$\Delta y$, м'},
                  'sigma_max':{'x_lab':r'$x$, м', 'y_lab':r'$\sigma$, МПа'},
                  'P':{'x_lab':r'$x$, м', 'y_lab':r'$P$, H'},
                  'deflect_x':{'x_lab':r'$x$, м', 'y_lab':r'$\Delta x$, H'},
                 }
    if not epure in epure_type:
        raise TypeError('No correct epure type')
        
    ep_dr = getattr(beam, epure)
    if epure == 'teta': ep_dr = np.degrees(ep_dr)
    if epure == 'sigma_max': ep_dr /= 1e6
    
    axis.plot(beam.x, ep_dr , color='black', linewidth=width)
    axis.fill_between(beam.x, 0, ep_dr , facecolor="white", hatch="|")
    axis.set_xlabel(epure_type[epure]['x_lab'],  fontsize=fs)
    axis.set_ylabel(epure_type[epure]['y_lab'],  fontsize=fs)
    axis.grid(linewidth=0.5)
    return axis