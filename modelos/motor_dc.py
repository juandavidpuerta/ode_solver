# ode_solver/modelos/motor_dc.py
import numpy as np
from integradores.solver import integrar

class MotorDC:
    """
    Motor DC con excitación independiente y saturación magnética.

    Estado: y = [If, Ia, w]
    """

    def __init__(self, Ra, La, Rf, J, B, Kv, Kt,
                 phi_max, Isat, Lf0, Lf_min,
                 Va, Vf, tau_L):

        # Parámetros eléctricos
        self.Ra = Ra;  self.La = La;  self.Rf = Rf
        # Parámetros mecánicos
        self.J  = J;   self.B  = B
        # Constantes electromagnéticas
        self.Kv = Kv;  self.Kt = Kt
        # Saturación magnética
        self.phi_max = phi_max;  self.Isat = Isat
        self.Lf0 = Lf0;         self.Lf_min = Lf_min
        # Entradas
        self.Va = Va;  self.Vf = Vf;  self.tau_L = tau_L

        # Resultados (se llenan al simular)
        self.t   = None
        self.If  = None
        self.Ia  = None
        self.w   = None
        self.tau = None
        self.phi = None
        self.Lf  = None
        self.Kfi = None

    def derivadas(self, t, y):
        """f(t, y) del sistema — compatible con integrar()."""
        If, Ia, w = y[0], y[1], y[2]

        x       = If / self.Isat
        tanh_x  = np.tanh(x)
        sech2_x = 1.0 / np.cosh(x)**2

        Kfi = (self.phi_max / self.Isat) * sech2_x
        Lf  = self.Lf_min + (self.Lf0 - self.Lf_min) * sech2_x
        dLf = -(2.0 * (self.Lf0 - self.Lf_min) / self.Isat) * tanh_x * sech2_x

        dIf = (self.Vf - self.Rf * If) / max(Lf + If * dLf, 0.05 * self.Lf0)
        dIa = (self.Va - self.Ra * Ia - self.Kv * Kfi * If * w) / self.La
        dw  = (self.Kt * Kfi * If * Ia - self.B * w - self.tau_L) / (self.J * 2 * np.pi)

        return np.array([dIf, dIa, dw])

    def simular(self, t0=0.0, tf=30.0, h=1e-3, metodo='rk4', y0=None):
        """
        Simula el motor en [t0, tf].

        Parámetros
        ----------
        t0, tf : float   intervalo de simulación
        h      : float   paso de integración
        metodo : str     'rk1', 'rk2', 'rk3', 'rk4'
        y0     : list    condición inicial [If, Ia, w]

        Retorna
        -------
        self   permite encadenamiento: motor.simular().graficar()
        """
        if y0 is None:
            y0 = [0.0, 0.0, 0.0]

        t_arr, y_arr = integrar(self.derivadas, t0, tf, y0, h, metodo)

        self.t  = t_arr
        self.If = y_arr[:, 0]
        self.Ia = y_arr[:, 1]
        self.w  = y_arr[:, 2]

        # Variables auxiliares post-proceso
        x        = self.If / self.Isat
        tanh_x   = np.tanh(x)
        sech2_x  = 1.0 / np.cosh(x)**2
        self.phi = self.phi_max * tanh_x
        self.Kfi = (self.phi_max / self.Isat) * sech2_x
        self.Lf  = self.Lf_min + (self.Lf0 - self.Lf_min) * sech2_x
        self.tau = self.Kt * self.Kfi * self.If * self.Ia

        return self