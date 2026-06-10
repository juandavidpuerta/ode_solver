# ode_solver/integradores/multipasos.py
"""
Métodos multipaso de orden 4 con arranque RK4.

AB4 (Adams-Bashforth, explícito):
    y_{n+1} = y_n + (h/24)(55 f_n - 59 f_{n-1} + 37 f_{n-2} - 9 f_{n-3})

PC4 (Predictor-Corrector, esquema PECE):
    P: AB4 predice y*_{n+1}
    E: evalúa f* = f(t_{n+1}, y*_{n+1})
    C: AM4 corrige
       y_{n+1} = y_n + (h/24)(9 f* + 19 f_n - 5 f_{n-1} + f_{n-2})

Ambos necesitan 4 valores iniciales; los 3 primeros pasos se generan
con RK4 (el método multipaso no tiene historia al arrancar).
"""
import numpy as np
from integradores.runge_kutta import rk4_paso

def _preparar(f, t0, tf, y0, h):
    """Crea malla, arranque RK4 e historia inicial. Devuelve también h_real."""
    n      = max(int(round((tf - t0) / h)), 4) + 1
    t_arr  = np.linspace(t0, tf, n)
    h_real = (tf - t0) / (n - 1)
    y0     = np.atleast_1d(np.array(y0, dtype=float))
    y_arr  = np.zeros((n, len(y0)))
    y_arr[0] = y0
    for i in range(3):
        y_arr[i+1] = rk4_paso(f, t_arr[i], y_arr[i], h_real)
    f_hist = [f(t_arr[i], y_arr[i]) for i in range(3)]
    return t_arr, y_arr, f_hist, n, h_real


def adams_bashforth4(f, t0, tf, y0, h):
    """Adams-Bashforth de 4 pasos (explícito, orden 4)."""
    t_arr, y_arr, f_hist, n, h = _preparar(f, t0, tf, y0, h)   # h ahora = h_real
    for i in range(3, n - 1):
        f_n = f(t_arr[i], y_arr[i])
        f_n1, f_n2, f_n3 = f_hist[-1], f_hist[-2], f_hist[-3]
        y_arr[i+1] = y_arr[i] + (h/24) * (55*f_n - 59*f_n1 + 37*f_n2 - 9*f_n3)
        f_hist.append(f_n); f_hist.pop(0)
    return t_arr, y_arr


def predictor_corrector(f, t0, tf, y0, h):
    """Predictor-Corrector AB4 + AM4, esquema PECE (orden 4)."""
    t_arr, y_arr, f_hist, n, h = _preparar(f, t0, tf, y0, h)   # h ahora = h_real
    for i in range(3, n - 1):
        f_n = f(t_arr[i], y_arr[i])
        f_n1, f_n2, f_n3 = f_hist[-1], f_hist[-2], f_hist[-3]
        y_pred = y_arr[i] + (h/24) * (55*f_n - 59*f_n1 + 37*f_n2 - 9*f_n3)
        f_pred = f(t_arr[i+1], y_pred)
        y_arr[i+1] = y_arr[i] + (h/24) * (9*f_pred + 19*f_n - 5*f_n1 + f_n2)
        f_hist.append(f_n); f_hist.pop(0)
    return t_arr, y_arr
