# ode_solver/integradores/solver.py
"""
Despachador unificado de métodos de integración.
Reconoce métodos de un paso (RK) y multipaso (AB4, Predictor-Corrector).
"""
import numpy as np
from integradores.runge_kutta import rk1_paso, rk2_paso, rk3_paso, rk4_paso
from integradores.multipasos import adams_bashforth4, predictor_corrector

# Métodos de UN PASO — el bucle se ejecuta aquí
_PASOS = {
    'rk1': rk1_paso,
    'rk2': rk2_paso,
    'rk3': rk3_paso,
    'rk4': rk4_paso,
}

# Métodos MULTIPASO — ya traen su propio bucle completo
_MULTIPASO = {
    'ab4': adams_bashforth4,
    'pc4': predictor_corrector,
}


def integrar(f, t0, tf, y0, h, metodo='rk4'):
    """
    Integra dy/dt = f(t, y) en [t0, tf] con el método indicado.

    Parámetros
    ----------
    f      : callable    f(t, y) → array de derivadas
    t0, tf : float       intervalo de integración
    y0     : array       condición inicial
    h      : float       paso de integración
    metodo : str         'rk1','rk2','rk3','rk4','ab4','pc4'

    Retorna
    -------
    t_arr : np.ndarray  shape (n,)
    y_arr : np.ndarray  shape (n, len(y0))
    """
    if metodo in _PASOS:
        paso   = _PASOS[metodo]
        n      = max(int(round((tf - t0) / h)), 1) + 1
        t_arr  = np.linspace(t0, tf, n)        # extremo EXACTO en tf
        h_real = (tf - t0) / (n - 1)           # paso = espaciamiento de la malla
        y0     = np.atleast_1d(np.array(y0, dtype=float))
        y_arr  = np.zeros((n, len(y0)))
        y_arr[0] = y0
        for i in range(n - 1):
            y_arr[i+1] = paso(f, t_arr[i], y_arr[i], h_real)
        return t_arr, y_arr

    elif metodo in _MULTIPASO:
        return _MULTIPASO[metodo](f, t0, tf, y0, h)

    else:
        disponibles = list(_PASOS) + list(_MULTIPASO)
        raise ValueError(f"Método '{metodo}' no reconocido. Usa: {disponibles}")