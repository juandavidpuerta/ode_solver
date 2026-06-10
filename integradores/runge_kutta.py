# ode_solver/integradores/runge_kutta.py
"""
Métodos de Runge-Kutta de un paso (órdenes 1 a 4).

Cada función `rkN_paso` avanza UN paso de tamaño h del problema
    dy/dt = f(t, y)
y devuelve la estimación de y en t + h. Trabajan con escalares
o con vectores de NumPy (sistemas de EDOs).
"""
import numpy as np


def rk1_paso(f, t, y, h):
    """RK1 — Euler explícito.   y_{n+1} = y_n + h·f(t_n, y_n)"""
    return y + h * f(t, y)


def rk2_paso(f, t, y, h):
    """RK2 — Heun.   promedio de la pendiente en t y en t+h."""
    k1 = f(t,     y)
    k2 = f(t + h, y + h * k1)
    return y + (h/2) * (k1 + k2)


def rk3_paso(f, t, y, h):
    """RK3 — Kutta de tercer orden."""
    k1 = f(t,       y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h,   y - h*k1 + 2*h*k2)
    return y + (h/6) * (k1 + 4*k2 + k3)


def rk4_paso(f, t, y, h):
    """RK4 — Clásico (cuatro evaluaciones por paso)."""
    k1 = f(t,       y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h,   y + h   * k3)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)