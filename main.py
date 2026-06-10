# ode_solver/main.py
import sys
import os
import numpy as np

# Permite importar los módulos desde cualquier ubicación
sys.path.insert(0, os.path.dirname(__file__))

from modelos.motor_dc import MotorDC
from utils.graficas import graficar_motor, comparar_metodos

# --------------------------------------------------
# PARÁMETROS DEL MOTOR
# --------------------------------------------------
params = {
    'Ra': 0.9,   'La': 0.01,  'Rf': 440,
    'J':  3.0,   'B':  0.03,
    'Kv': 40.0,  'Kt': 6.4,
    'phi_max': 0.25,  'Isat': 1.25,
    'Lf0': 300.0,     'Lf_min': 30.0,
    'Va': 440.0, 'Vf': 440.0, 'tau_L': 50.0
}

# --------------------------------------------------
# 1. SIMULACIÓN SIMPLE CON RK4
# --------------------------------------------------
print("Simulando con RK4...")
motor = MotorDC(**params)
motor.simular(t0=0.0, tf=30.0, h=0.003343, metodo='rk4')
graficar_motor(motor, titulo_extra='RK4')

# --------------------------------------------------
# 2. COMPARACIÓN DE TODOS LOS MÉTODOS RK
# --------------------------------------------------
print("Comparando métodos RK1, RK2, RK3, RK4...")
comparar_metodos(
    motor_class = MotorDC,
    params      = params,
    metodos     = ['rk1', 'rk2', 'rk3', 'rk4', 'ab4', 'pc4'],
    t0=0.0, tf=30, h=0.003343
)

print("Listo.")

# --------------------------------------------------
# 3. ESTUDIO DE CONVERGENCIA DE LOS MÉTODOS
# --------------------------------------------------

from utils.convergencia import estudio_convergencia

print("\nEstudio de convergencia...")
estudio_convergencia(
    motor_class = MotorDC,
    params      = params,
    metodos     = ['rk1', 'rk2', 'rk3', 'rk4', 'ab4', 'pc4'],
    tf=0.2,
    h_list = np.logspace(np.log10(3e-3), np.log10(8e-4), 7)
)