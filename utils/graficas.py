# ode_solver/utils/graficas.py
import matplotlib.pyplot as plt
import numpy as np


def graficar_motor(motor, titulo_extra=''):
    """
    Grafica los resultados de una simulación de MotorDC.

    Parámetros
    ----------
    motor       : MotorDC ya simulado
    titulo_extra: str opcional para identificar el método
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Motor DC — {titulo_extra}', fontsize=14)

    axs[0,0].plot(motor.t, motor.Ia)
    axs[0,0].set_title('Corriente de armadura')
    axs[0,0].set_xlabel('Tiempo [s]')
    axs[0,0].set_ylabel('Ia [A]')
    axs[0,0].grid(True)

    axs[0,1].plot(motor.t, motor.If)
    axs[0,1].set_title('Corriente de campo')
    axs[0,1].set_xlabel('Tiempo [s]')
    axs[0,1].set_ylabel('If [A]')
    axs[0,1].grid(True)

    axs[1,0].plot(motor.t, motor.w)
    axs[1,0].set_title('Velocidad mecánica')
    axs[1,0].set_xlabel('Tiempo [s]')
    axs[1,0].set_ylabel('ω [rev/s]')
    axs[1,0].grid(True)

    axs[1,1].plot(motor.t, motor.tau)
    axs[1,1].set_title('Torque electromagnético')
    axs[1,1].set_xlabel('Tiempo [s]')
    axs[1,1].set_ylabel('τ [N·m]')
    axs[1,1].grid(True)

    plt.tight_layout()
    plt.show()


def comparar_metodos(motor_class, params, metodos, t0=0.0, tf=30.0, h=1e-3, y0=None):
    """
    Simula el mismo motor con distintos métodos y compara en una figura.
    Parámetros
    ----------
    motor_class : clase MotorDC
    params      : dict con los parámetros del motor
    metodos     : list de str, ej: ['rk1', 'rk2', 'rk3', 'rk4']
    t0, tf, h   : parámetros de simulación
    y0          : condición inicial
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    fig.suptitle(f'Comparación de métodos  —  h = {h}', fontsize=14)

    etiquetas = {
        'rk1': 'RK1 — Euler',
        'rk2': 'RK2 — Heun',
        'rk3': 'RK3 — Kutta',
        'rk4': 'RK4 — Clásico',
        'ab4': 'AB4 — Adams-Bashforth',
        'pc4': 'PC4 — Predictor-Corrector',
    }

    estilos = ['--', ':', '-.', '-', '--', ':']
    colores = ['gray', 'orange', 'blue', 'red', 'green', 'purple']

    for i, metodo in enumerate(metodos):
        motor = motor_class(**params)
        motor.simular(t0=t0, tf=tf, h=h, metodo=metodo, y0=y0)

        estilo = estilos[i % len(estilos)]
        color  = colores[i % len(colores)]
        label  = etiquetas.get(metodo, metodo)

        axs[0,0].plot(motor.t, motor.If,  linestyle=estilo, color=color, label=label)
        axs[0,1].plot(motor.t, motor.Ia,  linestyle=estilo, color=color, label=label)
        axs[1,0].plot(motor.t, motor.w,   linestyle=estilo, color=color, label=label)
        axs[1,1].plot(motor.t, motor.tau, linestyle=estilo, color=color, label=label)

    axs[0,0].set_title('Corriente de campo If');    axs[0,0].set_xlabel('t [s]'); axs[0,0].set_ylabel('If [A]')
    axs[0,1].set_title('Corriente de armadura Ia'); axs[0,1].set_xlabel('t [s]'); axs[0,1].set_ylabel('Ia [A]')
    axs[1,0].set_title('Velocidad mecánica ω');     axs[1,0].set_xlabel('t [s]'); axs[1,0].set_ylabel('ω [rev/s]')
    axs[1,1].set_title('Torque electromagnético');  axs[1,1].set_xlabel('t [s]'); axs[1,1].set_ylabel('τ [N·m]')

    for ax in axs.flat:
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()