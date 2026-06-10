# ode_solver/utils/convergencia.py
import numpy as np
import matplotlib.pyplot as plt
from integradores.solver import integrar


def estudio_convergencia(motor_class, params, metodos,
                         t0=0.0, tf=1.0, h_ref=1e-5,
                         h_list=None, y0=None):
    """
    Estudio de convergencia: error relativo vs paso h en log-log.
    La PENDIENTE de cada recta = orden del método.

    El error se mide en el EXTREMO t = tf (punto exacto de la malla,
    sin interpolación) y con tf corto para quedar dentro del transitorio,
    donde el error de truncación domina y el orden es observable.
    """
    if y0 is None:
        y0 = [0.0, 0.0, 0.0]
    if h_list is None:
        # Todos < 0.0033 (AB4 estable) y > 4e-4 (sin tocar el piso de máquina)
        h_list = np.logspace(np.log10(3e-3), np.log10(4e-4), 7)

    h_arr = np.array(h_list, dtype=float)

    # --- Referencia: RK4 muy fino. Error en el extremo, sin interpolar ---
    print(f"Calculando referencia (RK4, h={h_ref}, tf={tf})... un momento.")
    motor_ref = motor_class(**params)
    _, y_ref = integrar(motor_ref.derivadas, t0, tf, y0, h_ref, metodo='rk4')
    y_ref_end = y_ref[-1]   # valor exacto en t = tf

    etiquetas = {
        'rk1': 'RK1 — Euler',
        'rk2': 'RK2 — Heun',
        'rk3': 'RK3 — Kutta',
        'rk4': 'RK4 — Clásico',
        'ab4': 'AB4 — Adams-Bashforth',
        'pc4': 'PC4 — Predictor-Corrector',
    }

    plt.figure(figsize=(9, 7))
    errores_todos = {}

    print("\nPendientes medidas (≈ orden del método):")
    for metodo in metodos:
        errores = []
        for h in h_arr:
            motor = motor_class(**params)
            _, y_arr = integrar(motor.derivadas, t0, tf, y0, h, metodo=metodo)
            y_end = y_arr[-1]   # extremo exacto, SIN interpolación
            err = np.linalg.norm(y_end - y_ref_end) / np.linalg.norm(y_ref_end)
            errores.append(err)

        errores = np.array(errores)
        errores_todos[metodo] = errores

        pendiente = np.polyfit(np.log(h_arr), np.log(errores), 1)[0]
        print(f"  {metodo}: {pendiente:.2f}")
        plt.loglog(h_arr, errores, 'o-',
                   label=f"{etiquetas.get(metodo, metodo)}  (m={pendiente:.2f})")

    # --- Rectas guía de pendiente 1, 2, 3, 4 ---
    err_max = max(e.max() for e in errores_todos.values())
    anchor = err_max * 2.0
    
    for p, col in zip([1, 2, 3, 4], ['#cccccc', '#aaaaaa', '#888888', '#555555']):
        guia = anchor * (h_arr / h_arr[0])**p
        plt.loglog(h_arr, guia, '--', color=col, linewidth=1, label=f'guía pendiente {p}')
    
    plt.xlabel('Paso  h')
    plt.ylabel(f'Error relativo en t = {tf} s')
    plt.title('Estudio de convergencia (log-log)')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()