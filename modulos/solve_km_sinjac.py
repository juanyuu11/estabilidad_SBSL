import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit, prange
from concurrent.futures import ProcessPoolExecutor
import os
import matplotlib
matplotlib.use("TkAgg")  # Motor gráfico interactivo para que salte la ventana
import matplotlib.pyplot as plt
import time

# =============================================================================
# 1. PRECOMPUTACIÓN DE CONSTANTES DERIVADAS
# Evita recalcular h³, R₀³, Pg₀, etc. en cada una de las millones de
# evaluaciones del ODE. Se llama una sola vez antes de iniciar solve_ivp.
# =============================================================================
def preparar_params(rho, c, Pa, omega, P0, sigma, eta, R0, h, gamma):
    h3      = h ** 3
    R03_mh3 = R0 ** 3 - h3           # R₀³ - h³  (constante del equilibrio)
    Pg0     = P0 + 2.0 * sigma / R0  # Presión de referencia del gas
    return (
        rho,
        1.0 / c,       # c_inv  → evita divisiones repetidas
        Pa, omega, P0,
        2.0 * sigma,   # two_sig
        4.0 * eta,     # four_eta
        gamma,
        R0, h, h3, R03_mh3, Pg0,
        3.0 * gamma,   # three_gam
        1.5 * rho,     # half_rho  (coeficiente cinético)
        1.0 / rho,     # rho_inv
    )


# =============================================================================
# 2. NÚCLEO FÍSICO — Keller-Miksis
# cache=True  → compila una vez y reutiliza el binario entre sesiones.
# fastmath=True → habilita SIMD y reordenación de operaciones float.
# =============================================================================
@njit(cache=True, fastmath=True)
def nucleo_fisico(t, R, V,
                  rho, c_inv, Pa, omega, P0, two_sig, four_eta, gamma,
                  R0, h, h3, R03_mh3, Pg0, three_gam, half_rho, rho_inv):
    """dR/dt = V  y  dV/dt = A  — ecuación de Keller-Miksis."""
    R_safe = R if R > h + 1e-12 else h + 1e-12
    R2     = R_safe * R_safe
    R3     = R2    * R_safe
    denom  = R3 - h3
    R_inv  = 1.0 / R_safe

    # Presión del gas (estado adiabático)
    Pg     = Pg0 * ((R03_mh3 / denom) ** gamma)

    # Término de radiación acústica: (R/c) · dPg/dt
    dPg_dt = -(three_gam * R2 * Pg * V) / denom
    F_rad  = R_safe * c_inv * dPg_dt

    # Suma de fuerzas por unidad de área
    F = (Pg - P0
         + Pa * np.sin(omega * t)
         - four_eta * V  * R_inv
         - two_sig       * R_inv)

    # Aceleración  dV/dt
    A = (F + F_rad - half_rho * V * V) * rho_inv * R_inv
    return V, A


# =============================================================================
# 3. POST-PROCESO PARALELO con prange
# Numba distribuye los N puntos entre todos los núcleos disponibles.
# =============================================================================
@njit(cache=True, parallel=True)
def _calcular_aceleraciones(t_arr, R_arr, V_arr,
                             rho, c_inv, Pa, omega, P0, two_sig, four_eta,
                             gamma, R0, h, h3, R03_mh3, Pg0,
                             three_gam, half_rho, rho_inv):
    N     = len(t_arr)
    A_arr = np.empty(N)
    for i in prange(N):
        _, a = nucleo_fisico(
            t_arr[i], R_arr[i], V_arr[i],
            rho, c_inv, Pa, omega, P0, two_sig, four_eta, gamma,
            R0, h, h3, R03_mh3, Pg0, three_gam, half_rho, rho_inv,
        )
        A_arr[i] = a
    return A_arr


bubble_dtype = np.dtype([('t', 'f8'), ('R', 'f8'), ('V', 'f8'), ('A', 'f8')])

def _post_procesar(sol, params):
    t_arr, R_arr, V_arr = sol.t, sol.y[0], sol.y[1]
    A_arr  = _calcular_aceleraciones(t_arr, R_arr, V_arr, *params)
    result = np.empty(len(t_arr), dtype=bubble_dtype)
    result['t'] = t_arr
    result['R'] = R_arr
    result['V'] = V_arr
    result['A'] = A_arr
    return result


# =============================================================================
# 4. FUNCIÓN PRINCIPAL
# =============================================================================
def simular_burbuja_estacionaria(
    Pa, R0, f=25000.0,
    ciclos_trans=10, ciclos_save=2,
    rtol=1e-9, atol=1e-11,
):
    """
    Simula la burbuja y retorna un Record Array con (t, R, V, A).
    """
    # Constantes físicas (Agua a 20 °C)
    rho, c, P0        = 1000.0, 1500.0, 101325.0
    sigma, eta, gamma = 0.0728, 0.001, 1.4
    h     = R0 / 8.5
    omega = 2.0 * np.pi * f

    params = preparar_params(rho, c, Pa, omega, P0, sigma, eta, R0, h, gamma)

    periodo = 1.0 / f
    t_end   = (ciclos_trans + ciclos_save) * periodo
    t_eval  = np.linspace(ciclos_trans * periodo, t_end, int(ciclos_save * 1000))

    def ode(t, y):
        dr, dv = nucleo_fisico(t, y[0], y[1], *params)
        return [dr, dv]

    sol = solve_ivp(
        ode, [0.0, t_end], [R0, 0.0],
        method='Radau',
        t_eval=t_eval,
        rtol=rtol, atol=atol,
        first_step=1e-11,
        dense_output=False,
    )

    if sol.success:
        return _post_procesar(sol, params), True
    return None, False


# =============================================================================
# 5. SIMULACIÓN EN LOTE — ProcessPoolExecutor
# Ideal para barridos de Pa (diagramas de bifurcación, mapas de fase).
# =============================================================================
def simular_batch(Pa_arr, n_workers=None, **kwargs):
    """
    Simula múltiples amplitudes de presión en paralelo.

    Ejemplo:
        Pa_vals = np.linspace(0.5e5, 1.5e5, 64)
        resultados = simular_batch(Pa_vals)
        for Pa, (rec, ok) in resultados.items():
            if ok:
                print(f"Pa={Pa:.0f} Pa  R_max={rec['R'].max()*1e6:.2f} µm")
    """
    n = n_workers or os.cpu_count()
    with ProcessPoolExecutor(max_workers=n) as ex:
        futures = {Pa: ex.submit(simular_burbuja_estacionaria, Pa, **kwargs)
                   for Pa in Pa_arr}
    return {Pa: fut.result() for Pa, fut in futures.items()}


# =============================================================================
# CALENTAMIENTO DEL JIT
# Pre-compila todas las funciones Numba antes de la simulación real.
# =============================================================================
def warm_up():
    _params = preparar_params(1000., 1500., 1e5, 2*np.pi*26500,
                              101325., 0.0728, 0.001, 5e-6, 5e-6/8.5, 1.4)
    nucleo_fisico(0.0, 5e-6, 0.0, *_params)
    t = np.array([0.0]); R = np.array([5e-6]); V = np.array([0.0])
    _calcular_aceleraciones(t, R, V, *_params)
    print("JIT warm-up completo.")


# =============================================================================
# ENTRADA
# =============================================================================
if __name__ == "__main__":
    warm_up()

    Pa = 1.2e5  # Pa
    R0 = 5.0e-6  # m
    inicio = time.perf_counter()
    rec, ok = simular_burbuja_estacionaria(Pa,R0)
    fin = time.perf_counter()
    print(f"Simulación completa en {fin - inicio:.2f} segundos.")
    plt.figure(figsize=(10, 6))
    plt.plot(rec['t']*1e6, rec['R']*1e6)
    plt.xlabel("Tiempo (µs)")
    plt.ylabel("Radio (µm)")
    plt.title(f"Simulación de burbuja con Pa={Pa:.1e} Pa")
    plt.grid()
    plt.show()
    if ok:
        print(f"Puntos guardados : {len(rec)}")
        print(f"R_max            : {rec['R'].max()*1e6:.3f}  µm")
        print(f"R_min            : {rec['R'].min()*1e6:.4f} µm")
        print(f"|A|_max          : {np.abs(rec['A']).max():.3e} m/s²")