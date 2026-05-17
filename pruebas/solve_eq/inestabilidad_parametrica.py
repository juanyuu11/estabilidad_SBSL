"""
inestabilidad_parametrica.py
============================
Analiza la estabilidad de la superficie de la burbuja bajo perturbaciones
de armónicos esféricos Y_n^m resolviendo la ecuación de amplitud:

    ä_n + [3Ṙ/R] ȧ_n − [(n−1)R̈/R − (n−1)(n+1)(n+2)σ/(ρR³)] a_n = 0

Reescrita como sistema de primer orden  y = [a_n, ȧ_n]:

    ẏ₁ =  y₂
    ẏ₂ =  Q(t)·y₁ − P(t)·y₂

donde:
    P(t) = 3Ṙ/R
    Q(t) = (n−1)R̈/R − (n−1)(n+1)(n+2)σ/(ρR³)

Estrategia de precisión
-----------------------
R̈ no se interpola desde el array guardado (interpolación de A amplifica
errores en R³ cerca del colapso). En cambio, se re-evalúa exactamente desde
la ecuación K-M en cada llamada al ODE, usando el nucleo_fisico compilado
con Numba.

El coeficiente Q(t) escala como R⁻³ y puede alcanzar ~10¹⁵ m/s² cerca del
colapso. La regla de propagación de error impone:

    rtol_KM ≤ rtol_perturbacion / 3    →    rtol=1e-9, atol=1e-11

La dinámica radial se re-corre con dense_output=True para obtener una
representación continua de R(t) y Ṙ(t) sin error de interpolación de malla.

Jacobiano analítico para la ecuación de perturbación
------------------------------------------------------
El sistema es lineal ⟹ J(t) = [[0, 1], [Q(t), −P(t)]] exactamente.
Computar J no cuesta nada extra: P y Q ya se calcularon en el ODE.
A diferencia de K-M (2×2 no lineal donde Radau congela J muchos pasos),
aquí Q(t) varía órdenes de magnitud en un solo período → Radau actualiza J
con frecuencia → el Jacobiano analítico sí ahorra evaluaciones ODE.

Análisis de Floquet
-------------------
Se integra la matriz fundamental Φ(t) sobre un período T. Los multiplicadores
de Floquet μ son los autovalores de Φ(T). Como det Φ(T) = 1 (trayectoria
periódica + amortiguamiento geométrico promedio nulo), los multiplicadores
vienen en pares reciprocos: μ₁·μ₂ = 1.

    |μ_max| > 1  →  modo n INESTABLE (a_n crece exponencialmente)
    |μ_max| ≤ 1  →  modo n estable

Dependencias
------------
    radym_claude_conjac.py  (preparar_params, nucleo_fisico, warm_up)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

# Importar núcleo compilado del módulo de dinámica radial
from radym_claude_conjac import nucleo_fisico, preparar_params, warm_up


# =============================================================================
# 1. RE-EJECUCIÓN DE K-M CON DENSE OUTPUT Y TOLERANCIAS AJUSTADAS
#    La solución estacionaria se re-corre SOLO para los ciclos de análisis,
#    arrancando desde las condiciones iniciales del rec ya calculado.
# =============================================================================
def simular_km_denso(
    Pa, f=26500.0, R0=5.0e-6,
    ciclos_trans=50, ciclos_analisis=2,
    rtol=1e-9, atol=1e-11,
):
    """
    Re-corre la dinámica radial con dense_output=True y tolerancias estrictas.

    Devuelve
    --------
    sol      : OdeResult con sol.sol callable — R(t) = sol.sol(t)[0],
                                                V(t) = sol.sol(t)[1]
    params   : tupla de parámetros K-M (para re-evaluar R̈)
    t0, T    : inicio del intervalo estacionario y período acústico
    ok       : bool
    """
    rho, c, P0        = 1000.0, 1500.0, 101325.0
    sigma, eta, gamma = 0.0728, 0.001, 1.4
    h     = R0 / 8.5
    omega = 2.0 * np.pi * f

    params  = preparar_params(rho, c, Pa, omega, P0, sigma, eta, R0, h, gamma)
    periodo = 1.0 / f
    t0      = ciclos_trans * periodo
    t_end   = t0 + ciclos_analisis * periodo

    def ode(t, y):
        dr, dv = nucleo_fisico(t, y[0], y[1], *params)
        return [dr, dv]

    sol = solve_ivp(
        ode, [0.0, t_end], [R0, 0.0],
        method='Radau',
        rtol=rtol, atol=atol,
        first_step=1e-12,
        dense_output=True,    # representación polinómica continua
    )
    return sol, params, t0, periodo, sol.success


# =============================================================================
# 2. ODE Y JACOBIANO DE LA ECUACIÓN DE PERTURBACIÓN
#    Sistema 4-D: dos soluciones independientes integradas simultáneamente
#    para construir la matriz fundamental Φ(T) (análisis de Floquet).
# =============================================================================
def _construir_ode_perturbacion(sol_km, params, n, sigma, rho, R0):
    """
    Devuelve (ode_fn, jac_fn) para el modo esférico n.

    Estrategia de R̈
    ----------------
    Se llama a nucleo_fisico(t, R, V, *params) en lugar de interpolar A(t).
    Esto garantiza que R̈ sea siempre consistente con la ecuación K-M y evita
    la amplificación del error de interpolación en el término σ/R³.
    """
    h_km       = R0 / 8.5
    R_floor    = h_km + 1e-12
    coef_sigma = float((n - 1) * (n + 1) * (n + 2) * sigma / rho)
    coef_n1    = float(n - 1)

    # Callable continuo de la solución K-M (dense output de scipy)
    km_sol = sol_km.sol   # km_sol(t) → [R(t), V(t)]

    def _PQ(t):
        """Coeficientes P(t) y Q(t) en el instante t."""
        rv  = km_sol(t)
        R   = max(float(rv[0]), R_floor)
        V   = float(rv[1])
        # R̈ recalculado exactamente desde K-M
        _, A = nucleo_fisico(t, R, V, *params)
        R_inv = 1.0 / R
        P = 3.0 * V * R_inv
        Q = coef_n1 * A * R_inv - coef_sigma * R_inv * R_inv * R_inv
        return P, Q

    def ode_4d(t, y):
        """
        y = [a¹, ȧ¹, a², ȧ²]
        Dos soluciones independientes (condiciones iniciales [1,0] y [0,1]).
        """
        P, Q = _PQ(t)
        # ä = Q·a − P·ȧ
        return [y[1], Q * y[0] - P * y[1],
                y[3], Q * y[2] - P * y[3]]

    def jac_4d(t, y):
        """
        Jacobiano analítico exacto ∂f/∂y.
        Para el sistema lineal ẏ = A(t)·y, J = diag(A, A) (estructura bloque).
        Coste adicional: cero — P y Q ya se calcularon en ode_4d.
        """
        P, Q = _PQ(t)
        # Cada bloque diagonal 2×2 es [[0, 1], [Q, -P]]
        return np.array([
            [ 0.,  1.,  0.,  0.],
            [ Q,  -P,   0.,  0.],
            [ 0.,  0.,  0.,  1.],
            [ 0.,  0.,   Q,  -P],
        ])

    return ode_4d, jac_4d


# =============================================================================
# 3. ANÁLISIS DE FLOQUET PARA UN MODO n
# =============================================================================
def analizar_modo(
    sol_km, params, t0, T, n,
    sigma=0.0728, rho=1000.0, R0=5.0e-6,
    rtol=1e-9, atol=1e-11,
):
    """
    Integra la ecuación de perturbación para el modo n sobre un período T
    y devuelve los multiplicadores de Floquet.

    Parámetros
    ----------
    sol_km  : OdeResult con dense_output=True de simular_km_denso()
    params  : tupla de preparar_params()
    t0, T   : inicio del intervalo estacionario y período
    n       : modo esférico (entero ≥ 2)
    rtol/atol: tolerancias para la ODE de perturbación

    Devuelve
    --------
    dict con:
      'n'               : modo
      'mu'              : array (2,) — multiplicadores de Floquet
      'mu_max'          : max(|μ|)
      'estable'         : bool
      'tasa_crecimiento': Re(ln μ_max) / T  [s⁻¹]
      'det_C'           : det(Φ(T)) — debe ser ≈ 1.0 (verificación)
      'exito'           : bool
    """
    ode_fn, jac_fn = _construir_ode_perturbacion(sol_km, params, n, sigma, rho, R0)

    # Condiciones iniciales: matriz identidad 2×2 aplanada
    y0 = [1.0, 0.0,   # solución 1: a(t0)=1, ȧ(t0)=0
          0.0, 1.0]   # solución 2: a(t0)=0, ȧ(t0)=1

    sol = solve_ivp(
        ode_fn,
        [t0, t0 + T],
        y0,
        method='Radau',
        jac=jac_fn,          # Jacobiano analítico (coste cero; ayuda mucho en el colapso)
        rtol=rtol,
        atol=atol,
        first_step=1e-13,    # arranque conservador — Q puede ser ~10¹⁵ en t=t_colapso
        dense_output=False,
    )

    if not sol.success:
        return {'n': n, 'exito': False, 'mensaje': sol.message}

    yT = sol.y[:, -1]

    # Matriz de monodromía Φ(T)  —  columnas = soluciones en t = t0 + T
    C = np.array([[yT[0], yT[2]],
                  [yT[1], yT[3]]])

    mu     = np.linalg.eigvals(C)
    mu_max = float(np.max(np.abs(mu)))
    det_C  = float(np.linalg.det(C))
    tasa   = float(np.log(mu_max) / T)

    return {
        'n'               : n,
        'mu'              : mu,
        'mu_max'          : mu_max,
        'estable'         : mu_max <= 1.0 + 1e-6,   # margen numérico
        'tasa_crecimiento': tasa,
        'det_C'           : det_C,    # ≈ 1.0 si la trayectoria es periódica
        'exito'           : True,
    }


# =============================================================================
# 4. ANÁLISIS COMPLETO — PIPELINE PRINCIPAL
# =============================================================================
def analizar_estabilidad(
    Pa,
    f         = 26500.0,
    R0        = 5.0e-6,
    n_modes   = (2, 3, 4, 5, 6),
    sigma     = 0.0728,
    rho       = 1000.0,
    ciclos_trans   = 50,
    ciclos_analisis = 2,
    rtol_km   = 1e-9,
    atol_km   = 1e-11,
    rtol_pert = 1e-9,
    atol_pert = 1e-11,
    verbose   = True,
):
    """
    Pipeline completo: dinámica radial → análisis de Floquet por modo.

    Parámetros
    ----------
    Pa            : amplitud de presión acústica [Pa]
    f             : frecuencia [Hz]
    R0            : radio de equilibrio [m]
    n_modes       : modos esféricos a analizar (enteros ≥ 2)
    sigma, rho    : propiedades del líquido
    ciclos_trans  : ciclos de transitorio a descartar
    ciclos_analisis: ciclos estacionarios con dense output
    rtol_km/atol_km  : tolerancias para K-M (regla: ≤ rtol_pert / 3)
    rtol_pert/atol_pert: tolerancias para la ODE de perturbación

    Devuelve
    --------
    dict {n: resultado_floquet}  —  ver analizar_modo() para estructura
    """
    # --- 1. Dinámica radial con dense output ---
    sol_km, params, t0, T, ok_km = simular_km_denso(
        Pa, f, R0, ciclos_trans, ciclos_analisis, rtol_km, atol_km
    )

    if not ok_km:
        raise RuntimeError(f"La simulación K-M no convergió para Pa={Pa:.2e} Pa")

    if verbose:
        print(f"\nPa = {Pa:.3e} Pa   f = {f/1e3:.2f} kHz   R0 = {R0*1e6:.1f} µm")
        print(f"{'Modo':<6} {'|μ_max|':>10} {'γ [s⁻¹]':>14} {'det(Φ)':>10} {'estado':>12}")
        print("-" * 56)

    # --- 2. Floquet por modo ---
    resultados = {}
    for n in n_modes:
        res = analizar_modo(
            sol_km, params, t0, T, n,
            sigma=sigma, rho=rho, R0=R0,
            rtol=rtol_pert, atol=atol_pert,
        )
        resultados[n] = res

        if verbose:
            if res['exito']:
                estado = "INESTABLE ⚠" if not res['estable'] else "estable ✓"
                print(
                    f"  n={n:<3d} {res['mu_max']:>10.6f} "
                    f"{res['tasa_crecimiento']:>+14.4e} "
                    f"{res['det_C']:>10.6f}  {estado}"
                )
            else:
                print(f"  n={n:<3d} FALLO: {res.get('mensaje','?')}")

    return resultados


# =============================================================================
# 5. BARRIDO PARAMÉTRICO — múltiples Pa con análisis de Floquet
# =============================================================================
def barrido_estabilidad(
    Pa_arr,
    n_modes = (2, 3, 4, 5, 6),
    **kwargs,
):
    """
    Barrido de Pa_arr con análisis de Floquet para cada modo.

    Devuelve
    --------
    mapa : dict {Pa: {n: resultado_floquet}}

    Ejemplo de uso
    --------------
    Pa_vals = np.linspace(0.5e5, 2.0e5, 40)
    mapa = barrido_estabilidad(Pa_vals, n_modes=[2, 3, 4])

    # Extraer mapa de estabilidad
    for Pa, res_modos in mapa.items():
        for n, res in res_modos.items():
            if res['exito'] and not res['estable']:
                print(f"Pa={Pa:.2e}  n={n}  |μ|={res['mu_max']:.4f}  INESTABLE")
    """
    mapa = {}
    for i, Pa in enumerate(Pa_arr):
        print(f"\n[{i+1}/{len(Pa_arr)}] Pa = {Pa:.3e} Pa")
        try:
            mapa[Pa] = analizar_estabilidad(Pa, n_modes=n_modes, verbose=True, **kwargs)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            mapa[Pa] = {n: {'exito': False, 'mensaje': str(e)} for n in n_modes}
    return mapa


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    warm_up()

    # ---- Análisis para un único Pa ----
    Pa = 1.2e5
    t0_ = time.perf_counter()
    resultados = analizar_estabilidad(
        Pa,
        f             = 26500.0,
        R0            = 5.0e-6,
        n_modes       = [2, 3, 4, 5, 6, 7, 8],
        ciclos_trans  = 50,
        ciclos_analisis = 1,
        rtol_km       = 1e-9,
        atol_km       = 1e-11,
        rtol_pert     = 1e-9,
        atol_pert     = 1e-11,
        verbose       = True,
    )
    print(f"\nAnálisis completado en {time.perf_counter() - t0_:.2f} s")

    # ---- Gráfica: |μ_max| por modo ----
    modos   = sorted(n for n, r in resultados.items() if r['exito'])
    mu_vals = [resultados[n]['mu_max'] for n in modos]
    colores = ['#D6604D' if resultados[n]['mu_max'] > 1.0 else '#2166AC' for n in modos]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(modos, mu_vals, color=colores, edgecolor='white', width=0.6)
    ax.axhline(1.0, color='black', lw=1.2, ls='--', label='umbral |μ| = 1')
    ax.set_xlabel("Modo esférico n", fontsize=11)
    ax.set_ylabel("|μ_max|  (multiplicador de Floquet)", fontsize=11)
    ax.set_title(f"Estabilidad de la superficie — Pa = {Pa:.1e} Pa", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("estabilidad_modos.png", dpi=150, bbox_inches='tight')
    print("Figura guardada en estabilidad_modos.png")
    plt.show()