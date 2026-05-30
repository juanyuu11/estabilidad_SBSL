"""
criterio_estabilidad_difusiva.py
=================================
Criterio de estabilidad difusiva para burbujas SBSL.
Brenner, Hilgenfeldt & Lohse — Rev. Mod. Phys. 74, 425 (2002) — Sección IV.B

USO BÁSICO
----------
    from criterio_estabilidad_difusiva import warm_up, estabilidad_difusiva

    warm_up()   # SIEMPRE PRIMERO

    condicion = estabilidad_difusiva(
        Pa       = 1.3e5,   # [Pa]  presión acústica
        R0       = 5.0e-6,  # [m]   radio ambiente
        c_inf_c0 = 0.002,   # [-]   C∞/C0
    )
    # 'ESTABLE' | 'INESTABLE' | 'IMPLOSION' | 'DISOLUCION' | 'ERROR'

MAPA DE ESTABILIDAD
-------------------
    import numpy as np
    from criterio_estabilidad_difusiva import warm_up, estabilidad_difusiva

    warm_up()

    Pa_vals = np.linspace(1.0e5, 1.4e5, 20)
    R0_vals = np.linspace(1e-6, 10e-6, 20)

    mapa = {
        (Pa, R0): estabilidad_difusiva(Pa, R0, c_inf_c0=0.002)
        for Pa in Pa_vals
        for R0 in R0_vals
    }

VERBOSIDAD
----------
    import logging
    logging.basicConfig(level=logging.INFO)   # activa mensajes internos

MODOS NUMÉRICOS
---------------
    'exploracion'     — rápido, baja precisión
    'analisis_rapido' — ~90 s por punto, precisión suficiente para mapas  ← defecto
    'analisis'        — ~10 min por punto, alta precisión
    'publicacion'     — máxima precisión

REFERENCIAS
-----------
[1] Brenner, Hilgenfeldt & Lohse, Rev. Mod. Phys. 74, 425 (2002)
[2] Fyrillas & Szeri, J. Fluid Mech. 277, 381 (1994)
[3] Löfstedt et al., Phys. Rev. E 51, 4400 (1995)
[4] Keller & Miksis, J. Acoust. Soc. Am. 68, 628 (1980)
[5] Virtanen et al., Nature Methods 17, 261 (2020)  [SciPy]
[6] Fritsch & Carlson, SIAM J. Numer. Anal. 17, 238 (1980)  [PCHIP]
[7] Brent, Algorithms for Minimization without Derivatives (1973)
"""

import os
import time
import logging
import warnings
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator   # [6] Fritsch & Carlson 1980
from scipy.optimize import brentq                  # [7] Brent 1973
from numba import njit, prange
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

logger = logging.getLogger(__name__)

# tqdm — barra de progreso (instalar con: pip install tqdm)
try:
    from tqdm.auto import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False
    warnings.warn(
        "tqdm no está instalado. La barra de progreso no estará disponible.\n"
        "Instalar con: pip install tqdm"
    )

# =============================================================================
# CONSTANTES FÍSICAS — Agua a 20 °C, Argón
# =============================================================================
RHO   = 998.0       # kg/m³  densidad del agua
C_SND = 1481.0      # m/s    velocidad del sonido en agua
P0    = 101325.0    # Pa     presión ambiente (1 atm)
SIGMA = 0.0725      # N/m    tensión superficial agua-gas
ETA   = 1.002e-3    # Pa·s   viscosidad dinámica agua
GAMMA = 5/3         # —      índice adiabático ARGÓN (monoatómico)
                    #         ⚠️  usar 7/5 para aire
P_VAP = 2337.0      # Pa     presión de vapor agua a 20°C
F_DEF = 25000.0     # Hz     frecuencia de forzamiento por defecto

# =============================================================================
# CONFIGURACIONES NUMÉRICAS
# rtol es siempre el parámetro determinante (análisis de convergencia)
# =============================================================================
CONFIGS = {
    'exploracion'    : dict(ciclos_trans=5,  ciclos_save=1, rtol=1e-6, atol=1e-8),
    'analisis_rapido': dict(ciclos_trans=10, ciclos_save=1, rtol=1e-7, atol=1e-9),
    'analisis'       : dict(ciclos_trans=10, ciclos_save=2, rtol=1e-8, atol=1e-10),
    'publicacion'    : dict(ciclos_trans=10, ciclos_save=2, rtol=1e-9, atol=1e-11),
}
# Número de puntos de barrido por defecto según modo
_N_DEFAULT = {
    'exploracion'    : 40,
    'analisis_rapido': 25,
    'analisis'       : 60,
    'publicacion'    : 80,
}

bubble_dtype = np.dtype([('t','f8'), ('R','f8'), ('V','f8'), ('A','f8')])


# =============================================================================
# BLOQUE 1 — INTEGRADOR KELLER-MIKSIS
# Keller & Miksis, J. Acoust. Soc. Am. 68, 628 (1980) [4]
# =============================================================================

def _preparar_params(Pa: float, omega: float, R0: float) -> tuple:
    """
    Precomputa h³, R₀³-h³, Pg₀ una sola vez antes de cada integración.
    Evita recalcularlos en cada una de las ~100 000 evaluaciones del ODE.
    """
    h      = R0 / 8.5
    h3     = h**3
    R03mh3 = R0**3 - h3
    Pg0    = P0 + 2.0*SIGMA/R0
    return (
        RHO, 1.0/C_SND, Pa, omega, P0,
        2.0*SIGMA, 4.0*ETA, GAMMA,
        R0, h, h3, R03mh3, Pg0,
        3.0*GAMMA, 1.5*RHO, 1.0/RHO, P_VAP,
    )


@njit(cache=True, fastmath=True)
def _nucleo_km(t, R, V,
               rho, c_inv, Pa, omega, P0, two_sig, four_eta, gamma,
               R0, h, h3, R03mh3, Pg0, three_gam, half_rho, rho_inv, p_vav):
    """
    Núcleo Keller-Miksis: retorna (dR/dt, dV/dt).
    Numba JIT — compilado a código máquina.
    Incluye van der Waals, radiación acústica, viscosidad y tensión superficial.
    """
    R_s   = R if R > h + 1e-12 else h + 1e-12
    R2    = R_s*R_s
    R3    = R2*R_s
    denom = R3 - h3
    R_inv = 1.0/R_s
    Pg     = Pg0*(R03mh3/denom)**gamma + p_vav
    dPg_dt = -(three_gam*R2*(Pg - p_vav)*V)/denom
    F_rad  = R_s*c_inv*dPg_dt
    Pa_t   = -Pa*np.sin(omega*t)
    F = (Pg - P0 - p_vav + Pa_t
         - four_eta*V*R_inv
         - two_sig*R_inv)
    A = ((1.0 + V*c_inv)*F + F_rad - half_rho*V*V)*rho_inv*R_inv
    return V, A


@njit(cache=True, parallel=True)
def _aceleraciones_paralelo(t_arr, R_arr, V_arr,
                             rho, c_inv, Pa, omega, P0, two_sig, four_eta,
                             gamma, R0, h, h3, R03mh3, Pg0,
                             three_gam, half_rho, rho_inv, p_vav):
    """
    Post-proceso paralelo con prange.
    Numba distribuye los N puntos entre todos los núcleos disponibles.
    """
    N     = len(t_arr)
    A_arr = np.empty(N)
    for i in prange(N):
        _, a = _nucleo_km(
            t_arr[i], R_arr[i], V_arr[i],
            rho, c_inv, Pa, omega, P0, two_sig, four_eta, gamma,
            R0, h, h3, R03mh3, Pg0, three_gam, half_rho, rho_inv, p_vav,
        )
        A_arr[i] = a
    return A_arr


def _post_procesar(sol, params: tuple) -> np.ndarray:
    t_arr, R_arr, V_arr = sol.t, sol.y[0], sol.y[1]
    A_arr  = _aceleraciones_paralelo(t_arr, R_arr, V_arr, *params)
    result = np.empty(len(t_arr), dtype=bubble_dtype)
    result['t'] = t_arr
    result['R'] = R_arr
    result['V'] = V_arr
    result['A'] = A_arr
    return result


def simular_burbuja_estacionaria(
    Pa: float,
    R0: float,
    f:            float = F_DEF,
    ciclos_trans: int   = 10,
    ciclos_save:  int   = 1,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> tuple:
    """
    Integra Keller-Miksis y retorna el tramo estacionario.

    Retorna
    -------
    (rec, True)   si converge — rec tiene campos ('t','R','V','A')
    (None, False) si falla
    """
    omega  = 2.0*np.pi*f
    params = _preparar_params(Pa, omega, R0)
    Td     = 1.0/f
    t_end  = (ciclos_trans + ciclos_save)*Td
    t_eval = np.linspace(ciclos_trans*Td, t_end, ciclos_save*1000)

    def ode(t, y):
        dr, dv = _nucleo_km(t, y[0], y[1], *params)
        return [dr, dv]

    try:
        sol = solve_ivp(
            ode,
            t_span      = [0.0, t_end],
            y0          = [R0, 0.0],
            method      = 'Radau',
            t_eval      = t_eval,
            rtol        = rtol,
            atol        = atol,
            first_step  = 1e-11,
            dense_output= False,
        )
    except Exception as e:
        logger.warning('ODE falló (Pa=%.2f atm, R0=%.1f µm): %s',
                       Pa/P0, R0*1e6, e)
        return None, False

    if not sol.success or np.any(sol.y[0] <= 0):
        return None, False

    return _post_procesar(sol, params), True


# =============================================================================
# BLOQUE 2 — EJECUCIÓN EN LOTE CON BARRA DE PROGRESO
# ProcessPoolExecutor + tqdm.auto
# =============================================================================

def _simular_batch(
    casos: list,
    n_workers: Optional[int],
    desc: str,
    **kwargs,
) -> dict:
    """
    Simula múltiples casos (Pa, R0) en paralelo con barra de progreso tqdm.
    Retorna dict { (Pa, R0): (rec, ok) }.
    """
    n          = n_workers if n_workers is not None else os.cpu_count()
    resultados = {}

    logger.info('Barrido: %d simulaciones en %d núcleos (de %d disponibles)',
                len(casos), n, os.cpu_count())

    with ProcessPoolExecutor(max_workers=n) as ex:
        futures = {
            ex.submit(simular_burbuja_estacionaria, Pa, R0, **kwargs): (Pa, R0)
            for Pa, R0 in casos
        }

        pbar = (tqdm(total=len(futures), desc=desc, unit='sim',
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                                '[{elapsed}<{remaining}, {rate_fmt}]')
                if _TQDM else None)

        try:
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    resultados[key] = fut.result()
                except Exception as e:
                    resultados[key] = (None, False)
                    logger.warning('Error en caso %s: %s', key, e)

                if pbar is not None:
                    n_ok   = sum(1 for _, ok in resultados.values() if ok)
                    n_fail = len(resultados) - n_ok
                    pbar.set_postfix(ok=n_ok, fail=n_fail)
                    pbar.update(1)
        finally:
            if pbar is not None:
                pbar.close()

    return resultados


# =============================================================================
# BLOQUE 3 — CRITERIO DIFUSIVO
# Fyrillas & Szeri, J. Fluid Mech. 277, 381 (1994) [2]
# Löfstedt et al., Phys. Rev. E 51, 4400 (1995)    [3]
# =============================================================================

def _calcular_pg4(rec: np.ndarray, R0: float) -> float:
    """
    ⟨Pg⟩₄ = ∫ Pg(t)·R⁴(t) dt / ∫ R⁴(t) dt
    Fyrillas & Szeri (1994) [2], Ec. 52.
    Regla del trapecio vectorizada — O(N) sin bucles Python.
    """
    h      = R0/8.5
    h3     = h**3
    Pg0    = P0 + 2.0*SIGMA/R0
    R03mh3 = R0**3 - h3
    R  = rec['R']
    t  = rec['t']
    denom = np.where(R**3 - h3 > 1e-30, R**3 - h3, 1e-30)
    Pg = Pg0*(R03mh3/denom)**GAMMA
    R4 = R**4
    num = np.trapz(Pg*R4, t)
    den = np.trapz(R4, t)
    if den == 0.0:
        raise ValueError('Denominador nulo en ⟨Pg⟩₄: revisar R(t).')
    return num/den


def _pg4_ratio(rec: np.ndarray, R0: float) -> float:
    """⟨Pg⟩₄ / P0 (adimensional)."""
    return _calcular_pg4(rec, R0)/P0


def _aprox_lofstedt(rec: np.ndarray, R0: float) -> float:
    """
    Aproximación Löfstedt et al. (1995) [3], Ec. 55:
        Pg(Rmax)/P0 ≈ (R0/Rmax)³
    Verificación cruzada del promedio exacto.
    """
    return (R0/rec['R'].max())**3


def _detectar_equilibrios(
    R0_arr:   np.ndarray,
    pg4_arr:  np.ndarray,
    c_inf_c0: float,
) -> list:
    """
    Detecta cruces de ⟨Pg⟩₄/P0 con c_inf_c0 y evalúa estabilidad.

    Método (totalmente citable):
    - Interpolación PCHIP — Fritsch & Carlson (1980) [6]
    - Localización exacta del cruce — brentq, Brent (1973) [7]
    - Estabilidad: signo de d⟨Pg⟩₄/dR0 — Brenner et al. (2002) [1], Ec. 53

    Retorna lista de dicts {R0_eq, pg4_eq, dpg4_dR0, estable}
    """
    mask = ~np.isnan(pg4_arr)
    R0_v = R0_arr[mask]
    pg_v = pg4_arr[mask]
    if len(R0_v) < 4:
        return []

    interp  = PchipInterpolator(R0_v, pg_v)
    dinterp = interp.derivative()
    res_fn  = lambda R: float(interp(R)) - c_inf_c0

    equilibrios = []
    for i in range(len(R0_v) - 1):
        fa = res_fn(R0_v[i])
        fb = res_fn(R0_v[i+1])
        if fa*fb >= 0:
            continue
        try:
            R0_eq = brentq(res_fn, R0_v[i], R0_v[i+1], xtol=1e-11)
            dpg4  = float(dinterp(R0_eq))
            equilibrios.append({
                'R0_eq'    : R0_eq,
                'pg4_eq'   : float(interp(R0_eq)) + c_inf_c0,
                'dpg4_dR0' : dpg4,
                'estable'  : dpg4 > 0,
            })
        except Exception:
            pass

    return equilibrios


def _barrido_R0(
    Pa:        float,
    f:         float,
    cfg:       dict,
    R0_min:    float,
    R0_max:    float,
    N:         int,
    n_workers: Optional[int],
    desc:      str = 'Barrido R0',
) -> tuple:
    """
    Barre R0 en N puntos usando _simular_batch (multinúcleo + tqdm).
    Retorna (R0_array, pg4_array).
    """
    R0_arr = np.linspace(R0_min, R0_max, N)
    casos  = [(Pa, R0) for R0 in R0_arr]

    resultados = _simular_batch(
        casos,
        n_workers = n_workers,
        desc      = desc,
        f         = f,
        **cfg,
    )

    pg4_arr = np.full(N, np.nan)
    n_ok    = 0
    n_fail  = 0

    for j, R0 in enumerate(R0_arr):
        rec, ok = resultados.get((Pa, R0), (None, False))
        if ok and rec is not None:
            ratio = _pg4_ratio(rec, R0)
            pg4_arr[j] = ratio
            n_ok += 1
            # Verificación cruzada Löfstedt [3]
            aprox = _aprox_lofstedt(rec, R0)
            if ratio > 0.01 and abs(ratio - aprox)/(ratio + 1e-30) > 0.5:
                logger.debug('Löfstedt difiere >50%% en R0=%.1f µm '
                             '(esperado para R0 pequeño)', R0*1e6)
        else:
            n_fail += 1

    logger.info('Barrido completo: %d OK, %d fallidas de %d puntos',
                n_ok, n_fail, N)

    return R0_arr, pg4_arr


# =============================================================================
# BLOQUE 4 — FUNCIÓN PÚBLICA
# =============================================================================

def estabilidad_difusiva(
    Pa:        float,
    R0:        float,
    c_inf_c0:  float,
    f:         float        = F_DEF,
    modo:      str          = 'analisis_rapido',
    R0_min:    float        = 1.0e-6,
    R0_max:    float        = 10.0e-6,
    N:         Optional[int] = None,
    n_workers: Optional[int] = None,
) -> str:
    """
    Criterio de estabilidad difusiva — Sección IV.B
    Brenner, Hilgenfeldt & Lohse (2002) [1]

    Determina el régimen difusivo de una burbuja en (Pa, R0).

    Parámetros
    ----------
    Pa        : presión acústica [Pa]            (ej: 1.3e5)
    R0        : radio ambiente [m]               (ej: 5e-6)
    c_inf_c0  : C∞/C0 [-]                        (ej: 0.002)
    f         : frecuencia [Hz]                   (default: 25 kHz)
    modo      : precisión numérica
                  'exploracion'     — rápido, baja precisión
                  'analisis_rapido' — ~90 s, suficiente para mapas  ← defecto
                  'analisis'        — ~10 min, alta precisión
                  'publicacion'     — máxima precisión
    R0_min    : límite inferior del barrido [m]   (default: 1 µm)
    R0_max    : límite superior del barrido [m]   (default: 10 µm)
    N         : puntos del barrido interno
                  None → según modo: exploracion=40, analisis_rapido=25,
                                     analisis=60, publicacion=80
    n_workers : núcleos para paralelización
                  None → todos los disponibles (os.cpu_count())
                  int  → número exacto de núcleos a usar

    Retorna
    -------
    str — uno de:
        'ESTABLE'    burbuja en equilibrio difusivo estable
        'INESTABLE'  burbuja en equilibrio difusivo inestable
        'IMPLOSION'  sin equilibrio: la burbuja crece indefinidamente
        'DISOLUCION' sin equilibrio: la burbuja se disuelve
        'ERROR'      la simulación ODE no convergió en (Pa, R0)
    """
    cfg        = CONFIGS.get(modo, CONFIGS['analisis_rapido'])
    N_efectivo = N if N is not None else _N_DEFAULT.get(modo, 25)

    # ── PASO 1: Evaluación directa — diagnóstico rápido sin barrido ──────────
    rec_directo, ok_directo = simular_burbuja_estacionaria(Pa, R0, f=f, **cfg)
    if not ok_directo or rec_directo is None:
        logger.warning('ODE no convergió en Pa=%.2f atm, R0=%.1f µm',
                       Pa/P0, R0*1e6)
        return 'ERROR'

    pg4_en_R0  = _pg4_ratio(rec_directo, R0)
    residuo_rel = (pg4_en_R0 - c_inf_c0) / (c_inf_c0 + 1e-30)

    # ── PASO 2: Diagnóstico rápido — evita el barrido si el caso es trivial ──
    if residuo_rel > 5.0:
        return 'DISOLUCION'

    if residuo_rel < -0.9:
        return 'IMPLOSION'

    # ── PASO 3: Barrido multinúcleo ───────────────────────────────────────────
    R0_arr, pg4_arr = _barrido_R0(
        Pa        = Pa,
        f         = f,
        cfg       = cfg,
        R0_min    = R0_min,
        R0_max    = R0_max,
        N         = N_efectivo,
        n_workers = n_workers,
        desc      = f'Pa={Pa/P0:.2f} atm',
    )

    # ── PASO 4: Detectar equilibrios ─────────────────────────────────────────
    equilibrios = _detectar_equilibrios(R0_arr, pg4_arr, c_inf_c0)

    if not equilibrios:
        residuo_medio = float(np.nanmean(pg4_arr) - c_inf_c0)
        return 'DISOLUCION' if residuo_medio > 0 else 'IMPLOSION'

    # ── PASO 5: Equilibrio más cercano al R0 de entrada ──────────────────────
    distancias = [abs(eq['R0_eq'] - R0) for eq in equilibrios]
    eq_cercano = equilibrios[int(np.argmin(distancias))]

    return 'ESTABLE' if eq_cercano['estable'] else 'INESTABLE'


# =============================================================================
# WARM-UP — SIEMPRE LLAMAR ANTES DE CUALQUIER SIMULACIÓN
# =============================================================================

def warm_up():
    """
    Compila el JIT de Numba con valores ficticios.

    ⚠️  Llamar SIEMPRE al inicio del script, antes de cualquier simulación.
    Sin esto, la primera simulación incluye el tiempo de compilación (~15 s)
    y los tiempos reportados no son representativos.
    """
    t0 = time.perf_counter()
    _p = _preparar_params(1e5, 2*np.pi*25000, 5e-6)
    _nucleo_km(0.0, 5e-6, 0.0, *_p)
    _t = np.array([0.0])
    _R = np.array([5e-6])
    _V = np.array([0.0])
    _aceleraciones_paralelo(_t, _R, _V, *_p)
    logger.info('Warm-up JIT completo en %.1f s', time.perf_counter()-t0)


# =============================================================================
# EJEMPLO DE USO — mapa de estabilidad
# =============================================================================

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s | %(message)s')

    warm_up()

    # Punto único
    cond = estabilidad_difusiva(Pa=1.3e5, R0=5.0e-6, c_inf_c0=0.002)
    print(f'Punto único → {cond}')

    # Mapa de estabilidad (ejemplo pequeño)
    #Pa_vals = np.linspace(1.0e5, 1.4e5, 5)
    #R0_vals = np.linspace(2e-6, 8e-6, 5)
#
    #print(f'\n{"R0 \\ Pa":>10}', end='')
    #for Pa in Pa_vals:
    #    print(f'  {Pa/P0:.2f} atm', end='')
    #print()
#
    #for R0 in R0_vals:
    #    print(f'{R0*1e6:8.1f} µm', end='')
    #    for Pa in Pa_vals:
    #        c = estabilidad_difusiva(Pa, R0, c_inf_c0=0.002)
    #        etiqueta = {'ESTABLE': 'EST', 'INESTABLE': 'INE',
    #                    'IMPLOSION': 'IMP', 'DISOLUCION': 'DIS',
    #                    'ERROR': 'ERR'}.get(c, '???')
    #        print(f'  {etiqueta:>10}', end='')
    #    print()
