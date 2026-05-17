"""
================================================================================
SBSL — Diagrama de Fase R0 vs Pa  (CRITERIOS DE ESTABILIDAD CORREGIDOS)
================================================================================
Problema anterior: R_MAX_FACTOR=150 → todo aparece estable (físicamente incorrecto)

Corrección basada en literatura (Hilgenfeldt, Lohse & Brenner, 1998):

  1. Criterio R-T (Rayleigh-Taylor): si R_max/R0 > RT_THRESHOLD (~8-12),
     la burbuja es inestable por inestabilidad de interfaz durante el colapso.

  2. Criterio de colapso: si R cae por debajo de 0.1·R0 (colapso total).

  3. Criterio de Mach: si |Rdot/c| > 0.9 se sale del régimen físico del modelo.

  4. N_CYCLES aumentado a 5 para dar tiempo a que la inestabilidad se manifieste.

  5. Se rastrea R_max DURANTE la integración para aplicar el criterio R-T.
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
import warnings
import time
import os

warnings.filterwarnings("ignore")

# ── Rutas de salida (Windows-compatible) ──────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

def out(filename):
    return os.path.join(OUT_DIR, filename)

# ================================================================================
# SECCIÓN 1 — CONSTANTES FÍSICAS
# ================================================================================
P0    = 101325.0
F     = 25e3
RHO_L = 998.0
MU    = 1e-3
SIGMA = 0.0725
C     = 1481.0
GAMMA = 1.66
P_VAP = 2330.0
OMEGA = 2 * np.pi * F

# ================================================================================
# SECCIÓN 2 — CUADRÍCULA PARAMÉTRICA
# ================================================================================
N_R0 = 50
N_PA = 50

R0_ARRAY = np.linspace(1e-6,  10e-6, N_R0)
PA_ARRAY = np.linspace(1.0e5, 1.6e5, N_PA)

# ================================================================================
# SECCIÓN 3 — PARÁMETROS NUMÉRICOS
# ================================================================================
N_CYCLES = 5                        # 5 ciclos para que la inestabilidad se desarrolle
T_END    = N_CYCLES / F
T_SPAN   = (0.0, T_END)

RTOL     = 1e-6                     # Un poco más estricto que antes
ATOL     = 1e-9
MAX_STEP = 1.0 / (F * 100)         # 1/100 del período acústico

# ── Criterios de estabilidad física ──────────────────────────────────────────
# ESTE ERA EL BUG PRINCIPAL: 150 → cambiado a 10
RT_THRESHOLD  = 10.0   # R_max/R0 > 10 → inestabilidad Rayleigh-Taylor
R_MIN_FACTOR  = 0.10   # R < 0.10·R0  → colapso total
MACH_MAX      = 0.90   # |Rdot/c| > 0.9 → fuera del régimen físico del modelo

N_JOBS = 20

# ================================================================================
# SECCIÓN 4 — ECUACIÓN DE KELLER-MIKSIS
# ================================================================================
def keller_miksis(t, y, R0, Pa, p_g0):
    R, R_dot = y[0], y[1]

    if R < 1e-12:
        return [R_dot, 0.0]

    ratio = R0 / R
    p_B = (p_g0 * ratio ** (3 * GAMMA)
           + P_VAP
           - 2.0 * SIGMA / R
           - 4.0 * MU * R_dot / R)

    dp_B_dt = ((-3.0 * GAMMA * p_g0 * ratio ** (3 * GAMMA) * R_dot / R)
               + 2.0 * SIGMA * R_dot / R**2)

    p_inf = P0 - Pa * np.sin(OMEGA * t)

    Mach      = R_dot / C
    LHS_coeff = (1.0 - Mach) * R

    if abs(LHS_coeff) < 1e-20:
        return [R_dot, 0.0]

    RHS = ((1.0 + Mach) * (p_B - p_inf) / RHO_L
           + (R / (RHO_L * C)) * dp_B_dt
           - 1.5 * R_dot**2 * (1.0 - Mach / 3.0))

    return [R_dot, RHS / LHS_coeff]


# ================================================================================
# SECCIÓN 5 — EVENTOS DE FALLA (solo los físicamente bien calibrados)
# ================================================================================
def make_events(R0):
    R_min  = R_MIN_FACTOR * R0
    R_rt   = RT_THRESHOLD * R0     # umbral Rayleigh-Taylor

    def event_collapse(t, y, *args):
        """Colapso total: R baja de 10% del radio en reposo."""
        return y[0] - R_min
    event_collapse.terminal  = True
    event_collapse.direction = -1

    def event_rt_instability(t, y, *args):
        """R-T: burbuja se expande más de RT_THRESHOLD veces R0."""
        return y[0] - R_rt
    event_rt_instability.terminal  = True
    event_rt_instability.direction = +1

    def event_mach(t, y, *args):
        """Mach: velocidad de pared supera 90% de la velocidad del sonido."""
        return MACH_MAX - abs(y[1]) / C
    event_mach.terminal  = True
    event_mach.direction = -1

    return [event_collapse, event_rt_instability, event_mach]


# ================================================================================
# SECCIÓN 6 — EVALUACIÓN DE UN PUNTO
# ================================================================================
def evaluate_stability(R0, Pa):
    """
    Devuelve:
      1 → ESTABLE   (la burbuja oscila dentro de los límites físicos)
      0 → INESTABLE (R-T, colapso, o Mach supercrítico detectados)
    """
    p_g0   = P0 + 2.0 * SIGMA / R0 - P_VAP
    y0     = [R0, 0.0]
    events = make_events(R0)

    try:
        sol = solve_ivp(
            fun          = keller_miksis,
            t_span       = T_SPAN,
            y0           = y0,
            method       = 'Radau',
            args         = (R0, Pa, p_g0),
            events       = events,
            rtol         = RTOL,
            atol         = ATOL,
            dense_output = False,
            max_step     = MAX_STEP
        )

        # Falla si algún evento se disparó
        if not sol.success or any(len(e) > 0 for e in sol.t_events):
            return 0

        # Verificación post-integración: R_max durante toda la trayectoria
        R_max = sol.y[0].max()
        if R_max > RT_THRESHOLD * R0:
            return 0

        # Velocidad máxima de pared (criterio Mach)
        Rdot_max = np.abs(sol.y[1]).max()
        if Rdot_max / C > MACH_MAX:
            return 0

        # Radio final razonable
        R_final = sol.y[0, -1]
        if R_final < R_MIN_FACTOR * R0:
            return 0

        return 1

    except Exception:
        return 0


# ================================================================================
# SECCIÓN 7 — BARRIDO PARALELO
# ================================================================================
def run_phase_diagram():
    total = N_R0 * N_PA
    print("=" * 65)
    print(f"  SBSL Diagrama de Fase — {N_R0}x{N_PA} = {total} puntos")
    print(f"  Ciclos: {N_CYCLES}  |  RTOL={RTOL}  |  ATOL={ATOL}")
    print(f"  RT_THRESHOLD={RT_THRESHOLD}  R_MIN={R_MIN_FACTOR}  MACH_MAX={MACH_MAX}")
    print(f"  Hilos: {N_JOBS}")
    print("=" * 65)

    params = [(R0, Pa) for Pa in PA_ARRAY for R0 in R0_ARRAY]

    t0 = time.time()
    results = Parallel(n_jobs=N_JOBS, verbose=10, backend='loky')(
        delayed(evaluate_stability)(R0, Pa) for R0, Pa in params
    )
    elapsed = time.time() - t0

    matrix   = np.array(results).reshape(N_PA, N_R0)
    n_stable = int(matrix.sum())
    print(f"\n  Tiempo: {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"  Estables: {n_stable}/{total} ({100*n_stable/total:.1f}%)")

    return matrix


# ================================================================================
# SECCIÓN 8 — VISUALIZACIÓN
# ================================================================================
def plot_phase_diagram(matrix):
    R0_um  = R0_ARRAY * 1e6
    Pa_atm = PA_ARRAY / P0

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('#0d0d0d')
    ax.set_facecolor('#0d0d0d')

    cmap = ListedColormap(['#c0392b', '#27ae60'])
    ax.pcolormesh(R0_um, Pa_atm, matrix, cmap=cmap, vmin=0, vmax=1,
                  shading='nearest')

    ax.set_xlabel(r'Radio en Reposo $R_0$ [µm]', color='white', fontsize=13)
    ax.set_ylabel(r'Presión Acústica $P_a$ [atm]', color='white', fontsize=13)
    ax.set_title(
        'Diagrama de Fase SBSL — Keller–Miksis\n'
        r'Argón  |  $f$ = 25 kHz  |  Agua 20 °C',
        color='white', fontsize=14, pad=10
    )
    ax.tick_params(colors='white', labelsize=10)
    for sp in ax.spines.values():
        sp.set_edgecolor('#555')

    patches = [
        mpatches.Patch(color='#27ae60', label='Estable — SBSL activo'),
        mpatches.Patch(color='#c0392b', label='Inestable — R-T / Colapso'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=10,
              facecolor='#1a1a1a', edgecolor='#555', labelcolor='white')

    info = (f'Cuadricula: {N_R0}x{N_PA}  |  Ciclos: {N_CYCLES}\n'
            f'RT_threshold={RT_THRESHOLD}·R0  |  Mach_max={MACH_MAX}\n'
            f'RTOL={RTOL}  ATOL={ATOL}  |  gamma={GAMMA}')
    ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=7.5,
            color='#aaaaaa', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='#1a1a1a',
                      ec='#444', alpha=0.85))

    plt.tight_layout()
    path = out('sbsl_phase_diagram_v3.png')
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\n  Figura guardada: {path}")
    plt.show()


# ================================================================================
# SECCIÓN 9 — TRAYECTORIA INDIVIDUAL
# Útil para inspeccionar R(t) en un punto específico y calibrar el umbral R-T
# ================================================================================
def plot_single_trajectory(R0=4e-6, Pa=1.3e5, label=""):
    p_g0 = P0 + 2.0 * SIGMA / R0 - P_VAP
    sol  = solve_ivp(keller_miksis, T_SPAN, [R0, 0.0],
                     method='Radau', args=(R0, Pa, p_g0),
                     rtol=1e-8, atol=1e-12,
                     dense_output=True, max_step=1.0/(F*500))

    R_traj = sol.y[0]
    R_max  = R_traj.max()
    print(f"  R_max = {R_max*1e6:.3f} µm  →  R_max/R0 = {R_max/R0:.2f}")
    print(f"  |Rdot|_max / c = {np.abs(sol.y[1]).max()/C:.3f}")

    t_us = sol.t * 1e6
    R_um = R_traj * 1e6

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.patch.set_facecolor('#0d0d0d')

    for ax in axes:
        ax.set_facecolor('#111')
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#555')

    # Radio vs tiempo
    axes[0].plot(t_us, R_um, color='#3498db', lw=0.8)
    axes[0].axhline(R0*1e6, color='#e74c3c', lw=0.8, ls='--',
                    label=f'R0 = {R0*1e6:.1f} µm')
    axes[0].axhline(RT_THRESHOLD*R0*1e6, color='#f39c12', lw=0.8, ls=':',
                    label=f'Umbral R-T ({RT_THRESHOLD}·R0)')
    axes[0].set_ylabel('Radio R [µm]', color='white')
    axes[0].legend(facecolor='#1a1a1a', labelcolor='white', edgecolor='#555',
                   fontsize=8)

    # Velocidad de pared normalizada (Mach)
    Mach_traj = sol.y[1] / C
    axes[1].plot(t_us, Mach_traj, color='#e67e22', lw=0.8)
    axes[1].axhline( MACH_MAX, color='#e74c3c', lw=0.8, ls='--',
                    label=f'Mach_max = {MACH_MAX}')
    axes[1].axhline(-MACH_MAX, color='#e74c3c', lw=0.8, ls='--')
    axes[1].set_ylabel('Ṙ / c  [Mach]', color='white')
    axes[1].set_xlabel('Tiempo [µs]', color='white')
    axes[1].legend(facecolor='#1a1a1a', labelcolor='white', edgecolor='#555',
                   fontsize=8)

    title = (f'Trayectoria SBSL — R0={R0*1e6:.1f} µm  '
             f'Pa={Pa/P0:.2f} atm  {label}')
    fig.suptitle(title, color='white', fontsize=11)
    plt.tight_layout()

    fname = f'sbsl_traj_R0{R0*1e6:.1f}um_Pa{Pa/P0:.2f}atm.png'
    fig.savefig(out(fname), dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  Guardado: {fname}")
    plt.show()


# ================================================================================
# SECCIÓN 10 — MAIN
# ================================================================================
if __name__ == '__main__':

    # --- Diagnóstico previo: inspecciona 3 puntos representativos ---
    # Punto que DEBERÍA ser estable (zona SBSL típica):
    plot_single_trajectory(R0=4e-6, Pa=1.3e5, label="[esperado ESTABLE]")

    # Punto que DEBERÍA ser inestable R-T (Pa alta, R0 grande):
    plot_single_trajectory(R0=8e-6, Pa=1.5e5, label="[esperado INESTABLE R-T]")

    # Punto límite:
    plot_single_trajectory(R0=6e-6, Pa=1.4e5, label="[zona frontera]")

    # --- Barrido completo ---
    matrix = run_phase_diagram()

    np.save(out('stability_matrix.npy'), matrix)
    np.save(out('R0_array.npy'), R0_ARRAY)
    np.save(out('Pa_array.npy'), PA_ARRAY)

    plot_phase_diagram(matrix)