#"""
#================================================================================
#SONOLUMINISCENCIA DE BURBUJA ÚNICA (SBSL) — Diagrama de Fase de Estabilidad
#================================================================================
#Modelo:  Ecuación de Keller-Miksis (compresibilidad de primer orden)
#Gas:     Argón (adiabático, γ = 1.66)
#Líquido: Agua a 20°C
#
#Descripción:
#    Se barre un espacio de parámetros 2D (R0 × Pa) y para cada punto se integra
#    la dinámica de la burbuja durante varios ciclos acústicos. El resultado es un
#    mapa de estabilidad que distingue burbujas "estables" (SBSL) de burbujas
#    que colapsan o crecen sin control ("breakup").
#
#Dependencias: numpy, scipy, matplotlib, joblib
#    pip install numpy scipy matplotlib joblib
#
#Ejecución recomendada:
#    python sbsl_phase_diagram.py
#
#Escalado de resolución:
#    • Prueba rápida:   RESOLUCION_R0 = RESOLUCION_Pa = 15  (~1 min en i7)
#    • Publicación:     RESOLUCION_R0 = RESOLUCION_Pa = 60  (~10 min en i7)
#================================================================================
#"""

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

warnings.filterwarnings("ignore")   # Silenciar advertencias internas de scipy


# ============================================================
# SECCIÓN 1 — CONSTANTES FÍSICAS
# Agua a 20°C y Argón como gas interior de la burbuja
# ============================================================

P0      = 101325.0    # [Pa]    Presión atmosférica estática
F_DRIVE = 25e3        # [Hz]    Frecuencia de la onda acústica
RHO_L   = 998.0       # [kg/m³] Densidad del agua
MU      = 1e-3        # [Pa·s]  Viscosidad dinámica del agua
SIGMA   = 0.0725      # [N/m]   Tensión superficial agua-vapor
C_L     = 1481.0      # [m/s]   Velocidad del sonido en el agua
GAMMA   = 1.66        # [—]     Índice adiabático del Argón
P_VAP   = 2330.0      # [Pa]    Presión de vapor del agua a 20°C

OMEGA   = 2.0 * np.pi * F_DRIVE   # [rad/s] Frecuencia angular


# ============================================================
# SECCIÓN 2 — ECUACIÓN DE KELLER-MIKSIS (KM)
# ============================================================
# Forma compacta del lado derecho (RHS) del sistema de 1er orden.
#
# Ecuación original:
#
#  (1 - Ṙ/c) R R̈  +  (3/2) Ṙ² (1 - Ṙ/3c)
#      = (1 + Ṙ/c)(p_B − p_inf)/ρ  +  R/(ρ c) ṗ_B
#
# Variables de estado:
#   y[0] = R    radio de la burbuja [m]
#   y[1] = Ṙ   velocidad de la pared [m/s]
# ============================================================

def keller_miksis(t, y, R0, Pa):
    """
    Lado derecho del sistema ODE de Keller-Miksis.

    Parámetros
    ----------
    t   : float   Tiempo actual [s]
    y   : array   [R, Ṙ]  Estado actual
    R0  : float   Radio en reposo [m]
    Pa  : float   Amplitud de presión acústica [Pa]

    Retorna
    -------
    list  [dR/dt, dṘ/dt] = [Ṙ, R̈]
    """
    R, Rdot = y

    # Salvaguarda numérica: radio nulo o negativo
    if R <= 1e-12:
        return [0.0, 0.0]

    # ── Presión del gas en equilibrio estático ────────────────────────────
    # Derivada del balance de fuerzas en la interfaz cuando Ṙ = 0 y R = R0:
    #   p_g0 + P_vap = P0 + 2σ/R0  →  p_g0 = P0 + 2σ/R0 − P_vap
    p_g0 = P0 + 2.0 * SIGMA / R0 - P_VAP

    # ── Presión interna p_B(R, Ṙ) ────────────────────────────────────────
    # Contribuciones:
    #   • gas adiabático:      p_g0 · (R0/R)^(3γ)
    #   • vapor saturado:      P_vap  (constante)
    #   • corrección Laplace:  −2σ/R
    #   • disipación viscosa:  −4μ Ṙ/R  (ecuación de Young-Laplace dinámica)
    p_B = (p_g0 * (R0 / R) ** (3.0 * GAMMA)
           + P_VAP
           - 2.0 * SIGMA / R
           - 4.0 * MU * Rdot / R)

    # ── Derivada analítica ṗ_B = dp_B/dt ─────────────────────────────────
    # Usando dR/dt = Ṙ  y  d²R/dt² = R̈ (que aún no conocemos → ignorar
    # el término ∝ R̈ que viene de d/dt[4μṘ/R], es de orden μ·R̈/R ≪ demás)
    #
    #   ṗ_gas  = −3γ p_g0 (R0/R)^(3γ) Ṙ/R
    #   ṗ_σ    = +2σ Ṙ/R²
    #   ṗ_visc = −4μ Ṙ²/R²  (de d/dt[Ṙ/R] ≈ Ṙ²/R² cuando R̈ es suave)
    dp_B_dt = (-3.0 * GAMMA * p_g0 * (R0 / R) ** (3.0 * GAMMA) * Rdot / R
               + 2.0 * SIGMA * Rdot / R ** 2
               - 4.0 * MU * Rdot ** 2 / R ** 2)

    # ── Presión forzante en el infinito ──────────────────────────────────
    p_inf = P0 - Pa * np.sin(OMEGA * t)

    # ── Ensamblado de la ecuación KM ─────────────────────────────────────
    # Reescrita como:  A · R̈  =  B
    delta_p = p_B - p_inf

    A = (1.0 - Rdot / C_L) * R

    B = ((1.0 + Rdot / C_L) * delta_p / RHO_L
         + R / (RHO_L * C_L) * dp_B_dt
         - 1.5 * Rdot ** 2 * (1.0 - Rdot / (3.0 * C_L)))

    if abs(A) < 1e-30:
        return [Rdot, 0.0]

    Rddot = B / A

    return [Rdot, Rddot]


# ============================================================
# SECCIÓN 3 — EVENTOS DE TERMINACIÓN TEMPRANA
# Usados por solve_ivp para detectar fallas físicas en vuelo
# ============================================================

def event_collapse(t, y, R0, Pa):
    """Activar cuando R ≤ 0 (colapso total — no físico)."""
    return y[0]

event_collapse.terminal  = True
event_collapse.direction = -1   # Solo cruce descendente


def event_expansion(t, y, R0, Pa):
    """Activar cuando R > 100·R0 (expansión explosiva — Breakup)."""
    return y[0] - 100.0 * R0

event_expansion.terminal  = True
event_expansion.direction = 1   # Solo cruce ascendente


# ============================================================
# SECCIÓN 4 — EVALUACIÓN DE UN PUNTO DEL ESPACIO PARAMÉTRICO
# Función pura, serializable por joblib (sin lambdas ni closures)
# ============================================================

def evaluate_point(R0, Pa, n_cycles=5, rtol=1e-8, atol=1e-12):
    """
    Integra la ecuación KM para un par (R0, Pa) y clasifica la burbuja.

    Criterios de inestabilidad (Breakup):
        1. El integrador chocó con un evento (colapso o expansión explosiva).
        2. El integrador no alcanzó el 95% del tiempo final.
        3. El radio final es < 0.1·R0  o  > 50·R0.
        4. La velocidad de pared al final supera 50% de la velocidad del sonido.

    Retorna
    -------
    status : int    1 = estable, 0 = inestable
    R_max  : float  Radio máximo durante la simulación [m]
    """
    T_period = 1.0 / F_DRIVE
    t_end    = n_cycles * T_period

    y0 = [R0, 0.0]  # Condición inicial: burbuja en reposo

    try:
        sol = solve_ivp(
            fun      = keller_miksis,
            t_span   = (0.0, t_end),
            y0       = y0,
            method   = 'Radau',              # Integrador implícito para ODEs rígidas
            args     = (R0, Pa),
            events   = [event_collapse, event_expansion],
            rtol     = rtol,
            atol     = atol,
            dense_output = False,
            max_step = T_period / 200        # Garantizar resolución mínima
        )

        # Criterio 1: evento disparado
        if sol.status == 1:
            return 0, np.nan

        # Criterio 2: el integrador no llegó al final
        if sol.t[-1] < 0.95 * t_end:
            return 0, np.nan

        R_final   = sol.y[0, -1]
        Rd_final  = sol.y[1, -1]
        R_max_val = float(np.nanmax(sol.y[0]))

        # Criterio 3: radio final fuera de rango físico
        if R_final < 0.1 * R0 or R_final > 50.0 * R0:
            return 0, R_max_val

        # Criterio 4: velocidad de pared excesiva al final del ciclo
        if abs(Rd_final) > 0.5 * C_L:
            return 0, R_max_val

        return 1, R_max_val

    except Exception:
        return 0, np.nan


# ============================================================
# SECCIÓN 5 — BARRIDO PARAMÉTRICO PARALELO (Grid Search)
# ============================================================

def run_phase_diagram(n_R0=40, n_Pa=40, n_jobs=-1, n_cycles=5):
    """
    Barre la cuadrícula R0 × Pa y construye el mapa de estabilidad.

    Parámetros
    ----------
    n_R0     : int  Puntos en el eje R0
    n_Pa     : int  Puntos en el eje Pa
    n_jobs   : int  Workers paralelos (−1 = todos los núcleos detectados)
    n_cycles : int  Ciclos acústicos por simulación

    Retorna
    -------
    R0_arr    : (n_R0,)       Valores de R0 [m]
    Pa_arr    : (n_Pa,)       Valores de Pa [Pa]
    stability : (n_Pa, n_R0) Mapa binario  1=estable / 0=inestable
    Rmax_map  : (n_Pa, n_R0) Radio máximo [m]
    """
    R0_arr = np.linspace(1e-6, 10e-6, n_R0)    # 1 μm → 10 μm
    Pa_arr = np.linspace(1.0e5, 1.6e5, n_Pa)   # 1.0 atm → ~1.58 atm

    # Lista plana de tareas: cada elemento es (i_R0, j_Pa, R0, Pa)
    tasks = [
        (i, j, R0, Pa)
        for i, R0 in enumerate(R0_arr)
        for j, Pa in enumerate(Pa_arr)
    ]

    print(f"\n{'═'*60}")
    print(f"  SBSL — Barrido paramétrico  R0 × Pa")
    print(f"  Resolución : {n_R0} × {n_Pa} = {len(tasks)} puntos")
    print(f"  Ciclos/pto : {n_cycles}")
    print(f"  Workers    : {'auto (todos los núcleos)' if n_jobs==-1 else n_jobs}")
    print(f"{'═'*60}\n")

    t0 = time.time()

    # ── Paralelización con joblib ─────────────────────────────────────────
    # • prefer="threads" es eficiente porque scipy libera el GIL durante
    #   la integración numérica.
    # • Para funciones puras sin estado compartido es seguro y sin overhead
    #   de serialización (pickling).
    results = Parallel(n_jobs=n_jobs, verbose=5, prefer="threads")(
        delayed(evaluate_point)(R0, Pa, n_cycles)
        for (_, _, R0, Pa) in tasks
    )

    elapsed = time.time() - t0
    n_stable = sum(r[0] for r in results)
    print(f"\n  Tiempo total     : {elapsed:.1f} s")
    print(f"  Tiempo por punto : {elapsed / len(tasks) * 1000:.1f} ms")
    print(f"  Puntos estables  : {n_stable}/{len(tasks)} "
          f"({100*n_stable/len(tasks):.1f}%)")

    # ── Reconstrucción de matrices 2D ─────────────────────────────────────
    stability = np.zeros((n_Pa, n_R0), dtype=np.int8)
    Rmax_map  = np.full((n_Pa, n_R0), np.nan)

    for idx, (i, j, _, _) in enumerate(tasks):
        status, R_max     = results[idx]
        stability[j, i]   = status
        Rmax_map[j, i]    = R_max

    return R0_arr, Pa_arr, stability, Rmax_map


# ============================================================
# SECCIÓN 6 — VISUALIZACIÓN DEL DIAGRAMA DE FASE
# ============================================================

def plot_phase_diagram(R0_arr, Pa_arr, stability, Rmax_map):
    """
    Genera y guarda la figura con dos paneles:
        Izquierdo : Mapa binario de estabilidad (verde / rojo)
        Derecho   : Mapa de radio máximo relativo Rmax/R0 (solo zona estable)
    """
    R0_um  = R0_arr * 1e6        # Convertir a μm para el eje X
    Pa_atm = Pa_arr / P0         # Convertir a atm para el eje Y

    BG     = '#0d0d1a'
    PANEL  = '#111122'
    WHITE  = '#e8e8f0'
    GREEN  = '#2ecc71'
    RED    = '#e74c3c'
    ACCENT = '#00d4ff'

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    fig.patch.set_facecolor(BG)

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor(PANEL)
        ax.set_title(title, fontsize=13.5, color=WHITE, pad=10, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12, color=WHITE, labelpad=8)
        ax.set_ylabel(ylabel, fontsize=12, color=WHITE, labelpad=8)
        ax.tick_params(colors=WHITE, labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    # ── Panel 1: Estabilidad binaria ──────────────────────────────────────
    ax1 = axes[0]
    cmap_bin = ListedColormap([RED, GREEN])   # 0 → rojo, 1 → verde

    ax1.pcolormesh(R0_um, Pa_atm, stability,
                   cmap=cmap_bin, vmin=0, vmax=1, shading='auto')

    # Contorno de la frontera entre zonas
    cs = ax1.contour(R0_um, Pa_atm, stability,
                     levels=[0.5], colors=[ACCENT], linewidths=1.8)
    ax1.clabel(cs, fmt={0.5: 'Frontera'}, fontsize=9, colors=[ACCENT])

    style_ax(ax1,
             title   = 'Diagrama de Fase  —  Estabilidad de Burbuja',
             xlabel  = 'Radio en Reposo  $R_0$  [μm]',
             ylabel  = 'Presión Acústica  $P_a$  [atm]')

    # Etiquetas de región
    ax1.text(0.97, 0.92, 'ESTABLE\n(SBSL)', transform=ax1.transAxes,
             ha='right', va='top', fontsize=11, color=GREEN,
             bbox=dict(boxstyle='round,pad=0.3', facecolor=PANEL, edgecolor=GREEN, alpha=0.8))
    ax1.text(0.05, 0.08, 'INESTABLE\n(Breakup)', transform=ax1.transAxes,
             ha='left', va='bottom', fontsize=11, color=RED,
             bbox=dict(boxstyle='round,pad=0.3', facecolor=PANEL, edgecolor=RED, alpha=0.8))

    # Leyenda
    handles = [
        mpatches.Patch(color=GREEN, label='Estable — SBSL'),
        mpatches.Patch(color=RED,   label='Inestable — Breakup'),
        plt.Line2D([0], [0], color=ACCENT, lw=1.8, label='Frontera de estabilidad'),
    ]
    ax1.legend(handles=handles, loc='upper left', fontsize=10,
               facecolor=PANEL, edgecolor='#444', labelcolor=WHITE)

    # ── Panel 2: Radio máximo relativo ───────────────────────────────────
    ax2 = axes[1]
    R0_grid, _ = np.meshgrid(R0_arr, Pa_arr)   # shape → (n_Pa, n_R0)
    ratio_map  = Rmax_map / R0_grid
    ratio_map[stability == 0] = np.nan          # Enmascarar zona inestable

    vmax = np.nanpercentile(ratio_map, 98) if not np.all(np.isnan(ratio_map)) else 10

    im2 = ax2.pcolormesh(R0_um, Pa_atm, ratio_map,
                         cmap='plasma', shading='auto',
                         vmin=1.0, vmax=vmax)

    cbar = fig.colorbar(im2, ax=ax2, pad=0.02, fraction=0.04)
    cbar.set_label('$R_{\\mathrm{max}} \\,/\\, R_0$', fontsize=12, color=WHITE)
    cbar.ax.yaxis.set_tick_params(color=WHITE, labelcolor=WHITE)

    style_ax(ax2,
             title  = 'Expansión Relativa  $R_{max}/R_0$  (zona estable)',
             xlabel = 'Radio en Reposo  $R_0$  [μm]',
             ylabel = 'Presión Acústica  $P_a$  [atm]')

    # ── Título global ─────────────────────────────────────────────────────
    fig.suptitle(
        'Sonoluminiscencia de Burbuja Única (SBSL)  —  Ecuación de Keller-Miksis\n'
        'Agua 20 °C  /  Argón  (γ = 1.66)  |  f = 25 kHz  |  Modelo Adiabático',
        fontsize=14.5, color=WHITE, y=1.025, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 1])
    output_name = 'sbsl_phase_diagram.png'
    plt.savefig(output_name, dpi=160, bbox_inches='tight', facecolor=BG)
    print(f"\n  ✓ Figura guardada: {output_name}")
    plt.show()


# ============================================================
# SECCIÓN 7 — DIAGNÓSTICO DE UN PUNTO INDIVIDUAL (Verificación)
# Grafica R(t) y Ṙ(t) para un único (R0, Pa).
# Ejecutar antes del barrido para validar el modelo.
# ============================================================

def plot_single_dynamics(R0=5e-6, Pa=1.35e5, n_cycles=5):
    """
    Integra y visualiza la dinámica temporal de la burbuja para un
    único punto del espacio paramétrico. Útil para debug y validación.
    """
    T  = 1.0 / F_DRIVE
    t_eval = np.linspace(0, n_cycles * T, 8000)

    print(f"\n  Integrando punto individual: R0={R0*1e6:.2f} μm, Pa={Pa/P0:.3f} atm")

    sol = solve_ivp(
        keller_miksis,
        (0, n_cycles * T),
        [R0, 0.0],
        method='Radau',
        args=(R0, Pa),
        t_eval=t_eval,
        rtol=1e-10, atol=1e-14,
        max_step=T / 500
    )

    t_us = sol.t * 1e6          # μs
    R_um = sol.y[0] * 1e6       # μm
    Rd   = sol.y[1]             # m/s

    BG, PANEL = '#0d0d1a', '#111122'

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    fig.patch.set_facecolor(BG)

    for ax, y_data, ylabel, color in [
        (ax1, R_um, 'Radio  $R(t)$  [μm]',          '#00d4ff'),
        (ax2, Rd,   'Velocidad de pared  $\\dot{R}(t)$  [m/s]', '#ff6b35'),
    ]:
        ax.set_facecolor(PANEL)
        ax.plot(t_us, y_data, color=color, lw=1.2)
        ax.set_ylabel(ylabel, fontsize=12, color='#e8e8f0')
        ax.tick_params(colors='#e8e8f0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    ax1.axhline(R0 * 1e6, color='#f1c40f', ls='--', lw=1,
                label=f'$R_0$ = {R0*1e6:.1f} μm')
    ax1.legend(fontsize=10, facecolor=PANEL, edgecolor='#444', labelcolor='#e8e8f0')
    ax1.set_title(
        f'Dinámica KM  —  $R_0$ = {R0*1e6:.1f} μm,  '
        f'$P_a$ = {Pa/P0:.3f} atm,  {n_cycles} ciclos',
        fontsize=13, color='#e8e8f0', pad=8, fontweight='bold')

    ax2.axhline(0, color='#555', lw=0.8)
    ax2.set_xlabel('Tiempo  [μs]', fontsize=12, color='#e8e8f0')

    fig.suptitle('Sonoluminiscencia de Burbuja Única — Verificación individual',
                 fontsize=14, color='#e8e8f0', y=1.01, fontweight='bold')

    plt.tight_layout()
    plt.savefig('sbsl_single_dynamics.png', dpi=160,
                bbox_inches='tight', facecolor=BG)
    print("  ✓ Figura individual guardada: sbsl_single_dynamics.png")
    plt.show()


# ============================================================
# SECCIÓN 8 — PUNTO DE ENTRADA PRINCIPAL
# ============================================================

if __name__ == '__main__':

    # ── PASO 0 (opcional): Verificación de un punto conocido ─────────────
    # Descomenta para validar el modelo antes de lanzar el barrido completo.
    # Un R0=5μm, Pa=1.35atm suele mostrar oscilaciones SBSL claras.
    #
    plot_single_dynamics(R0=5e-6, Pa=1.35e5, n_cycles=5)

    # ── PASO 1: Barrido paramétrico paralelo ──────────────────────────────
    # Ajustar resolución según tiempo disponible:
    #   15×15  → prueba rápida   (~1 min   en i7-14núcleos)
    #   40×40  → resultado medio (~8 min   en i7-14núcleos)
    #   60×60  → publicación     (~18 min  en i7-14núcleos)
    #
    RESOLUCION_R0 = 40      # ← Aumentar para figura final
    RESOLUCION_Pa = 40      # ← Aumentar para figura final

    R0_arr, Pa_arr, stability, Rmax_map = run_phase_diagram(
        n_R0     = RESOLUCION_R0,
        n_Pa     = RESOLUCION_Pa,
        n_jobs   = -1,           # Usar todos los hilos disponibles (20 en tu i7)
        n_cycles = 5             # 5 ciclos aseguran régimen estacionario en SBSL
    )

    # ── PASO 2: Generar y guardar la figura ───────────────────────────────
    plot_phase_diagram(R0_arr, Pa_arr, stability, Rmax_map)

    # ── PASO 3: Exportar matrices para postproceso en Python/MATLAB ───────
    np.save('stability_map.npy', stability)
    np.save('rmax_map.npy',      Rmax_map)
    np.save('R0_array.npy',      R0_arr)
    np.save('Pa_array.npy',      Pa_arr)
    print("\n  ✓ Datos numéricos guardados (.npy)")
    print("\n  ¡Simulación SBSL completada exitosamente!\n")