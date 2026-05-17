"""
benchmark_jacobiano.py
======================
Comparación estadística rigurosa entre:
  - radym_claude_conjac.py  (Radau + Jacobiano analítico)
  - radym_claude_sinjac.py  (Radau sin Jacobiano)

Ambos módulos tienen las mismas funciones; se cargan con importlib
en espacios de nombres separados para evitar colisiones.
"""

import importlib.util, sys, time, pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
PA          = 1.2e5   # Pa  — mismo que en los scripts originales
N_RUNS      = 10      # repeticiones por método — suficiente para Shapiro + Mann-Whitney
ALPHA       = 0.05    # nivel de significancia
SCRIPT_DIR  = pathlib.Path(__file__).parent  # mismo directorio que este script


# =============================================================================
# 1. CARGA DE MÓDULOS CON ESPACIOS DE NOMBRES SEPARADOS
#    importlib.util.spec_from_file_location permite cargar dos archivos
#    con funciones de igual nombre sin que se pisen entre sí.
# =============================================================================
def cargar_modulo(nombre_alias: str, ruta: pathlib.Path):
    spec = importlib.util.spec_from_file_location(nombre_alias, ruta)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[nombre_alias] = mod   # registrar para que numba/pickle funcionen
    spec.loader.exec_module(mod)
    return mod

print("Cargando módulos…")
mod_conjac = cargar_modulo("conjac", SCRIPT_DIR / "radym_claude_conjac.py")
mod_sinjac = cargar_modulo("sinjac", SCRIPT_DIR / "radym_claude_sinjac.py")

# =============================================================================
# 2. WARM-UP DEL JIT
#    Se ejecuta UNA sola vez por módulo. El primer run cronometrado
#    ya usa el binario cacheado → tiempos limpios.
# =============================================================================
print("\nWarm-up JIT (conjac)…")
mod_conjac.warm_up()
print("Warm-up JIT (sinjac)…")
mod_sinjac.warm_up()


# =============================================================================
# 3. BENCHMARK — runs aleatorizados e intercalados
#
# Los 2×N_RUNS runs se ordenan aleatoriamente ANTES de ejecutarlos.
# Esto elimina sesgos temporales: si la CPU throttlea al calentarse,
# o si el SO asigna más prioridad al inicio de la sesión, ambos métodos
# resultan igualmente afectados en promedio.
# =============================================================================
METODOS = {
    "conjac": mod_conjac,
    "sinjac": mod_sinjac,
}

# Lista completa de trabajos aleatorizados
rng_bench = np.random.default_rng(0)
trabajos  = rng_bench.permutation(
    ["conjac"] * N_RUNS + ["sinjac"] * N_RUNS
).tolist()

acumulados = {"conjac": [], "sinjac": []}
total = len(trabajos)

print(f"\nEjecutando {total} runs en orden aleatorizado…")
for idx, metodo in enumerate(trabajos, 1):
    mod = METODOS[metodo]
    t0  = time.perf_counter()
    rec, ok = mod.simular_burbuja_estacionaria(PA)
    t1  = time.perf_counter()
    if not ok:
        raise RuntimeError(f"[{metodo}] run {idx} falló.")
    dt = t1 - t0
    acumulados[metodo].append(dt)
    n_done = len(acumulados[metodo])
    print(f"  [{idx:02d}/{total}] {metodo}  run {n_done}/{N_RUNS}  →  {dt:.2f} s")

t_conjac = np.array(acumulados["conjac"])
t_sinjac = np.array(acumulados["sinjac"])


# =============================================================================
# 4. ESTADÍSTICAS DESCRIPTIVAS
# =============================================================================
def describir(arr: np.ndarray, etiqueta: str) -> dict:
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "etiqueta" : etiqueta,
        "n"        : len(arr),
        "media"    : arr.mean(),
        "mediana"  : np.median(arr),
        "std"      : arr.std(ddof=1),
        "cv_%"     : arr.std(ddof=1) / arr.mean() * 100,
        "min"      : arr.min(),
        "max"      : arr.max(),
        "q1"       : q1,
        "q3"       : q3,
        "iqr"      : q3 - q1,
        "ic95_low" : arr.mean() - 1.96 * arr.std(ddof=1) / np.sqrt(len(arr)),
        "ic95_high": arr.mean() + 1.96 * arr.std(ddof=1) / np.sqrt(len(arr)),
    }

d_cj = describir(t_conjac, "conjac")
d_sj = describir(t_sinjac, "sinjac")


# =============================================================================
# 5. PRUEBAS ESTADÍSTICAS
# =============================================================================

# 5a. Normalidad — Shapiro-Wilk (válido para N < 50)
sw_cj = stats.shapiro(t_conjac)
sw_sj = stats.shapiro(t_sinjac)
normal_cj = sw_cj.pvalue > ALPHA
normal_sj = sw_sj.pvalue > ALPHA

# 5b. Prueba paramétrica — Welch t-test (no asume varianzas iguales)
t_stat, p_welch = stats.ttest_ind(t_conjac, t_sinjac, equal_var=False)

# 5c. Prueba no paramétrica — Mann-Whitney U (más robusta con N pequeño)
u_stat, p_mwu = stats.mannwhitneyu(t_conjac, t_sinjac, alternative='two-sided')

# 5d. Tamaño del efecto — Cohen's d
pooled_std = np.sqrt(
    ((len(t_conjac)-1)*t_conjac.var(ddof=1) + (len(t_sinjac)-1)*t_sinjac.var(ddof=1))
    / (len(t_conjac) + len(t_sinjac) - 2)
)
cohen_d = (d_cj["media"] - d_sj["media"]) / pooled_std

def interpretar_cohen(d):
    d = abs(d)
    if d < 0.2:  return "despreciable"
    if d < 0.5:  return "pequeño"
    if d < 0.8:  return "mediano"
    return "grande"

# 5e. Intervalo de confianza de la diferencia de medias (bootstrap, 10 000 muestras)
rng = np.random.default_rng(42)
diffs_boot = np.array([
    rng.choice(t_conjac, size=N_RUNS, replace=True).mean()
    - rng.choice(t_sinjac, size=N_RUNS, replace=True).mean()
    for _ in range(10_000)
])
ic_diff_low, ic_diff_high = np.percentile(diffs_boot, [2.5, 97.5])


# =============================================================================
# 6. REPORTE EN CONSOLA
# =============================================================================
SEP = "=" * 62

print(f"\n{SEP}")
print("  REPORTE DE BENCHMARK — Keller-Miksis  (Pa = {:.1e})(rtol = 1e-9)(atol = 1e-11)".format(PA))
print(SEP)

header = f"{'Métrica':<22} {'conjac':>16} {'sinjac':>16}"
print(header)
print("-" * len(header))

metricas = [
    ("N runs",           "n",         "",    "d"),
    ("Media (s)",        "media",     "",    ".3f"),
    ("Mediana (s)",      "mediana",   "",    ".3f"),
    ("Desv. estándar",   "std",       "",    ".3f"),
    ("CV (%)",           "cv_%",      "",    ".1f"),
    ("Mínimo (s)",       "min",       "",    ".3f"),
    ("Máximo (s)",       "max",       "",    ".3f"),
    ("Q1 (s)",           "q1",        "",    ".3f"),
    ("Q3 (s)",           "q3",        "",    ".3f"),
    ("IQR (s)",          "iqr",       "",    ".3f"),
    ("IC 95% bajo (s)",  "ic95_low",  "",    ".3f"),
    ("IC 95% alto (s)",  "ic95_high", "",    ".3f"),
]
for label, key, _, fmt in metricas:
    vc = format(d_cj[key], fmt)
    vs = format(d_sj[key], fmt)
    print(f"  {label:<20} {vc:>16} {vs:>16}")

print(f"\n{'PRUEBAS ESTADÍSTICAS':^{len(header)}}")
print("-" * len(header))
print(f"  Shapiro-Wilk conjac  W={sw_cj.statistic:.4f}  p={sw_cj.pvalue:.4f}  "
      f"→ {'normal ✓' if normal_cj else 'NO normal ✗'}")
print(f"  Shapiro-Wilk sinjac  W={sw_sj.statistic:.4f}  p={sw_sj.pvalue:.4f}  "
      f"→ {'normal ✓' if normal_sj else 'NO normal ✗'}")
print(f"\n  Welch t-test         t={t_stat:.4f}  p={p_welch:.4f}  "
      f"→ {'SIGNIFICATIVO' if p_welch < ALPHA else 'no significativo'} (α={ALPHA})")
print(f"  Mann-Whitney U       U={u_stat:.1f}    p={p_mwu:.4f}  "
      f"→ {'SIGNIFICATIVO' if p_mwu < ALPHA else 'no significativo'} (α={ALPHA})")
print(f"\n  Cohen's d            d={cohen_d:.4f}  → efecto {interpretar_cohen(cohen_d)}")
print(f"  Diferencia de medias {d_cj['media']-d_sj['media']:+.3f} s  "
      f"(conjac − sinjac)")
print(f"  IC 95% bootstrap     [{ic_diff_low:.3f}, {ic_diff_high:.3f}] s")

faster = "conjac" if d_cj["media"] < d_sj["media"] else "sinjac"
pct    = abs(d_cj["media"] - d_sj["media"]) / max(d_cj["media"], d_sj["media"]) * 100
print(f"\n  Método más rápido    {faster}  ({pct:.1f}% de diferencia en media)")

sig = p_mwu < ALPHA  # Mann-Whitney más conservador para N pequeño
if not sig:
    print("  ⚠  La diferencia NO es estadísticamente significativa.")
    print("     Ambos métodos son equivalentes en rendimiento para este sistema.")
else:
    print("  ✓  La diferencia ES estadísticamente significativa.")
print(SEP)


# =============================================================================
# 7. FIGURA COMPARATIVA (4 paneles)
# =============================================================================
COLORES = {"conjac": "#2166ac", "sinjac": "#d6604d"}
fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    f"Benchmark Keller-Miksis — conjac vs sinjac\n"
    f"Pa = {PA:.1e} Pa  |  N = {N_RUNS} runs/método",
    fontsize=13, fontweight="bold",
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# ---- Panel A: Boxplot + puntos individuales ----
ax0 = fig.add_subplot(gs[0, 0])
bp  = ax0.boxplot(
    [t_conjac, t_sinjac],
    labels=["conjac", "sinjac"],
    patch_artist=True,
    widths=0.45,
    medianprops=dict(color="white", lw=2),
    whiskerprops=dict(lw=1.2),
    capprops=dict(lw=1.2),
    flierprops=dict(marker="o", ms=5, alpha=0.5),
)
for patch, key in zip(bp["boxes"], ["conjac", "sinjac"]):
    patch.set_facecolor(COLORES[key])
    patch.set_alpha(0.75)

for i, (arr, key) in enumerate([(t_conjac, "conjac"), (t_sinjac, "sinjac")], 1):
    jitter = rng.uniform(-0.12, 0.12, size=len(arr))
    ax0.scatter(i + jitter, arr, color=COLORES[key], s=30, zorder=5, alpha=0.8)

ax0.set_ylabel("Tiempo (s)")
ax0.set_title("A — Boxplot + puntos")
ax0.grid(axis="y", alpha=0.3)

# ---- Panel B: Evolución temporal de los runs ----
ax1 = fig.add_subplot(gs[0, 1])
runs = np.arange(1, N_RUNS + 1)
ax1.plot(runs, t_conjac, "o-", color=COLORES["conjac"], lw=1.5,
         ms=5, label="conjac")
ax1.plot(runs, t_sinjac, "s-", color=COLORES["sinjac"], lw=1.5,
         ms=5, label="sinjac")
ax1.axhline(d_cj["media"], color=COLORES["conjac"], ls="--", lw=1, alpha=0.6)
ax1.axhline(d_sj["media"], color=COLORES["sinjac"], ls="--", lw=1, alpha=0.6)
ax1.set_xlabel("Número de run")
ax1.set_ylabel("Tiempo (s)")
ax1.set_title("B — Evolución por run")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# ---- Panel C: Histograma de distribuciones ----
ax2 = fig.add_subplot(gs[1, 0])
bins = np.linspace(min(t_conjac.min(), t_sinjac.min()) - 1,
                   max(t_conjac.max(), t_sinjac.max()) + 1, 14)
ax2.hist(t_conjac, bins=bins, color=COLORES["conjac"], alpha=0.65,
         label="conjac", edgecolor="white")
ax2.hist(t_sinjac, bins=bins, color=COLORES["sinjac"], alpha=0.65,
         label="sinjac", edgecolor="white")
for d, key in [(d_cj, "conjac"), (d_sj, "sinjac")]:
    ax2.axvline(d["media"],   color=COLORES[key], lw=2, ls="-")
    ax2.axvline(d["mediana"], color=COLORES[key], lw=1.5, ls="--")
ax2.set_xlabel("Tiempo (s)")
ax2.set_ylabel("Frecuencia")
ax2.set_title("C — Distribución de tiempos")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# ---- Panel D: Diferencias bootstrap + IC 95% ----
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(diffs_boot, bins=50, color="#4d9221", alpha=0.75, edgecolor="white")
ax3.axvline(0, color="black", lw=1.5, ls="-",  label="H₀: diferencia = 0")
ax3.axvline(diffs_boot.mean(), color="#1a6e00", lw=2, ls="-",
            label=f"media boot = {diffs_boot.mean():.2f} s")
ax3.axvspan(ic_diff_low, ic_diff_high, alpha=0.2, color="#4d9221",
            label=f"IC 95%: [{ic_diff_low:.2f}, {ic_diff_high:.2f}] s")
ax3.set_xlabel("Diferencia de medias (conjac − sinjac)  [s]")
ax3.set_ylabel("Frecuencia")
ax3.set_title("D — Bootstrap diferencia de medias (10 000)")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# Anotación de p-values en la figura
textstr = (f"Welch  p = {p_welch:.4f}\n"
           f"M-W U  p = {p_mwu:.4f}\n"
           f"Cohen d = {cohen_d:.3f}  ({interpretar_cohen(cohen_d)})")
fig.text(0.5, 0.01, textstr, ha="center", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", alpha=0.8))

plt.savefig("benchmark_jacobiano.png", dpi=150, bbox_inches="tight")
print("\nFigura guardada en benchmark_jacobiano.png")
plt.show()