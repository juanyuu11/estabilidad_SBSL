import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Motor gráfico interactivo para que salte la ventana
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
from numba import njit

# =============================================================================
# 1. PARÁMETROS FÍSICOS (Ajusta el Pa según lo que tenías)
# =============================================================================
P0 = 1.0e5         # Presión atmosférica (Pa)
rho = 998.0          # Densidad del agua (kg/m^3)
c = 1481.0            # Velocidad del sonido en agua (m/s)
sigma = 0.0725        # Tensión superficial (N/m)
eta = 0.001           # Viscosidad dinámica (Pa·s)
R0_v = 5.0e-6         # Radio inicial de la burbuja (5 micrometros)
h_v = R0_v / 8.86      # Radio del núcleo duro de Van der Waals (aprox)
gamma_v = 5/3         # Índice politrópico (gas diatómico)
f_v = 26.5e3        # Frecuencia acústica (26.5 kHz)
omega_v = 2.0 * np.pi * f_v
Pa_v = 1.2e5      # Amplitud de presión acústica (Ajusta este valor)

args_v = (rho, c, Pa_v, omega_v, P0, sigma, eta, R0_v, h_v, gamma_v)

# =============================================================================
# 2. ECUACIÓN DE KELLER-MIKSIS (PURA, SIN NUMBA)
# =============================================================================
@njit
def sbsl_ode_raw(t, y, rho, c, Pa, omega, P0, sigma, eta, R0, h, gamma):
    R = y[0]
    V = y[1]
    
    # Límite de seguridad para evitar divisiones por cero cerca del núcleo duro
    R_safe = max(R, h + 1e-12)
    
    R2 = R_safe**2
    R3 = R_safe**3
    h3 = h**3
    denom = R3 - h3
    
    ratio = (R0**3 - h3) / denom
    
    # Presión del gas con la corrección de Laplace en t=0
    Pg0 = P0 + (2.0 * sigma) / R0 
    Pg = Pg0 * (ratio**gamma)
    
    # Derivada de la presión
    dPg_dR = - (3.0 * gamma * R2 / denom) * Pg
    dPg_dt = dPg_dR * V  
    
    # Presión de forzamiento acústico
    P_drive = -Pa * np.sin(omega * t)
    
    # Ecuación principal (términos agrupados)
    F = Pg - P0 - P_drive - (4.0 * eta * V) / R_safe - (2.0 * sigma) / R_safe
    F_rad = (R_safe / c) * dPg_dt
    
    dR_dt = V
    dV_dt = (F + F_rad - 1.5 * rho * V**2) / (rho * R_safe)
    
    return [dR_dt, dV_dt]

# =============================================================================
# 3. SIMULACIÓN Y CRONÓMETRO
# =============================================================================
y0 = [R0_v, 0.0]                  
t_span = [0, 800e-6]  # 800 microsegundos

# Cálculo automático de resolución: 1000 puntos por ciclo acústico
periodo_acustico = 1.0 / f_v
num_ciclos = t_span[1] / periodo_acustico
puntos_totales = int(num_ciclos * 1000)
t_eval = np.linspace(t_span[0], t_span[1], puntos_totales)

print("Iniciando simulación lenta (Python puro + Radau ciego)...")
print(f"Simulando {t_span[1]*1e6} us ({int(num_ciclos)} ciclos acústicos)")

inicio = time.perf_counter()

# Usamos Radau, pero sin entregarle la matriz Jacobiana
sol = solve_ivp(
    sbsl_ode_raw, 
    t_span, 
    y0, 
    method='Radau', 
    t_eval=t_eval,
    args=args_v,
    rtol=1e-8,   # Tolerancias relajadas como discutimos
    atol=1e-10,
    first_step=1e-11 
)

fin = time.perf_counter()
tiempo_total = fin - inicio

print("-" * 40)
print(f"¡Éxito: {sol.success}!")
if not sol.success:
    print(f"Mensaje: {sol.message}")
print(f"Evaluaciones de la función: {sol.nfev}")
print(f"Evaluaciones del Jacobiano (aproximadas numéricamente): {sol.njev}")
print(f"⏱️ TIEMPO TOTAL: {tiempo_total:.4f} segundos")
print("-" * 40)

# =============================================================================
# 4. GRÁFICA
# =============================================================================
if sol.success:
    plt.figure(figsize=(12, 6))
    plt.plot(sol.t * 1e6, sol.y[0] * 1e6, label=r'Radio $R(t)$', color='#1f77b4', linewidth=1.5)
    plt.axhline(R0_v * 1e6, color='gray', linestyle='--', alpha=0.7, label=r'$R_0 = 5\,\mu m$')
    
    plt.title('Dinámica de Burbuja - Python Puro (Sin Numba, Sin Jacobiano Analítico)')
    plt.xlabel(r'Tiempo ($\mu s$)')
    plt.ylabel(r'Radio ($\mu m$)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()