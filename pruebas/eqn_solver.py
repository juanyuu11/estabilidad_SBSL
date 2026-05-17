import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from numba import njit
import time
# =============================================================================
# 1. PARÁMETROS (Tus valores exactos)
# =============================================================================
rho_v = 998.0         # Densidad del agua (kg/m^3)
sigma_v = 0.0725      # Tensión superficial (N/m)
eta_v = 0.001         # Viscosidad (Pa.s)
P0_v = 1.0e5          # Presión hidrostática (Pa)
f_v = 26.5e3          # Frecuencia (Hz)
omega_v = 2 * np.pi * f_v  # Frecuencia angular (rad/s)
R0_v = 5.0e-6         # Radio inicial (m) (5 micrómetros)
h_v = R0_v / 8.86     # Radio del núcleo duro de Van der Waals (m)
gamma_v = 5/3         # Constante politrópica (Argón/Monoatómico)
c_v = 1481.0          # Velocidad del sonido en agua (m/s)
Pa_v = 1.2e5          # Amplitud de presión acústica (Pa)


# Empaquetamos en una tupla respetando el orden del Jacobiano y la ODE
args_v = (rho_v, c_v, Pa_v, omega_v, P0_v, sigma_v, eta_v, R0_v, h_v, gamma_v)

# =============================================================================
# 2. ECUACIÓN DIFERENCIAL (La Física)
# =============================================================================
@njit
def sbsl_ode(t, y, rho, c, Pa, omega, P0, sigma, eta, R0, h, gamma):
    R = y[0]
    V = y[1]
    
    # SOFT CLAMP
    R_safe = max(R, h + 1e-12)
    
    R2 = R_safe**2
    R3 = R_safe**3
    h3 = h**3
    denom = R3 - h3
    
    ratio = (R0**3 - h3) / denom
    Pg = P0 * (ratio**gamma)
    
    dPg_dR = - (3.0 * gamma * R2 / denom) * Pg
    dPg_dt = dPg_dR * V  
    
    P_drive = -Pa * np.sin(omega * t)
    
    F = Pg - P0 - P_drive - (4.0 * eta * V) / R_safe - (2.0 * sigma) / R_safe
    F_rad = (R_safe / c) * dPg_dt
    
    dR_dt = V
    dV_dt = (F + F_rad - 1.5 * rho * V**2) / (rho * R_safe)
    
    # Numba prefiere arreglos de NumPy en lugar de listas de Python
    return np.array([dR_dt, dV_dt])

@njit
def sbsl_jacobian(t, y, rho, c, Pa, omega, P0, sigma, eta, R0, h, gamma):
    R = y[0]
    V = y[1]
    
    # SOFT CLAMP
    R_safe = max(R, h + 1e-12)
    
    R2 = R_safe**2
    R3 = R_safe**3
    h3 = h**3
    denom = R3 - h3
    
    ratio = (R0**3 - h3) / denom
    Pg = P0 * (ratio ** gamma)
    dPg_dR = - (3.0 * gamma * R2 / denom) * Pg
    
    P_drive = -Pa * np.sin(omega * t)
    F = Pg - P0 - P_drive - (4.0 * eta * V) / R_safe - (2.0 * sigma) / R_safe
    F_rad = (R_safe / c) * (dPg_dR * V)
    f2 = (F + F_rad - 1.5 * rho * V**2) / (rho * R_safe)
    
    J00 = 0.0
    J01 = 1.0
    
    dF_dR = dPg_dR + (4.0 * eta * V) / R2 + (2.0 * sigma) / R2
    term1_Frad = (9.0 * gamma * V * R2 * h3) / (c * denom**2) * Pg
    term2_Frad = (9.0 * gamma**2 * V * R_safe**5) / (c * denom**2) * Pg
    dFrad_dR = term1_Frad + term2_Frad
    
    J10 = (dF_dR + dFrad_dR) / (rho * R_safe) - f2 / R_safe
    
    dF_dV = - (4.0 * eta) / R_safe
    dFrad_dV = (R_safe / c) * dPg_dR  
    dKinetic_dV = - 3.0 * rho * V
    
    J11 = (dF_dV + dFrad_dV + dKinetic_dV) / (rho * R_safe)
    
    # Retornamos una matriz 2x2 como arreglo de NumPy
    return np.array([[J00, J01], [J10, J11]])


# =============================================================================
# 4. SIMULACIÓN
# =============================================================================
y0 = [R0_v, 0.0]                  # Radio inicial R0, velocidad inicial 0
t_span = [0, 800e-6]              # Simulamos 200 microsegundos
t_eval = np.linspace(t_span[0], t_span[1], 10000) # 5000 puntos para resolución alta

print("Ejecutando Radau con Jacobiano...")


inicio = time.perf_counter()  # Tiempo de inicio
sol = solve_ivp(
    sbsl_ode, 
    t_span, 
    y0, 
    method='Radau', 
    jac=sbsl_jacobian, 
    t_eval=t_eval,
    args=args_v,
    rtol=1e-8,
    atol=1e-10,
    first_step=1e-11  # <-- Obliga a Radau a empezar con cuidado
)
fin = time.perf_counter()  # Tiempo de fin
tiempo_ejecucion = fin - inicio


print(f"¡Éxito: {sol.success}!")
print(f"Evaluaciones de la función: {sol.nfev}")
print(f"Evaluaciones del Jacobiano: {sol.njev}")

# =============================================================================
# 5. GRÁFICA
# =============================================================================
# =============================================================================
# 5. GRÁFICA
# =============================================================================
if sol.success:
    plt.figure(figsize=(10, 6))
    
    # Convertimos el tiempo a microsegundos (1e6) y el radio a micrómetros (1e6)
    plt.plot(sol.t * 1e6, sol.y[0] * 1e6, label=r'Radio $R(t)$', color='#1f77b4')
    
    # Línea de referencia del radio inicial
    plt.axhline(R0_v * 1e6, color='gray', linestyle='--', alpha=0.7, label=r'$R_0 = 5\,\mu m$')
    
    plt.title('Dinámica de Burbuja - Ecuación de Keller-Miksis (Radau + Jacobiano)')
    plt.xlabel(r'Tiempo ($\mu s$)')
    plt.ylabel(r'Radio ($\mu m$)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    # --- LA SOLUCIÓN PARA "Agg" ESTÁ AQUÍ ---
    # En lugar de plt.show(), guardamos la figura en la misma carpeta del script
    nombre_archivo = "colapso_burbuja.png"
    plt.savefig(nombre_archivo, dpi=300)
    print(f"\n¡Listo! La gráfica se ha guardado exitosamente como: {nombre_archivo}")
    print(f"Tiempo de ejecución del solver: {tiempo_ejecucion:.4f} segundos")




else:
    print("Ocurrió un problema con el solver:", sol.message)