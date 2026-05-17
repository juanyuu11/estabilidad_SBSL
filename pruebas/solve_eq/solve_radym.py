import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

# =============================================================================
# 1. EL NÚCLEO FÍSICO (Centralizamos las matemáticas aquí)
# =============================================================================
@njit
def fisica_burbuja(t, R, V, rho, c, Pa, omega, P0, sigma, eta, R0, h, gamma):
    """Calcula y retorna [Velocidad, Aceleración] en un instante t."""
    R_safe = max(R, h + 1e-12)
    R2, R3, h3 = R_safe**2, R_safe**3, h**3
    denom = R3 - h3
    ratio = (R0**3 - h3) / denom
    
    Pg0 = P0 + (2.0 * sigma) / R0 
    Pg = Pg0 * (ratio**gamma)
    
    dPg_dR = - (3.0 * gamma * R2 / denom) * Pg
    dPg_dt = dPg_dR * V  
    
    P_drive = -Pa * np.sin(omega * t)
    F = Pg - P0 - P_drive - (4.0 * eta * V) / R_safe - (2.0 * sigma) / R_safe
    F_rad = (R_safe / c) * dPg_dt
    
    dR_dt = V
    # dV_dt es exactamente la aceleración
    dV_dt = (F + F_rad - 1.5 * rho * V**2) / (rho * R_safe)
    
    return dR_dt, dV_dt

# =============================================================================
# 2. LAS FUNCIONES AUXILIARES (Llaman al núcleo físico)
# =============================================================================
@njit
def sbsl_ode_numba(t, y, rho, c, Pa, omega, P0, sigma, eta, R0, h, gamma):
    """Adaptador para que solve_ivp pueda leer la ecuación."""
    dR_dt, dV_dt = fisica_burbuja(t, y[0], y[1], rho, c, Pa, omega, P0, sigma, eta, R0, h, gamma)
    return [dR_dt, dV_dt]

@njit
def calcular_aceleracion_array(t_arr, R_arr, V_arr, rho, c, Pa, omega, P0, sigma, eta, R0, h, gamma):
    """Recalcula la aceleración al final de la simulación."""
    N = len(t_arr)
    A_arr = np.zeros(N)
    for i in range(N):
        # Descartamos la velocidad (ya la tenemos) y guardamos solo la aceleración
        _, A_arr[i] = fisica_burbuja(t_arr[i], R_arr[i], V_arr[i], rho, c, Pa, omega, P0, sigma, eta, R0, h, gamma)
    return A_arr

# =============================================================================
# 3. LA FUNCIÓN PRINCIPAL (El envoltorio para tu barrido)
# =============================================================================
def simular_burbuja_estacionario(Pa, f=26500.0, R0=5.0e-6, ciclos_transitorios=50, ciclos_guardados=2, rtol=1e-6, atol=1e-8):
    P0, rho, c = 101325.0, 1000.0, 1500.0            
    sigma, eta, gamma = 0.0728, 0.001, 1.4           
    h = R0 / 8.5          
    omega = 2.0 * np.pi * f
    
    args_numba = (rho, c, Pa, omega, P0, sigma, eta, R0, h, gamma)
    
    periodo = 1.0 / f
    tiempo_transitorio = ciclos_transitorios * periodo
    tiempo_total = (ciclos_transitorios + ciclos_guardados) * periodo
    
    # Integramos todo el tiempo
    t_span = [0, tiempo_total]
    
    # Pero solo guardamos los puntos de la etapa final
    puntos_guardados = int(ciclos_guardados * 1000)
    t_eval = np.linspace(tiempo_transitorio, tiempo_total, puntos_guardados)
    
    sol = solve_ivp(
        sbsl_ode_numba, t_span, [R0, 0.0], method='Radau', 
        t_eval=t_eval, args=args_numba, rtol=rtol, atol=atol, first_step=1e-11
    )
    
    if sol.success:
        t_res, R_res, V_res = sol.t, sol.y[0], sol.y[1]
        A_res = calcular_aceleracion_array(t_res, R_res, V_res, *args_numba)
        return t_res, R_res, V_res, A_res, True
    else:
        return None, None, None, None, False