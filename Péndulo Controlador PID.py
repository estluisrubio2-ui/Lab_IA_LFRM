import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import odeint

# Parámetros físicos
m = 0.1     # masa del péndulo
M = 1.0     # masa del carro
L = 0.5     # longitud del péndulo
g = 9.81    # gravedad
b = 0.1     # fricción

# Condiciones iniciales: posición x, velocidad x, ángulo θ, velocidad angular
y0 = [0.0, 0.0, np.pi + 1, 0.0]

  # péndulo ligeramente perturbado desde la vertical

dt = 0.02
t_max = 20
t = np.arange(0, t_max, dt)

# Controlador PID (solo para θ)
Kp = -150
Ki = -5
Kd = -20

integral_error = 0
prev_error = 0

# Ecuaciones del sistema
def deriv(y, t, F):
    x, x_dot, theta, theta_dot = y
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    denom = M + m * sin_theta**2

    theta_ddot = (g * sin_theta - cos_theta * (F + m * L * theta_dot**2 * sin_theta - b * x_dot) / denom) / \
                 (L * (4/3 - (m * cos_theta**2) / denom))

    x_ddot = (F + m * L * (theta_dot**2 * sin_theta - theta_ddot * cos_theta) - b * x_dot) / denom

    return [x_dot, x_ddot, theta_dot, theta_ddot]

# Variables del sistema
state = y0

# Configurar la animación
fig, ax = plt.subplots()
def update_axes(x):
    ax.set_xlim(x - 2.5, x + 2.5)

ax.set_ylim(-1.5, 1.5)

cart_line, = ax.plot([], [], 'k', lw=4)
pendulum_line, = ax.plot([], [], 'o-', lw=2, color='blue')

def animate(i):
    global state, integral_error, prev_error

    # Extraer el estado actual
    _, _, theta, _ = state

    # Calcular error respecto a posición vertical
    error = theta - np.pi
    integral_error += error * dt
    derivative = (error - prev_error) / dt
    prev_error = error

    # Control PID
    F = Kp * error + Ki * integral_error + Kd * derivative

    # Integrar el sistema con la fuerza calculada
    state_new = odeint(deriv, state, [0, dt], args=(F,))
    state = state_new[-1]

    # Actualizar visualización
    x, _, theta, _ = state
    px = x + L * np.sin(theta)
    py = -L * np.cos(theta)
    update_axes(x)

    cart_line.set_data([x - 0.2, x + 0.2], [0, 0])
    pendulum_line.set_data([x, px], [0, py])

    return cart_line, pendulum_line

# Ejecutar animación
ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=dt*1000, blit=True)
plt.title("Control automático del péndulo invertido")
plt.show()
