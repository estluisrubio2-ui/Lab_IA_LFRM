import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import odeint

# Parámetros físicos
m = 0.1
M = 1.0
L = 0.5
g = 9.81
b = 0.1

# Condiciones iniciales
y0 = [0.0, 0.0, np.pi + 0.0, 0.0]

dt = 0.02
t_max = 20
t = np.arange(0, t_max, dt)

# Fuerza externa inicial
F_ext = 0.0

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

# Variables para guardar el estado actual
state = y0

# Listas para datos de animación
X = []
THETA = []

# Crear figura y ejes
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(-1.5, 1.5)

cart_line, = ax.plot([], [], 'k', lw=4)
pendulum_line, = ax.plot([], [], 'o-', lw=2, color='blue')

# Variable para controlar la fuerza externa
force = 0.0

# Función para actualizar la animación
def animate(i):
    global state

    # Integrar usando la fuerza actual
    state = odeint(deriv, state, [0, dt], args=(force,))[-1]

    x, _, theta, _ = state
    px = x + L * np.sin(theta)
    py = -L * np.cos(theta)

    cart_line.set_data([x - 0.2, x + 0.2], [0, 0])
    pendulum_line.set_data([x, px], [0, py])

    return cart_line, pendulum_line

# Función para detectar teclas y cambiar la fuerza
def on_key(event):
    global force
    if event.key == 'left':
        force = -2.0
    elif event.key == 'right':
        force = 2.0
    else:
        force = 0.0

# Conectar la función de teclado
fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('key_release_event', lambda event: set_force_zero())

def set_force_zero():
    global force
    force = 0.0

# Ejecutar animación
ani = animation.FuncAnimation(fig, animate, frames=int(t_max/dt), interval=dt*1000, blit=True)
plt.title("Péndulo invertido")
plt.show()
