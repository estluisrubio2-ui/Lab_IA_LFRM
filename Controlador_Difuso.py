import pygame
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ---------------- Parámetros físicos ----------------
g = 9.81    # gravedad (m/s^2)
M = 1.0     # masa del carro (kg)
m = 2.0     # masa del péndulo (kg)
L = 0.5     # longitud del péndulo (m)
dt = 0.02   # paso de integración (s)
b = 20      # fricción del carro

# Estado inicial [x, x_dot, theta, theta_dot]
x = 0.0
x_dot = 0.0
theta = np.deg2rad(10)  # inclinación inicial
theta_dot = 0.0

# Saturación de fuerza (N)
F_max = 10.0

# ---------------- Controlador Difuso ----------------
# Variables lingüísticas
theta_var = ctrl.Antecedent(np.linspace(-1, 1, 200), 'theta')
theta_dot_var = ctrl.Antecedent(np.linspace(-3, 3, 200), 'theta_dot')
force_var = ctrl.Consequent(np.linspace(-F_max, F_max, 200), 'force')

# Funciones de membresía (cubrimos todo el rango)
theta_var['neg'] = fuzz.trimf(theta_var.universe, [-1, -0.5, 0])
theta_var['zero'] = fuzz.trimf(theta_var.universe, [-0.1, 0, 0.1])
theta_var['pos'] = fuzz.trimf(theta_var.universe, [0, 0.5, 1])

theta_dot_var['neg'] = fuzz.trimf(theta_dot_var.universe, [-3, -1.5, 0])
theta_dot_var['zero'] = fuzz.trimf(theta_dot_var.universe, [-0.5, 0, 0.5])
theta_dot_var['pos'] = fuzz.trimf(theta_dot_var.universe, [0, 1.5, 3])

force_var['left'] = fuzz.trimf(force_var.universe, [-F_max, -F_max/2, 0])
force_var['zero'] = fuzz.trimf(force_var.universe, [-1, 0, 1])
force_var['right'] = fuzz.trimf(force_var.universe, [0, F_max/2, F_max])

# Reglas difusas
rules = [
    ctrl.Rule(theta_var['neg'] & theta_dot_var['zero'], force_var['left']),
    ctrl.Rule(theta_var['pos'] & theta_dot_var['zero'], force_var['right']),
    ctrl.Rule(theta_var['zero'] & theta_dot_var['neg'], force_var['right']),
    ctrl.Rule(theta_var['zero'] & theta_dot_var['pos'], force_var['left']),
    ctrl.Rule(theta_var['neg'] & theta_dot_var['neg'], force_var['left']),
    ctrl.Rule(theta_var['pos'] & theta_dot_var['pos'], force_var['right']),
    ctrl.Rule(theta_var['zero'] & theta_dot_var['zero'], force_var['zero'])
]

# Construir el sistema de control difuso
system = ctrl.ControlSystem(rules)
fuzzy_controller = ctrl.ControlSystemSimulation(system)

# ---------------- Pygame setup ----------------
pygame.init()
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Péndulo Invertido con Carro (Control Difuso)")
clock = pygame.time.Clock()

origin_y = HEIGHT // 2 + 100  # altura del riel
scale = 200  # escala: 1m = 200px

running = True

# ---------------- Simulación ----------------
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Controlador Difuso ---
    fuzzy_controller.input['theta'] = theta
    fuzzy_controller.input['theta_dot'] = theta_dot

    try:
        fuzzy_controller.compute()
        F = fuzzy_controller.output['force']
    except:
        F = 0  # si no hay salida, ponemos fuerza neutra

    # --- Dinámica del carro-péndulo ---
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    denom = M + m * sin_theta**2

    x_ddot = (F - b*x_dot + m * sin_theta * (L * theta_dot**2 + g * cos_theta)) / denom
    theta_ddot = (-F * cos_theta - m * L * theta_dot**2 * cos_theta * sin_theta -
                  (M + m) * g * sin_theta + b * x_dot * cos_theta) / (L * denom)

    # Integración
    x_dot += x_ddot * dt
    x += x_dot * dt
    theta_dot += theta_ddot * dt
    theta += theta_dot * dt

    # ---------------- Dibujar ----------------
    screen.fill((255, 255, 255))

    # Cámara sigue al carro → offset en X
    offset_x = WIDTH//2 - int(x*scale)

    # Dibujar eje X (marcas cada 0.5 m)
    for k in range(-20, 21):
        pos_x = int(k*0.5*scale) + offset_x
        if 0 <= pos_x <= WIDTH:
            pygame.draw.line(screen, (200, 200, 200), (pos_x, 0), (pos_x, HEIGHT), 1)
            font = pygame.font.SysFont(None, 20)
            text = font.render(f"{k*0.5:.1f}", True, (100, 100, 100))
            screen.blit(text, (pos_x-10, origin_y+40))

    # Posición del carro en pantalla
    cart_x = int(x*scale) + offset_x
    cart_y = origin_y

    # Carro
    pygame.draw.rect(screen, (0, 0, 0), (cart_x-40, cart_y-20, 80, 40))

    # Péndulo
    pend_x = cart_x + int(L*scale * np.sin(theta))
    pend_y = cart_y - int(L*scale * np.cos(theta))
    pygame.draw.line(screen, (0, 0, 255), (cart_x, cart_y), (pend_x, pend_y), 4)
    pygame.draw.circle(screen, (255, 0, 0), (pend_x, pend_y), 12)

    # Piso
    pygame.draw.line(screen, (100, 100, 100), (0, cart_y+20), (WIDTH, cart_y+20), 2)

    pygame.display.flip()
    clock.tick(1/dt)

pygame.quit()
