import pygame
import numpy as np

# ---------------- Parámetros físicos ----------------
g = 9.81    # gravedad (m/s^2)
M = 1.0      # masa del carro (kg)
m = 2.0      # masa del péndulo (kg)
L = 0.5      # longitud del péndulo (m)
dt = 0.02    # paso de integración (s)
b = 20     # fricción del carro

# Controlador PID sobre el ángulo
Kp = 20.0
Ki = 0.5
Kd = 15.0

# Estado inicial [x, x_dot, theta, theta_dot]
x = 0.0
x_dot = 0.0
theta = np.deg2rad(5)  # inclinación inicial pequeña
theta_dot = 0.0

# PID variables
integral_error = 0.0
prev_error = 0.0

# Saturación de fuerza (N)
F_max = 5.0

# ---------------- Pygame setup ----------------
pygame.init()
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Péndulo Invertido con Carro (PID con cámara)")
clock = pygame.time.Clock()

origin_y = HEIGHT // 2 + 100  # altura del riel
scale = 200  # escala: 1m = 200px

running = True

# ---------------- Simulación ----------------
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- PID controller ---
    error = theta  # queremos que theta = 0
    integral_error += error * dt
    derivative_error = (error - prev_error) / dt
    prev_error = error
    F = -(Kp * error + Ki * integral_error + Kd * derivative_error)

    # Aplicar saturación
    F = max(min(F, F_max), -F_max)

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
