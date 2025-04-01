import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

# Параметры
l = 0.5  # длина маятника (м)
g = 9.81  # ускорение свободного падения (м/с²)
w = 1.0  # собственная частота (1/c)
delta = 1.0  # коэффициент диссипации


def inverted_pendulum(t, state):
    x, x_dot, theta, theta_dot = state
    x_ddot = -(g + l * w ** 2) * (theta - np.pi) + -2 * l * delta * theta_dot
    theta_ddot = x_ddot / l * np.cos(theta - np.pi) + g / l * np.sin(theta - np.pi)
    return [x_dot, x_ddot, theta_dot, theta_ddot]


# Параметры моделирования
initial_angle_offset = 0.2
t_span = (0, 15)
initial_angle = np.pi - initial_angle_offset
initial_state = [0, 0, initial_angle, 0]
t_eval = np.linspace(t_span[0], t_span[1], 2000)

# Решение системы
solution = solve_ivp(
    inverted_pendulum,
    t_span,
    initial_state,
    t_eval=t_eval,
    method='RK45',
    rtol=1e-8,
    atol=1e-10
)

# Критерий стабилизации
stable_threshold = 1e-6
stable_index = len(solution.t) - 1
for i in range(len(solution.t)):
    if (abs(solution.y[2, i] - np.pi) < stable_threshold and
            abs(solution.y[3, i]) < stable_threshold and
            abs(solution.y[1, i]) < stable_threshold):
        stable_index = i
        break

# Обрезаем решение
solution.t = solution.t[:stable_index + 1]
solution.y = solution.y[:, :stable_index + 1]

# Создаем фигуру
plt.rcParams.update({'font.size': 10})
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1])
ax1 = fig.add_subplot(gs[0])  # Анимация
ax2 = fig.add_subplot(gs[1])  # График угла
ax3 = fig.add_subplot(gs[2])  # График положения
ax4 = fig.add_subplot(gs[3])  # График угловой скорости

# Новая цветовая схема
colors = {
    'cart': '#2A5CAA',  # Темно-синий
    'pendulum': '#FF6347',  # Томатный
    'bob': '#8B0000',  # Темно-красный
    'angle': '#1E90FF',  # Ярко-синий
    'position': '#FF8C00',  # Темно-оранжевый
    'velocity': '#9370DB',  # Средний пурпурный
    'grid': '#E0E0E0'  # Светло-серый
}

# Настройка анимации
cart_width, cart_height = 0.4, 0.2
cart = Rectangle((0, 0), cart_width, cart_height,
                 fc=colors['cart'], ec='#404040', lw=1.5, alpha=0.9)
pendulum, = ax1.plot([], [], color=colors['pendulum'],
                     linestyle='-', linewidth=3.5, alpha=0.85)
bob, = ax1.plot([], [], 'o', color=colors['bob'],
                markersize=14, alpha=0.9, markeredgecolor='#4A0000')
time_text = ax1.text(0.72, 0.12, '', transform=ax1.transAxes,
                     bbox=dict(facecolor='white', alpha=0.85,
                               edgecolor='#696969', boxstyle='round'))
ax1.add_patch(cart)

ax1.set_aspect('equal')
ax1.grid(True, color=colors['grid'], linestyle='--')
ax1.set_xlim(-1, 3)
ax1.set_ylim(-1, 1)
ax1.set_title(f'Стабилизация перевернутого маятника (начальный угол: {initial_angle:.2f} рад)',
              fontsize=12, pad=10)

# Настройка графиков
for ax in [ax2, ax3, ax4]:
    ax.grid(True, color=colors['grid'], linestyle='--')
    ax.set_facecolor('#F8F8FF')

ax2.set_ylabel('Угол, θ (рад)')
ax2.axhline(np.pi, color='#2E8B57', linestyle=':', lw=2.5, alpha=0.8)
ax2.set_ylim(2.5, 3.5)

ax3.set_ylabel('Положение, x (м)')

ax4.set_xlabel('Время, t (с)')
ax4.set_ylabel('Угл. скорость, θ\' (рад/с)')
ax4.set_title('Динамика угловой скорости')
ax4.axhline(0, color='#808080', linestyle='--', alpha=0.6)


def init():
    cart.set_xy((-cart_width / 2, -cart_height / 2))
    pendulum.set_data([], [])
    bob.set_data([], [])
    time_text.set_text('')

    ax2.clear()
    ax3.clear()
    ax4.clear()

    for ax in [ax2, ax3, ax4]:
        ax.grid(True, color=colors['grid'], linestyle='--')
        ax.set_xlim(0, solution.t[-1])
        ax.set_facecolor('#F8F8FF')

    ax2.set_ylabel('Угол, θ (рад)')
    ax2.set_ylim(2.5, 3.5)
    ax2.axhline(np.pi, color='#2E8B57', linestyle=':', lw=2.5, alpha=0.8, label='θ=π (цель)')
    ax2.legend()

    ax3.set_ylabel('Положение, x (м)')

    ax4.set_xlabel('Время, t (с)')
    ax4.set_ylabel('Угл. скорость, θ\' (рад/с)')
    ax4.axhline(0, color='#808080', linestyle='--', alpha=0.6)

    return cart, pendulum, bob, time_text


def animate(i):
    x = solution.y[0, i]
    theta = solution.y[2, i]

    cart.set_xy((x - cart_width / 2, -cart_height / 2))
    pendulum.set_data([x, x + l * np.sin(theta)], [0, -l * np.cos(theta)])
    bob.set_data([x + l * np.sin(theta)], [-l * np.cos(theta)])

    ax2.plot(solution.t[:i + 1], solution.y[2, :i + 1],
             color=colors['angle'], alpha=0.85)
    ax3.plot(solution.t[:i + 1], solution.y[0, :i + 1],
             color=colors['position'], alpha=0.85)
    ax4.plot(solution.t[:i + 1], solution.y[3, :i + 1],
             color=colors['velocity'], alpha=0.85)

    ax1.set_xlim(x - 2, x + 2)

    time_text.set_text(
        f'Время: {solution.t[i]:.2f} с\n'
        f'Угол: {theta:.6f} рад\n'
        f'От π: {theta - np.pi:.2e} рад\n'
        f'Угл. скорость: {solution.y[3, i]:.2e} рад/с\n'
        f'Позиция: {x:.4f} м\n'
        f'Скорость: {solution.y[1, i]:.2e} м/с'
    )

    return cart, pendulum, bob, time_text


# Создаем анимацию
ani = animation.FuncAnimation(
    fig, animate, frames=len(solution.t),
    init_func=init, blit=False, interval=20, repeat=False
)
plt.tight_layout()
plt.show()