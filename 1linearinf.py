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
    x_ddot = -(g + l * w ** 2) * (theta - np.pi) - 2 * l * delta * theta_dot
    theta_ddot = x_ddot / l + g / l * (theta - np.pi)
    return [x_dot, x_ddot, theta_dot, theta_ddot]


# Параметры моделирования
initial_angle_offset = 0.5  # Начальное отклонение от вертикали
t_span = (0, 15)  # Время моделирования
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

# Создаем фигуру с новым дизайном
fig = plt.figure(figsize=(14, 12))
gs = fig.add_gridspec(3, 2, width_ratios=[3, 1], height_ratios=[2, 1, 1], hspace=0.3, wspace=0.25)
ax1 = fig.add_subplot(gs[0, :])  # Анимация
ax2 = fig.add_subplot(gs[1, 0])  # Угол
ax3 = fig.add_subplot(gs[2, 0])  # Положение
ax4 = fig.add_subplot(gs[1:, 1])  # Угловая скорость

# Стилизация анимации
cart_width, cart_height = 0.4, 0.2
cart = Rectangle((0, 0), cart_width, cart_height,
                 fc='#4682B4', ec='#2F4F4F', alpha=0.9, lw=1.5)
pendulum, = ax1.plot([], [], color='#FF4500', lw=4, alpha=0.8)
bob, = ax1.plot([], [], 'o', color='#8B0000', markersize=18, alpha=0.8)
time_text = ax1.text(0.72, 0.15, '', transform=ax1.transAxes,
                     bbox=dict(facecolor='#F0F8FF', alpha=0.9,
                               edgecolor='#696969', boxstyle='round'))
ax1.add_patch(cart)

# Общие настройки графиков
for ax in [ax1, ax2, ax3, ax4]:
    ax.grid(True, color='#D3D3D3', ls='--', alpha=0.7)
    ax.set_facecolor('#F5F5F5')
    ax.tick_params(axis='both', which='major', labelsize=9)

# Настройка анимации
ax1.set_aspect('equal')
ax1.set_xlim(-2, 4)
ax1.set_ylim(-1.5, 1.5)
ax1.set_title(f'Стабилизация перевернутого маятника\nНачальный угол: {initial_angle:.2f} рад',
              fontsize=12, pad=15)

# График угла
ax2.plot(solution.t, solution.y[2], '#1E90FF', alpha=0.8, lw=2)
ax2.axhline(np.pi, color='#2E8B57', ls=':', lw=2, label='Цель: θ=π')
ax2.set_ylabel('Угол, θ (рад)', fontsize=10)
ax2.set_ylim(2.5, 3.5)
ax2.legend(fontsize=9, loc='lower right')

# График положения
ax3.plot(solution.t, solution.y[0], '#FF8C00', alpha=0.8, lw=2)
ax3.set_ylabel('Положение, x (м)', fontsize=10)
ax3.set_xlabel('Время, t (с)', fontsize=10)

# График угловой скорости
ax4.plot(solution.t, solution.y[3], '#9370DB', alpha=0.8, lw=2)
ax4.axhline(0, color='#696969', ls=':', lw=1.5)
ax4.set_ylabel('Угл. скорость, θ\' (рад/с)', fontsize=10)
ax4.set_xlabel('Время, t (с)', fontsize=10)


def init():
    cart.set_xy((-cart_width / 2, -cart_height / 2))
    pendulum.set_data([], [])
    bob.set_data([], [])
    time_text.set_text('')
    return cart, pendulum, bob, time_text


def animate(i):
    x = solution.y[0, i]
    theta = solution.y[2, i]
    theta_dot = solution.y[3, i]

    # Обновление анимации
    cart.set_xy((x - cart_width / 2, -cart_height / 2))
    pendulum.set_data([x, x + l * np.sin(theta)], [0, -l * np.cos(theta)])
    bob.set_data([x + l * np.sin(theta)], [-l * np.cos(theta)])

    # Динамическое масштабирование
    ax1.set_xlim(x - 3, x + 3)

    # Обновление текста
    time_text.set_text(
        f'Время: {solution.t[i]:.2f} с\n'
        f'Угол: {theta:.4f} рад\n'
        f'Отклонение: {theta - np.pi:.2e} рад\n'
        f'Скорость: {solution.y[1, i]:.2e} м/с\n'
        f'Угл. скорость: {theta_dot:.2e} рад/с'
    )
    return cart, pendulum, bob, time_text


# Создание анимации
ani = animation.FuncAnimation(
    fig, animate, frames=len(solution.t),
    init_func=init, blit=False, interval=25, repeat=False
)

plt.tight_layout()
plt.show()