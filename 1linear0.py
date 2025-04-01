import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

# Параметры
l = 0.5  # длина маятника (м)
g = 9.81  # ускорение свободного падения (м/с²)


def inverted_pendulum(t, state):
    x, x_dot, theta, theta_dot = state

    # Вычисляем управляющее воздействие
    u = -l * (l / g + g / l + 6) * (theta - np.pi) - l * (4 * l / g + 4) * theta_dot + (l / g) * x + (4 * l / g) * x_dot

    x_ddot = u  # Управление напрямую задает ускорение тележки
    theta_ddot = x_ddot / l + g / l * (theta - np.pi)

    return [x_dot, x_ddot, theta_dot, theta_ddot], u  # Возвращаем и состояние, и управление


# Параметры моделирования
initial_angle_offset = 0.5  # Начальное отклонение от вертикали
t_span = (0, 15)  # Время моделирования
initial_angle = np.pi - initial_angle_offset
initial_state = [0, 0, initial_angle, 0]
t_eval = np.linspace(t_span[0], t_span[1], 2000)

# Решение системы с высокой точностью и сбор управления
u_history = np.zeros_like(t_eval)
states = np.zeros((4, len(t_eval)))
states[:, 0] = initial_state

for i in range(1, len(t_eval)):
    t = t_eval[i - 1]
    dt = t_eval[i] - t_eval[i - 1]

    # Решаем на одном шаге
    sol, u = inverted_pendulum(t, states[:, i - 1])
    states[:, i] = states[:, i - 1] + np.array(sol) * dt
    u_history[i] = u

solution = type('', (), {})()  # Создаем пустой объект для совместимости
solution.t = t_eval
solution.y = states
solution.u = u_history  # Добавляем историю управления

# Критерий стабилизации
stable_threshold = 1e-6
stable_index = len(solution.t) - 1
for i in range(len(solution.t)):
    if (abs(solution.y[2, i] - np.pi) < stable_threshold and
            abs(solution.y[3, i]) < stable_threshold and
            abs(solution.y[1, i]) < stable_threshold):
        stable_index = i
        break

# Обрезаем решение до момента стабилизации
solution.t = solution.t[:stable_index + 1]
solution.y = solution.y[:, :stable_index + 1]
solution.u = solution.u[:stable_index + 1]

# Создаем фигуру с новым расположением графиков
fig = plt.figure(figsize=(12, 14))
gs = fig.add_gridspec(4, 2, height_ratios=[2, 1, 1, 1], width_ratios=[3, 1])
ax1 = fig.add_subplot(gs[0, :])  # Анимация (занимает всю верхнюю строку)
ax2 = fig.add_subplot(gs[1, 0])  # График угла
ax3 = fig.add_subplot(gs[2, 0])  # График положения
ax4 = fig.add_subplot(gs[1, 1])  # График угловой скорости

ax5 = fig.add_subplot(gs[3, :])  # График управления (занимает всю нижнюю строку)

# Настройка анимации с новыми цветами
cart_width, cart_height = 0.4, 0.2
cart = Rectangle((0, 0), cart_width, cart_height, fc='#4682B4', ec='#2F4F4F', alpha=0.8)
pendulum, = ax1.plot([], [], color='#FF6347', linestyle='-', linewidth=3, alpha=0.8)
bob, = ax1.plot([], [], 'o', color='#8B0000', markersize=12, alpha=0.8)
time_text = ax1.text(0.65, 0.1, '', transform=ax1.transAxes,
                     bbox=dict(facecolor='white', alpha=0.6))
ax1.add_patch(cart)

ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-1, 3)
ax1.set_ylim(-1, 1)
ax1.set_title(f'Стабилизация перевернутого маятника (начальный угол: {initial_angle:.2f} рад)', pad=20)

# Настройка графиков с новыми цветами и прозрачностью
ax2.grid(True, alpha=0.3)
ax2.set_ylabel('Угол, θ (рад)')
ax2.set_title('Эволюция угла отклонения', pad=10)
ax2.axhline(np.pi, color='#2E8B57', linestyle='--', label='θ=π (цель)', alpha=0.7)
ax2.plot(solution.t, solution.y[2], color='#1E90FF', alpha=0.8)
ax2.legend()
ax2.set_ylim(2.5, 3.5)

ax3.grid(True, alpha=0.3)
ax3.set_ylabel('Положение тележки, x (м)')
ax3.set_title('Движение тележки', pad=10)
ax3.plot(solution.t, solution.y[0], color='#FF8C00', alpha=0.8)

ax4.grid(True, alpha=0.3)
ax4.set_ylabel('Угловая скорость, θ\' (рад/с)')
ax4.set_title('Эволюция угловой скорости', pad=10)
ax4.axhline(0, color='#2E8B57', linestyle='--', label='Цель: θ\'=0', alpha=0.7)
ax4.plot(solution.t, solution.y[3], color='#9370DB', alpha=0.8)
ax4.legend()

ax5.grid(True, alpha=0.3)
ax5.set_xlabel('Время, t (с)')
ax5.set_ylabel('u (м/с²)')
ax5.set_title('Управляющее воздействие', pad=10)
ax5.axhline(0, color='#2E8B57', linestyle='--', alpha=0.3)
ax5.plot(solution.t, solution.u, color='#8A2BE2', alpha=0.8)


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
    u = solution.u[i]

    # Анимация
    cart.set_xy((x - cart_width / 2, -cart_height / 2))
    pendulum.set_data([x, x + l * np.sin(theta)], [0, -l * np.cos(theta)])
    bob.set_data([x + l * np.sin(theta)], [-l * np.cos(theta)])

    # Динамическое масштабирование
    ax1.set_xlim(x - 2, x + 2)

    # Информация
    time_text.set_text(
        f'Время: {solution.t[i]:.2f} с\n'
        f'Угол: {theta:.6f} рад\n'
        f'От π: {theta - np.pi:.2e} рад\n'
        f'Угл. скорость: {theta_dot:.2e} рад/с\n'
        f'Позиция: {x:.4f} м\n'
        f'Скорость: {solution.y[1, i]:.2e} м/с\n'
        f'Управление: {u:.2e} м/с²'
    )

    # Обновляем график управления (красная точка на текущем моменте)
    if hasattr(animate, 'u_point'):
        animate.u_point.remove()
    animate.u_point = ax5.plot(solution.t[i], u, 'ro', markersize=4)[0]

    return cart, pendulum, bob, time_text, animate.u_point


# Создаем анимацию
ani = animation.FuncAnimation(
    fig, animate, frames=len(solution.t),
    init_func=init, blit=False, interval=20, repeat=False
)
plt.tight_layout()
plt.show()