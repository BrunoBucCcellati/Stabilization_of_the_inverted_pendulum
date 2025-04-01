import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

# Настройка бэкенда для анимации
plt.switch_backend('TkAgg')

# Параметры системы
g = 9.81
a1 = -35 - 4 * g
b1 = -10
a2 = 35 + 3 * g + 12 / g
b2 = 25 / g + 10
l1 = l2 = 1.0

# Цветовая палитра
colors = {
    'cart': '#4682B4',  # Стальной синий
    'cart_edge': '#2F4F4F',  # Темно-грифельный
    'link1': '#FF6347',  # Томатный
    'link2': '#1E90FF',  # Ярко-синий
    'joint1': '#8B0000',  # Темно-красный
    'joint2': '#00008B',  # Темно-синий
    'angle_ref': '#2E8B57',  # Морская зелень
    'plot_phi1': '#1E90FF',
    'plot_phi2': '#FF8C00',
    'plot_position': '#2E8B57',
    'plot_velocity1': '#9370DB',
    'plot_velocity2': '#FF4500',
    'plot_velocity3': '#20B2AA',
    'text_bg': '#F8F8FF'
}


def double_pendulum_cart(t, state):
    x, x_dot, φ1, φ1_dot, φ2, φ2_dot = state
    x_ddot = a1 * (φ1 - np.pi) + b1 * φ1_dot + a2 * (φ2 - np.pi) + b2 * φ2_dot
    φ2_ddot = -2 * g * (φ1 - np.pi) + 2 * g * (φ2 - np.pi)
    φ1_ddot = x_ddot + g * (φ2 - np.pi) - φ2_ddot
    return [x_dot, x_ddot, φ1_dot, φ1_ddot, φ2_dot, φ2_ddot]


# Начальные условия
offset1 = 0.1
offset2 = 0.1
initial_state = [0.0, 0.0, np.pi - offset1, 0.0, np.pi - offset2, 0.0]

# Время моделирования
t_span = (0.0, 20.0)
t_eval = np.linspace(t_span[0], t_span[1], 2000)

# Решение системы
solution = solve_ivp(double_pendulum_cart, t_span, initial_state,
                     t_eval=t_eval, method='RK45',
                     rtol=1e-8,
                     atol=1e-10)

if not solution.success:
    raise RuntimeError(f"Решение не найдено: {solution.message}")

print(f"Количество временных точек: {len(solution.t)}")
print(f"Временной диапазон: {solution.t[0]:.2f} - {solution.t[-1]:.2f} с")

# Создание фигуры с настроенными отступами
fig = plt.figure(figsize=(12, 12))
plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08, hspace=0.4)
gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1])
ax1 = fig.add_subplot(gs[0])  # Анимация
ax2 = fig.add_subplot(gs[1])  # График углов
ax3 = fig.add_subplot(gs[2])  # График положения
ax4 = fig.add_subplot(gs[3])  # График скоростей

# Настройка анимации
cart_width, cart_height = 0.8, 0.4
cart = Rectangle((-cart_width / 2, -cart_height / 2), cart_width, cart_height,
                 fc=colors['cart'], ec=colors['cart_edge'],
                 alpha=0.9, lw=1.5, zorder=0)
link1, = ax1.plot([], [], color=colors['link1'], linestyle='-',
                  lw=3, alpha=0.85, zorder=5)
link2, = ax1.plot([], [], color=colors['link2'], linestyle='-',
                  lw=3, alpha=0.85, zorder=5)
joint1, = ax1.plot([], [], 'o', color=colors['joint1'],
                   markersize=10, alpha=0.9, markeredgecolor='#4A0000', zorder=10)
joint2, = ax1.plot([], [], 'o', color=colors['joint2'],
                   markersize=10, alpha=0.9, markeredgecolor='#000060', zorder=10)

# Информационное окно в правом верхнем углу
time_text = ax1.text(0.98, 0.95, '', transform=ax1.transAxes,
                     bbox=dict(facecolor=colors['text_bg'], alpha=0.85,
                               edgecolor='#696969', boxstyle='round'),
                     verticalalignment='top', horizontalalignment='right')
ax1.add_patch(cart)

ax1.set_aspect('equal')
ax1.grid(True, color='#D3D3D3', linestyle='--', alpha=0.3)
ax1.set_xlim(-5, 5)
ax1.set_ylim(-3, 3)
ax1.set_title(
    f'Двухзвенный маятник на тележке\nНачальные углы: φ1 = {(np.pi - offset1):.2f}, φ2 = {(np.pi - offset2):.2f} рад',
    fontsize=12, pad=15)

# Настройка графиков
for ax in [ax2, ax3, ax4]:
    ax.grid(True, color='#D3D3D3', linestyle='--', alpha=0.3)
    ax.set_facecolor(colors['text_bg'])
    ax.set_xlim(0, solution.t[-1])

# График углов
ax2.set_ylabel('Углы (рад)', fontsize=10)
ax2.axhline(np.pi, color=colors['angle_ref'], linestyle=':',
            lw=2, alpha=0.7, label='Вертикаль (π)')
line_phi1, = ax2.plot([], [], color=colors['plot_phi1'],
                      alpha=0.85, lw=2, label='φ₁ (нижний)')
line_phi2, = ax2.plot([], [], color=colors['plot_phi2'],
                      alpha=0.85, lw=2, label='φ₂ (верхний)')
ax2.legend(fontsize=9)

# График положения
ax3.set_ylabel('Положение тележки (м)', fontsize=10)
line_position, = ax3.plot([], [], color=colors['plot_position'],
                          alpha=0.85, lw=2, label='x(t)')
ax3.legend(fontsize=9)

# График скоростей
ax4.set_xlabel('Время (с)', fontsize=10)
ax4.set_ylabel('Скорости', fontsize=10)
line_xdot, = ax4.plot([], [], color=colors['plot_velocity1'],
                      alpha=0.85, lw=2, label='ẋ(t)')
line_phi1dot, = ax4.plot([], [], color=colors['plot_velocity2'],
                         linestyle='--', alpha=0.85, lw=2, label='φ̇₁(t)')
line_phi2dot, = ax4.plot([], [], color=colors['plot_velocity3'],
                         linestyle='-.', alpha=0.85, lw=2, label='φ̇₂(t)')
ax4.legend(fontsize=9)


def init():
    cart.set_xy((-cart_width / 2, -cart_height / 2))
    link1.set_data([], [])
    link2.set_data([], [])
    joint1.set_data([], [])
    joint2.set_data([], [])
    time_text.set_text('')

    for line in [line_phi1, line_phi2, line_position, line_xdot, line_phi1dot, line_phi2dot]:
        line.set_data([], [])

    for ax in [ax2, ax3, ax4]:
        ax.relim()
        ax.autoscale_view()

    return (cart, link1, link2, joint1, joint2, time_text,
            line_phi1, line_phi2, line_position,
            line_xdot, line_phi1dot, line_phi2dot)


def animate(i):
    x = solution.y[0, i]
    phi1 = solution.y[2, i]
    phi2 = solution.y[4, i]

    # Координаты звеньев
    x0, y0 = x, 0.0
    x1 = x0 + l1 * np.sin(phi1)
    y1 = y0 - l1 * np.cos(phi1)
    x2 = x1 + l2 * np.sin(phi2)
    y2 = y1 - l2 * np.cos(phi2)

    # Динамическое масштабирование анимации
    ax1.set_xlim(x - 5, x + 5)

    # Обновление анимации
    cart.set_xy((x - cart_width / 2, -cart_height / 2))
    link1.set_data([x0, x1], [y0, y1])
    link2.set_data([x1, x2], [y1, y2])
    joint1.set_data([x1], [y1])
    joint2.set_data([x2], [y2])

    # Обновление графиков
    line_phi1.set_data(solution.t[:i + 1], solution.y[2, :i + 1])
    line_phi2.set_data(solution.t[:i + 1], solution.y[4, :i + 1])
    line_position.set_data(solution.t[:i + 1], solution.y[0, :i + 1])
    line_xdot.set_data(solution.t[:i + 1], solution.y[1, :i + 1])
    line_phi1dot.set_data(solution.t[:i + 1], solution.y[3, :i + 1])
    line_phi2dot.set_data(solution.t[:i + 1], solution.y[5, :i + 1])

    # Автоматическое масштабирование графиков
    for ax, lines in zip([ax2, ax3, ax4],
                         [[line_phi1, line_phi2],
                          [line_position],
                          [line_xdot, line_phi1dot, line_phi2dot]]):
        ax.relim()
        for line in lines:
            ax.update_datalim(np.column_stack([solution.t[:i + 1], line.get_ydata()]))
        ax.autoscale_view()

    # Информация
    time_text.set_text(
        f'Время: {solution.t[i]:.2f} с\n'
        f'Позиция: {x:.3f} м\n'
        f'Скорость: {solution.y[1, i]:.2f} м/с\n'
        f'φ₁: {np.degrees(phi1):.1f}°\n'
        f'φ₂: {np.degrees(phi2):.1f}°'
    )

    return (cart, link1, link2, joint1, joint2, time_text,
            line_phi1, line_phi2, line_position,
            line_xdot, line_phi1dot, line_phi2dot)


# Создание анимации
ani = animation.FuncAnimation(
    fig, animate,
    frames=len(solution.t),
    init_func=init,
    blit=False,
    interval=20,
    repeat=False
)

plt.tight_layout()
plt.show()