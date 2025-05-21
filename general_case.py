import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import iOpt

# Параметры системы
m = 1  # Масса груза(кг) - фиксирована
l = 1  # Длина звена(м) - фиксирована
betta = 0.5  # Коэффициент вязкого трения(Н * м * с) - фиксирован
g = 9.81  # Сonst - ускорение свободного падения(м/с²)

# Аналог интерфейса AnyLogic
"""
Варианты стабилизации:
    - стабилизация при малом начальном угле отклонения / при угле, который нельзя считать малым,
    - стабилизация верхнего положения / стабилизация верхнего положения и точки опоры,
    - автоматический режим управления(заданные н.у. + автоподбор параметров a, b, c, d) / подбор параметров управления пользователем
    - стабилизация однохвенного/двухзвенного матяника
"""
angle = "small_angle"
# angle = "any_angle"
stabilization = "only_upper_position"
# stabilization = "upper_position_and_stand"
mode = "auto"
# mode = "user_input"
# number_of_links = 1  # Число звеньев
number_of_links = 2

# Начальные условия
initial_angle_offset = 5  # Начальное отклонение от вертикали(в градусах)
if angle == "small_angle" and initial_angle_offset > 5:
    raise Exception(
        "В режиме стабилизации с малым начальным углом - начальный угол не должен превышать пяти градусов, введенное значение: "
        "{}.".format(initial_angle_offset))
if initial_angle_offset >= 90 or initial_angle_offset <= -90:
    raise Exception(
        "Задайте начальный угол отклонения от вертикали в градусах в диапазоне (-90, 90), введенное значение: "
        "{}.".format(initial_angle_offset))

initial_angle = np.pi - np.radians(initial_angle_offset)
initial_angular_velocity = 0  # Начальная угловая скорость(рад./с)

initial_u0 = 1  # Начальное положение опоры(м)
if initial_u0 < 0:
    raise Exception(
        "Задайте начальное положение опоры в метрах (u0 > 0), введенное значение: "
        "{}.".format(initial_u0))

initial_u0_speed = 0  # Начальная скорость опоры(м/с)

initial_angle2_offset = 5  # Начальное отклонение от вертикали для второго звена(в градусах)
if angle == "small_angle" and initial_angle2_offset > 5:
    raise Exception(
        "В режиме стабилизации с малым начальным углом - начальный угол не должен превышать пяти градусов, введенное значение: "
        "{}.".format(initial_angle2_offset))
if initial_angle2_offset >= 90 or initial_angle2_offset <= -90:
    raise Exception(
        "Задайте начальный угол отклонения от вертикали в градусах в диапазоне (-90, 90), введенное значение: "
        "{}.".format(initial_angle2_offset))

initial_angle2 = np.pi - np.radians(initial_angle2_offset)
initial_angular_velocity2 = 1  # Начальная угловая скорость для второго звена(рад./с)

# Параметры моделирования
t_span = (0, 15)  # Диапазон времени стабилизации(с)
number_of_partitions = 1000  # Число разбиений временного интервала
t_eval = np.linspace(t_span[0], t_span[1],
                     number_of_partitions)  # Время моделирования
initial_state = []
if number_of_links == 1:
    initial_state = [
        initial_u0, initial_u0_speed, initial_angle, initial_angular_velocity
    ]  # Состояние динамической системы в четырёхмерном фазовом пространстве в начальный момент времени
if number_of_links == 2:
    initial_state = [
        initial_u0, initial_u0_speed, initial_angle, initial_angular_velocity,
        initial_angle2, initial_angular_velocity2
    ]  # Состояние динамической системы в шестимерном фазовом пространстве в начальный момент времени

# Задайте шаг приращения параметров при автоматическом управлении
h = 1

# Или задайте параметры управления в ручном режиме
a = 2 * g
b = betta / (m * l)
с = l / g
d = 4 * l / g

# Для двузвенного маятника также задайте
a1 = g
b1 = l

if number_of_links == 1:
    if angle == "any_angle":
        a -= g
    if stabilization == "only_upper_position":
        if mode == "auto":
            if h <= 0:
                raise Exception(
                    "Задайте положительный шаг приращения параметров при автоматическом управлении, введенное значение: {}."
                    .format(h))
            while a <= g or b <= 0:
                a += h
                b += h
        if mode == "user_input":
            if a <= g or b <= 0:
                raise Exception(
                    "Задайте параметры, при которых выполняется условие асимптотической устойчивости, текущие значения параметров: a = {}, b = {}."
                    .format(a, b))
    if stabilization == "upper_position_and_stand":
        if mode == "auto":

            def objective(trial):
                c = trial.suggest_float("c", l / g, 2 * l / g)
                d = trial.suggest_float("d", 4 * l / g, 8 * l / g)
                b = trial.suggest_float("b", betta / (m * l),
                                        2 * betta / (m * l))
                return c * b / d + g * (1 + d * l / (b - d * l))

            study = iOpt.create_study()
            study.optimize(objective=objective,
                           solver_parameters=iOpt.SolverParameters(
                               r=3,
                               eps=0.01,
                               iters_limit=100,
                               refine_solution=True))
            c = study.Best_float_params()[0]
            d = study.Best_float_params()[1]
            b = study.Best_float_params()[2]
            a = study.Best_values()
            if b <= d / l:
                b = d / l + h
            if a <= c * b / d + g * (1 + d * l / (b - d * l)):
                a = c * b / d + g * (1 + d * l / (b - d * l)) + h
        if mode == "user_input":
            if c <= 0 or d <= 0 or b <= d / l or a <= c * b / d + g * (
                    1 + d * l / (b - d * l)):
                raise Exception(
                    "Задайте параметры, при которых выполняется условие асимптотической устойчивости, текущие значения параметров: a = {}, b = {}, c = {}, d = {}."
                    .format(a, b, c, d))

if number_of_links == 2:
    if stabilization == "only_upper_position":
        if angle == "small_angle":
            if mode == "auto":
                b1 = 10
                b = -25 / g - 10
                a1 = 35 + 4 * g
                a = -35 - 3 * g - 12 / g
            if mode == "user_input":
                if a + b <= 3 * l or a1 <= a + b + 2 * l:
                    raise Exception(
                        "Задайте параметры, при которых выполняется условие асимптотической устойчивости, текущие значения параметров: a = {}, b = {}, a1 = {}."
                        .format(a, b, a1))
        if angle == "any_angle":
            a -= g
            if mode == "auto":

                def objective(trial):
                    a = trial.suggest_float("c", g / l, 2 * g / l)
                    a1 = trial.suggest_float("c", 2 * g / l, 4 * g / l)
                    b = trial.suggest_float("b",
                                            l * np.sqrt(2) / 2, l * np.sqrt(2))
                    b1 = trial.suggest_float("b", l / 2, l)
                    return a1 / b + a / b1

                study = iOpt.create_study()
                study.optimize(objective=objective,
                               solver_parameters=iOpt.SolverParameters(
                                   r=5,
                                   eps=0.001,
                                   iters_limit=1000,
                                   refine_solution=True))
                a = study.Best_float_params()[0]
                a1 = study.Best_float_params()[1]
                b = study.Best_float_params()[2]
                b1 = study.Best_float_params()[3]
        if mode == "user_input":
            if a <= g / l or a1 <= 2 * g / l or b <= l * np.sqrt(
                    2) / 2 or b1 <= l / 2 or a1 / b + a / b1 >= 2 / l:
                raise Exception(
                    "Задайте параметры, при которых выполняется условие асимптотической устойчивости, текущие значения параметров: a = {}, b = {}, a1 = {}, b1 = {}."
                    .format(a, b, a1, b1))

    if stabilization == "upper_position_and_stand":
        if angle == "small_angle":
            if mode == "auto":

                def objective(trial):
                    c = trial.suggest_float("c", -g, g)
                    b = trial.suggest_float("b", (m * l**2) / 2, m * l**2)
                    d = trial.suggest_float("d", -2, np.sqrt(1.34 * g / l))
                    a = trial.suggest_float("a", -2.5, 2.5 * l * g)
                    return (l / (2 * g)) * (
                        ((2 * b * g + 2 * d * g * m**2 * l**4)**2) /
                        ((2 * b + d * m * l**2) *
                         (4 * g * m * l**3 + b**2)) - 4 * g / l - b**2 /
                        (m**2 * l**4))

                study = iOpt.create_study()
                study.optimize(objective=objective,
                               solver_parameters=iOpt.SolverParameters(
                                   r=5,
                                   eps=0.001,
                                   iters_limit=1000,
                                   refine_solution=True))
                c = study.Best_float_params()[0]
                b = study.Best_float_params()[1]
                d = study.Best_float_params()[0]
                a = study.Best_float_params()[1]
                a1 = study.Best_values() + h
            if mode == "user_input":
                if c <= -g or c >= g or b <= (
                        m * l**2
                ) / 2 or b >= m * l**2 or d <= -2 or d >= np.sqrt(
                        1.34 * g / l
                ) or a <= -2.5 or a >= 2.5 * l * g or a1 <= (l / (2 * g)) * (
                    ((2 * b * g + 2 * d * g * m**2 * l**4)**2) /
                    ((2 * b + d * m * l**2) *
                     (4 * g * m * l**3 + b**2)) - 4 * g / l - b**2 /
                    (m**2 * l**4)):
                    raise Exception(
                        "Задайте параметры, при которых выполняется условие асимптотической устойчивости, текущие значения параметров: a = {}, b = {}, a1 = {}, c = {}, d = {}."
                        .format(a, b, a1, c, d))
        if angle == "any_angle":
            if mode == "user_input":
                a1 = (2 * g * l) / 3
                a = (4 * g * l) / 3
                b = np.sqrt(2 * g * l / 3)
                b1 = np.sqrt(g * l / 3)
                d = np.sqrt(g * l / 2)
                # Задайте жесткость управления
                с = 2.5 * g / l + h
                if с <= 2.5 * g / l:
                    raise Exception(
                        "Задайте параметры, при которых выполняется условие асимптотической устойчивости, текущие значения параметров: a = {}, b = {}, a1 = {}, b1 = {}, c = {}, d = {}."
                        .format(a, b, a1, b1, c, d))


def inverted_pendulum(t, state):
    global a
    if number_of_links == 1:
        x, x_dot, theta, theta_dot = state
        if angle == "small_angle":
            x_ddot = -a * (theta - np.pi) - b * theta_dot
        if angle == "any_angle":
            x_ddot = -g * np.tan(theta - np.pi) - (
                a * (theta - np.pi) + b * theta_dot) / np.cos(theta - np.pi)
        if stabilization == "upper_position_and_stand":
            x_ddot += c * x + d * x_dot
        if angle == "small_angle":
            theta_ddot = (x_ddot + g * (theta - np.pi)) / l
        if angle == "any_angle":
            theta_ddot = (x_ddot * np.cos(theta - np.pi) +
                          g * np.sin(theta - np.pi)) / l
        return [x_dot, x_ddot, theta_dot, theta_ddot]
    if number_of_links == 2:
        x, x_dot, theta, theta_dot, theta2, theta2_dot = state
        if angle == "small_angle":
            x_ddot = -a1 * (theta - np.pi) - a * (
                theta2 - np.pi) - b1 * theta_dot - b * theta2_dot
        if angle == "any_angle":
            x_ddot = (1 / np.cos(theta2 - np.pi)) * (
                -b * (np.cos((theta - np.pi) -
                             (theta2 - np.pi)) + 1) * theta_dot +
                b * np.cos((theta - np.pi) - (theta2 - np.pi)) * theta2_dot -
                a1 * np.cos((theta - np.pi) - (theta2 - np.pi)) *
                (theta - np.pi) - a * (1 + np.sin(theta2 - np.pi) /
                                       (theta2 - np.pi)) *
                (theta2 - np.pi) - b1 * np.sin((theta - np.pi) -
                                               (theta2 - np.pi)) * theta_dot**2
                - g * np.sin(theta2 - np.pi))
        if stabilization == "upper_position_and_stand":
            x_ddot += c * x + d * x_dot
        if angle == "small_angle":
            theta2_ddot = -2 * g * (theta - np.pi) / l + 2 * g * (theta2 -
                                                                  np.pi) / l
            theta_ddot = x_ddot / l + g * (theta2 - np.pi) / l - theta2_ddot
        if angle == "any_angle":
            theta_ddot = (x_ddot * np.cos(theta2 - np.pi) +
                          l * theta2_ddot * np.cos((theta - np.pi) -
                                                   (theta2 - np.pi)) -
                          l * theta_dot**2 * np.sin((theta - np.pi) -
                                                    (theta2 - np.pi)) -
                          g * np.sin(theta2 - np.pi)) / (l * np.cos(
                              (theta - np.pi) - (theta2 - np.pi)))
            theta2_ddot = (
                2 * x_ddot * np.cos(theta2 - np.pi) -
                2 * x_dot * np.cos(theta - np.pi) + 2 * g *
                (np.sin(theta2 - np.pi) - np.sin(theta - np.pi)) + l *
                (theta_dot**2 + 2 * theta2_dot**2) * np.sin(
                    (theta - np.pi) -
                    (theta2 - np.pi))) / (2 * l *
                                          (np.cos((theta - np.pi) -
                                                  (theta2 - np.pi)) - 2))
        return [x_dot, x_ddot, theta_dot, theta_ddot, theta2_dot, theta2_ddot]


# Решение системы
solution = solve_ivp(inverted_pendulum,
                     t_span,
                     initial_state,
                     t_eval=t_eval,
                     method='RK45',
                     rtol=1e-8,
                     atol=1e-10)

# Создаем фигуру
fig = plt.figure(figsize=(12, 12))
plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08, hspace=0.4)
gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1])
ax1 = fig.add_subplot(gs[0])  # Анимация
ax2 = fig.add_subplot(gs[1])  # График угла
ax3 = fig.add_subplot(gs[2])  # График положения
ax4 = fig.add_subplot(gs[3])  # График угловой скорости

# Стилизация анимации
colors = {
    'cart': '#4682B4',  # Стальной синий
    'pendulum': '#FF6347',  # Томатный
    'bob': '#8B0000',  # Темно-красный
    'angle': '#1E90FF',  # Ярко-синий
    'position': '#FF8C00',  # Темно-оранжевый
    'velocity': '#9370DB',  # Средний пурпурный
    'grid': '#E0E0E0',  # Светло-серый
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

cart_width = 0
cart_height = 0
cart = Rectangle((0, 0), 0, 0)
if number_of_links == 1:
    cart_width, cart_height = 0.4, 0.2
    cart = Rectangle((0, 0),
                     cart_width,
                     cart_height,
                     fc=colors['cart'],
                     ec='#404040',
                     lw=1.5,
                     alpha=0.9)
    pendulum, = ax1.plot([], [],
                         color=colors['pendulum'],
                         linestyle='-',
                         linewidth=3.5,
                         alpha=0.85)
    bob, = ax1.plot([], [],
                    'o',
                    color=colors['bob'],
                    markersize=14,
                    alpha=0.9,
                    markeredgecolor='#4A0000')
    time_text = ax1.text(0.72,
                         0.12,
                         '',
                         transform=ax1.transAxes,
                         bbox=dict(facecolor='white',
                                   alpha=0.85,
                                   edgecolor='#696969',
                                   boxstyle='round'))
    ax1.add_patch(cart)

    ax1.set_aspect('equal')
    ax1.grid(True, color=colors['grid'], linestyle='--')
    ax1.set_xlim(-1, 3)
    ax1.set_ylim(-1, 1)
    ax1.set_title(
        f'Стабилизация перевернутого маятника (начальный угол: {initial_angle:.2f} рад)',
        fontsize=12,
        pad=10)

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

if number_of_links == 2:
    cart_width, cart_height = 0.8, 0.4
    cart = Rectangle((-cart_width / 2, -cart_height / 2),
                     cart_width,
                     cart_height,
                     fc=colors['cart'],
                     ec=colors['cart_edge'],
                     alpha=0.9,
                     lw=1.5,
                     zorder=0)
    link1, = ax1.plot([], [],
                      color=colors['link1'],
                      linestyle='-',
                      lw=3,
                      alpha=0.85,
                      zorder=5)
    link2, = ax1.plot([], [],
                      color=colors['link2'],
                      linestyle='-',
                      lw=3,
                      alpha=0.85,
                      zorder=5)
    joint1, = ax1.plot([], [],
                       'o',
                       color=colors['joint1'],
                       markersize=10,
                       alpha=0.9,
                       markeredgecolor='#4A0000',
                       zorder=10)
    joint2, = ax1.plot([], [],
                       'o',
                       color=colors['joint2'],
                       markersize=10,
                       alpha=0.9,
                       markeredgecolor='#000060',
                       zorder=10)

    # Информационное окно в правом верхнем углу
    time_text = ax1.text(0.98,
                         0.95,
                         '',
                         transform=ax1.transAxes,
                         bbox=dict(facecolor=colors['text_bg'],
                                   alpha=0.85,
                                   edgecolor='#696969',
                                   boxstyle='round'),
                         verticalalignment='top',
                         horizontalalignment='right')
    ax1.add_patch(cart)

    ax1.set_aspect('equal')
    ax1.grid(True, color='#D3D3D3', linestyle='--', alpha=0.3)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-3, 3)
    ax1.set_title(
        f'Двухзвенный маятник на тележке\nНачальные углы: φ1 = {(initial_angle):.2f}, φ2 = {(initial_angle2):.2f} рад',
        fontsize=12,
        pad=15)

    # Настройка графиков
    for ax in [ax2, ax3, ax4]:
        ax.grid(True, color='#D3D3D3', linestyle='--', alpha=0.3)
        ax.set_facecolor(colors['text_bg'])
        ax.set_xlim(0, solution.t[-1])

    # График углов
    ax2.set_ylabel('Углы (рад)', fontsize=10)
    ax2.axhline(np.pi,
                color=colors['angle_ref'],
                linestyle=':',
                lw=2,
                alpha=0.7,
                label='Вертикаль (π)')
    line_phi1, = ax2.plot([], [],
                          color=colors['plot_phi1'],
                          alpha=0.85,
                          lw=2,
                          label='φ₁ (нижний)')
    line_phi2, = ax2.plot([], [],
                          color=colors['plot_phi2'],
                          alpha=0.85,
                          lw=2,
                          label='φ₂ (верхний)')
    ax2.legend(fontsize=9)

    # График положения
    ax3.set_ylabel('Положение тележки (м)', fontsize=10)
    line_position, = ax3.plot([], [],
                              color=colors['plot_position'],
                              alpha=0.85,
                              lw=2,
                              label='x(t)')
    ax3.legend(fontsize=9)

    # График скоростей
    ax4.set_xlabel('Время (с)', fontsize=10)
    ax4.set_ylabel('Скорости', fontsize=10)
    line_xdot, = ax4.plot([], [],
                          color=colors['plot_velocity1'],
                          alpha=0.85,
                          lw=2,
                          label='ẋ(t)')
    line_phi1dot, = ax4.plot([], [],
                             color=colors['plot_velocity2'],
                             linestyle='--',
                             alpha=0.85,
                             lw=2,
                             label='φ̇₁(t)')
    line_phi2dot, = ax4.plot([], [],
                             color=colors['plot_velocity3'],
                             linestyle='-.',
                             alpha=0.85,
                             lw=2,
                             label='φ̇₂(t)')
    ax4.legend(fontsize=9)


def init():
    cart.set_xy((-cart_width / 2, -cart_height / 2))
    time_text.set_text('')
    if number_of_links == 1:
        pendulum.set_data([], [])
        bob.set_data([], [])

        ax2.clear()
        ax3.clear()
        ax4.clear()

        for ax in [ax2, ax3, ax4]:
            ax.grid(True, color=colors['grid'], linestyle='--')
            ax.set_xlim(0, solution.t[-1])
            ax.set_facecolor('#F8F8FF')

        ax2.set_ylabel('Угол, θ (рад)')
        ax2.set_ylim(2.5, 3.5)
        ax2.axhline(np.pi,
                    color='#2E8B57',
                    linestyle=':',
                    lw=2.5,
                    alpha=0.8,
                    label='θ=π (цель)')
        ax2.legend()

        ax3.set_ylabel('Положение, x (м)')

        ax4.set_xlabel('Время, t (с)')
        ax4.set_ylabel('Угл. скорость, θ\' (рад/с)')
        ax4.axhline(0, color='#808080', linestyle='--', alpha=0.6)

        return cart, pendulum, bob, time_text
    if number_of_links == 2:
        link1.set_data([], [])
        link2.set_data([], [])
        joint1.set_data([], [])
        joint2.set_data([], [])

        for line in [
                line_phi1, line_phi2, line_position, line_xdot, line_phi1dot,
                line_phi2dot
        ]:
            line.set_data([], [])

        for ax in [ax2, ax3, ax4]:
            ax.relim()
            ax.autoscale_view()

        return (cart, link1, link2, joint1, joint2, time_text, line_phi1,
                line_phi2, line_position, line_xdot, line_phi1dot,
                line_phi2dot)


def animate(i):
    x = solution.y[0, i]
    cart.set_xy((x - cart_width / 2, -cart_height / 2))
    if number_of_links == 1:
        theta = solution.y[2, i]

        pendulum.set_data([x, x + l * np.sin(theta)], [0, -l * np.cos(theta)])
        bob.set_data([x + l * np.sin(theta)], [-l * np.cos(theta)])

        ax2.plot(solution.t[:i + 1],
                 solution.y[2, :i + 1],
                 color=colors['angle'],
                 alpha=0.85)
        ax3.plot(solution.t[:i + 1],
                 solution.y[0, :i + 1],
                 color=colors['position'],
                 alpha=0.85)
        ax4.plot(solution.t[:i + 1],
                 solution.y[3, :i + 1],
                 color=colors['velocity'],
                 alpha=0.85)

        ax1.set_xlim(x - 2, x + 2)

        time_text.set_text(f'Время: {solution.t[i]:.2f} с\n'
                           f'Угол: {theta:.6f} рад\n'
                           f'От π: {theta - np.pi:.2e} рад\n'
                           f'Угл. скорость: {solution.y[3, i]:.2e} рад/с\n'
                           f'Позиция: {x:.4f} м\n'
                           f'Скорость: {solution.y[1, i]:.2e} м/с')

        return cart, pendulum, bob, time_text
    if number_of_links == 2:
        phi1 = solution.y[2, i]
        phi2 = solution.y[4, i]

        # Координаты звеньев
        x0, y0 = x, 0.0
        x1 = x0 + l * np.sin(phi1)
        y1 = y0 - l * np.cos(phi1)
        x2 = x1 + l * np.sin(phi2)
        y2 = y1 - l * np.cos(phi2)

        # Динамическое масштабирование анимации
        ax1.set_xlim(x - 5, x + 5)

        # Обновление анимации
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
                             [[line_phi1, line_phi2], [line_position],
                              [line_xdot, line_phi1dot, line_phi2dot]]):
            ax.relim()
            for line in lines:
                ax.update_datalim(
                    np.column_stack([solution.t[:i + 1],
                                     line.get_ydata()]))
            ax.autoscale_view()

        # Информация
        time_text.set_text(f'Время: {solution.t[i]:.2f} с\n'
                           f'Позиция: {x:.3f} м\n'
                           f'Скорость: {solution.y[1, i]:.2f} м/с\n'
                           f'φ₁: {np.degrees(phi1):.1f}°\n'
                           f'φ₂: {np.degrees(phi2):.1f}°')

        return (cart, link1, link2, joint1, joint2, time_text, line_phi1,
                line_phi2, line_position, line_xdot, line_phi1dot,
                line_phi2dot)


# Создаем анимацию
ani = animation.FuncAnimation(fig,
                              animate,
                              frames=len(solution.t),
                              init_func=init,
                              blit=False,
                              interval=20,
                              repeat=False)
plt.tight_layout()
plt.show()
