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
number_of_links = 1  # Число звеньев
# number_of_links = 2

# Начальные условия
initial_angle_offset = 30  # Начальное отклонение от вертикали(в градусах)
if angle == "small_angle" and initial_angle_offset > 5:
    raise Exception(
        "В режиме стабилизации с малым начальным углом - начальный угол не должен превышать пяти градусов, введенное значение: "
        "{}.".format(initial_angle_offset))
if initial_angle_offset >= 90 or initial_angle_offset <= -90:
    raise Exception(
        "Задайте начальный угол отклонения от вертикали в градусах в диапазоне (-90, 90), введенное значение: "
        "{}.".format(initial_angle_offset))

initial_angle = np.pi - np.radians(initial_angle_offset)
initial_angular_velocity = 1  # Начальная угловая скорость(рад./с)

initial_u0 = 1  # Начальное положение опоры(м)
if initial_u0 < 0:
    raise Exception(
        "Задайте начальное положение опоры в метрах (u0 > 0), введенное значение: "
        "{}.".format(initial_u0))

initial_u0_speed = 1  # Начальная скорость опоры(м/с)

initial_angle2_offset = 30  # Начальное отклонение от вертикали для второго звена(в градусах)
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
с = 0.1
d = 0.25

# Для двузвенного маятника также задайте
a1 = g
b1 = l

if number_of_links == 1:
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
                c = trial.suggest_float("c", 0, 10)
                d = trial.suggest_float("d", 0, 10)
                b = trial.suggest_float("b", 10 / l, 10 / l + 10)
                return 1 / (c * b / d + g * (1 + d * l / (b - d * l)))

            study = iOpt.create_study()
            study.optimize(objective=objective,
                           solver_parameters=iOpt.SolverParameters(
                               r=5,
                               eps=0.001,
                               iters_limit=100,
                               refine_solution=True))
            c = study.Best_float_params()[0]
            d = study.Best_float_params()[1]
            b = study.Best_float_params()[2]
            a = study.Best_values()
        if mode == "user_input":
            if c <= 0 or d <= 0 or b <= d / l or a <= c * b / d + g * (
                    1 + d * l / (b - d * l)):
                raise Exception(
                    "Задайте параметры, при которых выполняется условие асимптотической устойчивости, текущие значения параметров: a = {}, b = {}, c = {}, d = {}."
                    .format(a, b, c, d))


def inverted_pendulum(state):
    if number_of_links == 1:
        x, x_dot, theta, theta_dot = state
        if angle == "small_angle":
            x_ddot = -a * (theta - np.pi) - b * theta_dot
        if angle == "any_angle":
            a -= g
            x_ddot = -g * np.tg(theta - np.pi) - (
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
            x_ddot = -a1 * (theta - np.pi) - a * (theta2 -
                                                  np.pi) - b * theta2_dot
        if angle == "any_angle":
            a -= g
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
            theta_ddot = (2 * g * np.sin(theta - np.pi) - g *
                          np.sin(theta2 - np.pi) - 2 * x_ddot / l) / (2 * l)
            theta2_ddot = (g * np.sin(theta - np.pi) -
                           g * np.sin(theta2 - np.pi) + x_ddot / l) / l
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
    if (abs(solution.y[2, i] - np.pi) < stable_threshold
            and abs(solution.y[3, i]) < stable_threshold
            and abs(solution.y[1, i]) < stable_threshold):
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

ax5 = fig.add_subplot(
    gs[3, :])  # График управления (занимает всю нижнюю строку)

# Настройка анимации с новыми цветами
cart_width, cart_height = 0.4, 0.2
cart = Rectangle((0, 0),
                 cart_width,
                 cart_height,
                 fc='#4682B4',
                 ec='#2F4F4F',
                 alpha=0.8)
pendulum, = ax1.plot([], [],
                     color='#FF6347',
                     linestyle='-',
                     linewidth=3,
                     alpha=0.8)
bob, = ax1.plot([], [], 'o', color='#8B0000', markersize=12, alpha=0.8)
time_text = ax1.text(0.65,
                     0.1,
                     '',
                     transform=ax1.transAxes,
                     bbox=dict(facecolor='white', alpha=0.6))
ax1.add_patch(cart)

ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-1, 3)
ax1.set_ylim(-1, 1)
ax1.set_title(
    f'Стабилизация перевернутого маятника (начальный угол: {initial_angle:.2f} рад)',
    pad=20)

# Настройка графиков с новыми цветами и прозрачностью
ax2.grid(True, alpha=0.3)
ax2.set_ylabel('Угол, θ (рад)')
ax2.set_title('Эволюция угла отклонения', pad=10)
ax2.axhline(np.pi,
            color='#2E8B57',
            linestyle='--',
            label='θ=π (цель)',
            alpha=0.7)
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
    time_text.set_text(f'Время: {solution.t[i]:.2f} с\n'
                       f'Угол: {theta:.6f} рад\n'
                       f'От π: {theta - np.pi:.2e} рад\n'
                       f'Угл. скорость: {theta_dot:.2e} рад/с\n'
                       f'Позиция: {x:.4f} м\n'
                       f'Скорость: {solution.y[1, i]:.2e} м/с\n'
                       f'Управление: {u:.2e} м/с²')

    # Обновляем график управления (красная точка на текущем моменте)
    if hasattr(animate, 'u_point'):
        animate.u_point.remove()
    animate.u_point = ax5.plot(solution.t[i], u, 'ro', markersize=4)[0]

    return cart, pendulum, bob, time_text, animate.u_point


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
