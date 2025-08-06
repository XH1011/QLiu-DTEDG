# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal.windows import kaiser
from scipy.signal import detrend
from scipy.interpolate import CubicSpline
import os

# ================== 参数定义 ==================
# ------ 齿轮几何参数 ------
r_1 = 0.041  # 直齿轮分度圆半径 (m)
r_2 = 0.094  # 面齿轮分度圆半径 (m)
m = 0.0045  # 模数 (m)
h_a = 1.0 * m
h_f = 1.25 * m

# ------ 动力学参数 ------
m1 = 0.805  # 质量 (kg)
m2 = 2.46
me = 0.3036
alpha = np.deg2rad(25)  # 压力角
a1 = np.sin(alpha)
a2 = np.cos(alpha)

k_x1 = k_z1 = k_x2 = k_z2 = 1.2e6  # 支撑刚度 (N/m)

# ------ 阻尼参数 ------
zeta = 0.01
c_x1 = c_z1 = 1  # 阻尼 (N·s/m)
c_x2 = c_z2 = 1

# ------ 负载参数 ------
T_load = 20  # 阻力矩 (N·m)
F = T_load / r_2  # 负载力 (N)

# ------ 间隙和误差激励 ------
b0 = 0.2e-3  # 齿侧间隙 (m)
Delta_lambda = 0.1e-3
e0 = 100e-6  # 误差常量 (m)
er = 200e-6
Omega_h = 420 * 2 * np.pi  # 啮合角速度 (精确105Hz)
c_m = 11

def enhanced_fft(signal, fs):
    """改进的FFT分析函数"""
    signal = detrend(signal)
    window = kaiser(len(signal), beta=38)
    nfft = 2 ** 20

    fft_result = np.fft.fft(signal * window, n=nfft)
    freqs = np.fft.fftfreq(nfft, 1 / fs)[:nfft // 2]
    mag = np.abs(fft_result)[:nfft // 2] * 2 / np.sum(window)
    return freqs, mag
# ================== 新增频谱分析函数 ==================
class EnhancedStiffnessGenerator:
    def __init__(self, normal_data, fault_data, fs, theory_points):
        self.fs = fs
        self.cycle_time = 1 / (fs / theory_points)
        self.transition_ratio = 0.1

        # 确保故障数据包含4个周期的刚度数据
        assert len(fault_data) == 4 * len(normal_data), "故障数据应包含4个周期的刚度数据"

        self.normal_data = normal_data
        self.fault_data = fault_data

        # 创建正常周期的插值器
        self.normal_interp = CubicSpline(
            np.linspace(0, self.cycle_time, len(normal_data)),
            normal_data,
            bc_type=((2, 0.0), (2, 0.0))
        )

        # 创建故障周期的插值器（分为4个独立周期）
        self.fault_cycles = []
        for i in range(4):
            cycle_data = fault_data[i * len(normal_data):(i + 1) * len(normal_data)]
            self.fault_cycles.append(
                CubicSpline(
                    np.linspace(0, self.cycle_time, len(normal_data)),
                    cycle_data,
                    bc_type=((2, 0.0), (2, 0.0))
                )
            )  # 添加闭合括号

    def get_stiffness(self, t):  # 正确的方法定义
        phase_time = t % self.cycle_time
        cycle_num = int(t * 105)  # 105Hz对应每秒啮合周期数

        # 计算当前周期和下一个周期的刚度
        current_value = self._get_raw_stiffness(phase_time, cycle_num)
        next_value = self._get_raw_stiffness(phase_time, cycle_num + 1)

        # 平滑过渡处理
        transition_width = self.transition_ratio * self.cycle_time
        if phase_time > (self.cycle_time - transition_width):
            weight = 0.5 * (1 - np.cos(
                np.pi * (phase_time - (self.cycle_time - transition_width)) / transition_width
            ))
            return current_value * (1 - weight) + next_value * weight
        else:
            return current_value

    def _get_raw_stiffness(self, t, cycle_num):
        t_clipped = np.clip(t, 0, self.cycle_time)

        # 每48个周期中最后4个为故障周期（44-47）
        if (cycle_num % 48) >= 44:
            fault_cycle_index = (cycle_num % 4)  # 在4个故障模式中循环
            return self.fault_cycles[fault_cycle_index](t_clipped)
        else:
            return self.normal_interp(t_clipped)


# ================== 请在此处填入实际刚度数据 ==================
normal_k = np.array([])
fault_k = np.array([
   ])  # 总长度应为normal_k长度的4倍

# 初始化动态刚度生成器
stiffness_gen = EnhancedStiffnessGenerator(
    normal_data=normal_k,
    fault_data=fault_k,
    fs=10e3,
    theory_points=95.23809523809524
)


# ================== 系统方程 ==================
def k_bar(t):
    return stiffness_gen.get_stiffness(t)


def e_func(t):
    return e0 + er * np.sin(Omega_h * t)


def f_lambda(lambd):
    return np.where(lambd > b0, lambd - b0, np.where(lambd < -b0, lambd + b0, 0))


def system_odes(t, y):
    X1, dX1, Z1, dZ1, X2, dX2, Z2, dZ2, lambd, dlambd = y

    kt = k_bar(t)
    et = e_func(t)

    F_n = kt * f_lambda(lambd) + c_m * dlambd
    F_x = F_n * a2
    F_z = F_n * a1

    A = np.array([
        [m1, 0, 0, 0, 0],
        [0, m1, 0, 0, 0],
        [0, 0, m2, 0, 0],
        [0, 0, 0, m2, 0],
        [-me * a2, -me * a1, me * a2, me * a1, me]
    ])

    B = np.array([
        -c_x1 * dX1 - k_x1 * X1 - F_x,
        -c_z1 * dZ1 - k_z1 * Z1 - F_z,
        F_z - c_z2 * dZ2 - k_z2 * Z2,
        F_x - c_x2 * dX2 - k_x2 * X2,
        r_2 * F + et - a2 * kt * f_lambda(lambd) - a2 * c_m * dlambd
    ])

    accels = np.linalg.solve(A, B)
    ddX1, ddZ1, ddZ2, ddX2, ddlambd = accels

    return [dX1, ddX1, dZ1, ddZ1, dX2, ddX2, dZ2, ddZ2, dlambd, ddlambd]


# ================== 初始条件 ==================
X2_static = (F * np.cos(alpha) / k_x2) * (r_2 / r_1)
Z2_static = (F * np.sin(alpha) / k_z2) * (r_2 / r_1)
lambda_initial = -b0 - Delta_lambda

y0 = [1e-6, 0, 1e-6, 0, X2_static, 0, Z2_static, 0, lambda_initial, 0]

# ================== 执行仿真 ==================
t_span = (0, 100)
sampling_rate = 10e3
t_eval = np.linspace(*t_span, int(sampling_rate * t_span[1]))

sol = solve_ivp(
    fun=system_odes,
    t_span=t_span,
    y0=y0,
    t_eval=t_eval,
    method='BDF',
    rtol=1e-8,
    atol=1e-10,
    max_step=1e-4
)

# ================== 后处理 ==================
# 提取Z1加速度数据（修改点1）
ddX1 = sol.y[1]  # X1方向的加速度
ddZ1 = sol.y[3]  # Z1方向的加速度

# 创建输出目录（修改点2）
output_dir = 'D:/'
os.makedirs(output_dir, exist_ok=True)

# 保存加速度数据（修改点3）
accel_data = pd.DataFrame({
    '时间(s)': sol.t,
    'X1加速度(m/s²)': ddX1,
    'Z1加速度(m/s²)': ddZ1
})
output_path = os.path.join(output_dir, '.xlsx')
accel_data.to_excel(output_path, index=False)

# ================== 频谱分析（修改点4）==================
last_10s_mask = sol.t >= (t_span[1] - 10)
ddZ1_last10s = ddZ1[last_10s_mask]

freqs, mag = enhanced_fft(ddZ1_last10s, sampling_rate)

# ================== 可视化 ==================
plt.figure(figsize=(15, 12))

# Z1加速度时域图（最后10秒）
plt.subplot(2, 1, 1)
plt.plot(sol.t[last_10s_mask], ddZ1_last10s, 'b', linewidth=0.5)
plt.title('Z1方向振动加速度时域响应 (最后10秒)')
plt.xlabel('时间 (s)')
plt.grid(True)

# Z1加速度频谱分析
plt.subplot(2, 1, 2)
plt.semilogy(freqs, mag, 'g', linewidth=0.8)
plt.xlim(0, 500)
plt.axvline(105, color='r', linestyle='--', label='啮合频率105Hz')
plt.axvline(105 / 48, color='orange', linestyle='--', label='故障特征频率2.1875Hz')

# 标注边频带
for n in range(1, 6):
    freq = 105 + n * (105 / 48)
    plt.axvline(freq, color='grey', linestyle=':', alpha=0.5)
    plt.axvline(105 - n * (105 / 48), color='grey', linestyle=':', alpha=0.5)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()