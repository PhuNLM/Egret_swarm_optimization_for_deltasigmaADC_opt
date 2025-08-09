import numpy as np
import os
import csv

P = 8
D = 15
t_max = 80
beta1, beta2 = 0.9, 0.99
r_accept = 0.3	

lb = np.array([0.22e-6,  0.42,  0.42,  0.42, 0.42, 0.41, 0.42,  0.42,    1.8,    15,     7,   25,    7,     7,   10])
ub = np.array([ 1e-6,    0.60,  0.60,  0.60, 0.60, 0.60, 0.60,  0.60,    3.2,    60,    15,   80,   15,    18,   60])
#		         Cc      12345   68     79    8     1     2      5      12345    68     79     8     1      2     5
hop = ub - lb

#write widths, lengths to txt file
def write_ESOA_para(x):
    size_pop, n_dim = x.shape
    with open("OTA_para.txt", "w") as f:
        for j in range(n_dim):
            for i in range(size_pop):
                f.write(f"{x[i, j] * 1e-6}\n")

#read res from cadence
def read_simulation_results():
    with open("Result.txt", "r") as f:
        data = np.array([float(line.strip()) for line in f])
    cond = data[0:8]
    GBW  = data[8:16]
    PM   = data[16:24]
    Gain = data[24:32]
    SR   = data[32:40]
    PWR  = data[40:48]

    FoM = np.zeros(8)
    for i in range(8):
        if PM[i] < 60 or Gain[i] < 45:     #constraints for phase margin and gain
            FoM[i] = 1e20
        else:
            FoM[i] = 1/(Gain[i])  

    return FoM, cond, GBW, PM, Gain, SR, PWR

def func(x):
    write_ESOA_para(x)
    os.system("ocean -nograph -restore scrip.ocn")       #call cadence 
    return read_simulation_results()

def bound(x):
    return np.clip(x, lb, ub)

x = np.random.uniform(lb, ub, (P, D))
m = np.zeros((P, D))
v = np.zeros((P, D))
w = np.random.uniform(-1, 1, (P, D))

x_best_hist = x.copy()
y_best_hist = np.ones(P) * np.inf
g_best_hist = np.zeros_like(x)

x_global_best = x[0].copy()
y_global_best = np.inf
g_global_best = np.zeros(D)



for t in range(t_max):
        print(f"\n epoch {t+1}/{t_max}")

        y, cond, GBW, PM, Gain, SR, PWR = func(x)
        p = np.sum(w * x, axis=1)
        g_temp = (p - y).reshape(-1, 1) * x

        mask = y < y_best_hist
        y_best_hist = np.where(mask, y, y_best_hist)
        x_best_hist = np.where(mask[:, None], x, x_best_hist)
        g_best_hist = np.where(mask[:, None], g_temp, g_best_hist)

        if y.min() < y_global_best:
            idx = y.argmin()
            y_global_best = y[idx]
            x_global_best = x[idx].copy()
            g_global_best = g_temp[idx] / (np.linalg.norm(g_temp[idx]) + 1e-8)

        #update gradient 
        pdiff = x_best_hist - x
        cdiff = x_global_best - x
        f_p = (y_best_hist - y).reshape(-1, 1)
        f_c = (y_global_best - y).reshape(-1, 1)
        d_p = pdiff * f_p / (np.linalg.norm(pdiff, axis=1, keepdims=True) ** 2 + 1e-8)
        d_g = cdiff * f_c / (np.linalg.norm(cdiff, axis=1, keepdims=True) ** 2 + 1e-8)
        r1, r2, r3 = np.random.rand(P, 1), np.random.rand(P, 1), np.random.rand(P, 1)
        g = r1 * g_temp + r2 * d_p + r3 * d_g
        g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)

        #update weights (egret a)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        w = w - m / (np.sqrt(v) + 1e-8)

        
        decay = np.exp(-t / (0.1 * t_max))     
        #egret a                         
        x_a = bound(x + decay * 0.1 * hop * g)                
        #egret b              
        r_bi = np.random.uniform(-np.pi/2, np.pi/2, (P, D))
        x_b = bound(x + np.tan(r_bi) * hop / (1 + t) * 0.5)      
        #egret c           
        r_h = np.random.rand(P, D) * 0.5
        r_g = np.random.rand(P, D) * 0.5
        x_c = bound((1 - r_h - r_g) * x + r_h * (x_best_hist - x) + r_g * (x_global_best - x))          

        #fitness a b c
        y_a, *_ = func(x_a)
        y_b, *_ = func(x_b)
        y_c, *_ = func(x_c)

        x_new = np.copy(x)
        y_new = np.copy(y)

        #updae new fitness
        for i in range(P):
            ys = [y_a[i], y_b[i], y_c[i]]
            xs = [x_a[i], x_b[i], x_c[i]]
            best_i = np.argmin(ys)
            if ys[best_i] < y[i] or np.random.rand() < r_accept:
                x_new[i] = xs[best_i]
                y_new[i] = ys[best_i]
        x = x_new
        y = y_new




        # write to csv
        with open("ota_best_log_1.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
            "Iteration",
            "Cc(pF)",
            "LM12345", "LM68", "LM79", "LM8", "LM1", "LM2", "LM5",
            "WM12345", "WM68", "WM79", "WM8", "WM1", "WM2", "WM5",
            "GBW(MHz)", "PM(deg)", "Gain(dB)", "SlewRate(V/s)", "Power(W)"
            ])

            y_chk, cond, GBW, PM, Gain, SR, PWR = read_simulation_results()
            best_idx = y_chk.argmin()
            writer.writerow([
                t + 1,
                x_global_best[0] * 1e6,               # Cc in pF
                *x_global_best[1:],                   # μm
                GBW[best_idx] / 1e6,                  # GBW in MHz
                PM[best_idx],
                Gain[best_idx],
                SR[best_idx],
                PWR[best_idx]
            ])

        print(f"best FOM round {t+1}: {y_global_best:.4e}")

print("\n completed")
print(f"min FOM: {y_global_best:.6e}")
print(f" best x:\n{x_global_best}")

