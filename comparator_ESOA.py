import numpy as np
import os
import csv

P = 8
D = 10
t_max = 25
beta1, beta2 = 0.9, 0.99
r_accept = 0.3

lb = np.array([0.26, 0.26, 0.26, 0.26, 0.26,   1,     1,      1,       1,     1])
ub = np.array([0.52 ,0.52, 0.52, 0.52, 0.52,   50,    10,     10,     60,    10])
hop = ub - lb

#write widths, lengths to txt file
def write_ESOA_para(x):
    size_pop, n_dim = x.shape
    with open("Comp_para.txt", "w") as f:
        for j in range(n_dim):
            for i in range(size_pop):
                f.write(f"{x[i, j] * 1e-6}\n")


#read res from cadence
def read_simulation_results():
    with open("Result.txt", "r") as f:
        data = np.array([float(line.strip()) for line in f])
    delay = data[0:8]
    pwr  = data[8:16]
    PDP = pwr * delay
    for i in range(8):
        if PDP[i] < 0:                      #contraints for fail simulation
            PDP[i] = 1e20
    return PDP, delay, pwr

def func(x):
    write_ESOA_para(x)
    os.system("ocean -nograph -restore 	comp_run.ocn")
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

with open("comp_best_log.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "Iteration", "LM12", "LM34", "LM56", "LM7", "LS",
        "WM12", "WM34", "WM56", "WM7", "WS",
        "Best PDP (fJ)", "Best Power (W)", "Best Delay (ps)"
    ])

    for t in range(t_max):
        print(f"\n epoch {t+1}/{t_max}")

        y, delay, pwr = func(x)
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
            best_power = pwr[idx]
            best_delay = delay[idx]
            best_pdp = y[idx]




        #update gradient 
        pdiff = x_best_hist - x
        cdiff = x_global_best - x
        f_p = (y_best_hist - y).reshape(-1, 1)
        f_c = (y_global_best - y).reshape(-1, 1)
        d_p = pdiff * f_p / (np.linalg.norm(pdiff, axis=1, keepdims=True)**2 + 1e-8)
        d_g = cdiff * f_c / (np.linalg.norm(cdiff, axis=1, keepdims=True)**2 + 1e-8)

        r1, r2, r3 = np.random.rand(P, 1), np.random.rand(P, 1), np.random.rand(P, 1)
        g = r1 * g_temp + r2 * d_p + r3 * d_g
        g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)


        #weights
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        w = w - m / (np.sqrt(v) + 1e-8)

        #egrets
        decay = np.exp(-t / (0.1 * t_max))
        x_a = bound(x + decay * 0.1 * hop * g)
        r_bi = np.random.uniform(-np.pi/2, np.pi/2, (P, D))
        x_b = bound(x + np.tan(r_bi) * hop / (1 + t) * 0.5)
        r_h = np.random.rand(P, D) * 0.5
        r_g = np.random.rand(P, D) * 0.5
        x_c = bound((1 - r_h - r_g) * x + r_h * (x_best_hist - x) + r_g * (x_global_best - x))

        y_a, *_ = func(x_a)
        y_b, *_ = func(x_b)
        y_c, *_ = func(x_c)


        #new fitness
        x_new = np.copy(x)
        y_new = np.copy(y)
        for i in range(P):
            ys = [y_a[i], y_b[i], y_c[i]]
            xs = [x_a[i], x_b[i], x_c[i]]
            best_i = np.argmin(ys)
            if ys[best_i] < y[i] or np.random.rand() < r_accept:
                x_new[i] = xs[best_i]
                y_new[i] = ys[best_i]
        x = x_new
        y = y_new

        writer.writerow([
            t + 1,
            *x_global_best,                            
            best_pdp * 1e15,                           # PDP (fJ)
            best_power,                                # power (W)
            best_delay * 1e12                          # delay (ps)
        ])
        print(f"best PDP round {t+1}: {y_global_best:.4e}")

print("\n completed")
print(f"min FOM: {y_global_best:.6e}")
print(f" best x:\n{x_global_best}")