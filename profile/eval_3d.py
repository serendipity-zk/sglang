#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a 3D trilinear grid with optional ONLINE local correction (bias-only),
and visualize the deviation (residual) distribution.

Default parameters: conservative, well-performing (α=0.6, k=64, r=0.30, h=0.20, half_life=50, W0=4).

Residual r = pred - real (ms).
We report signed percentiles: +p99, +p90, +p50, -p50, -p90, -p99.

Usage:
  python eval_grid3d_with_local_corr.py grid3d.json data.csv
  # customize correction:
  python eval_grid3d_with_local_corr.py grid3d.json data.csv --alpha 0.65 --W0 3 --plot resid.png
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- Trilinear interpolation ----------
def trilinear_predict(x, y, z, X, Y, Z, G):
    ix = np.searchsorted(X, x, side="right") - 1
    iy = np.searchsorted(Y, y, side="right") - 1
    iz = np.searchsorted(Z, z, side="right") - 1
    ix = np.clip(ix, 0, len(X) - 2)
    iy = np.clip(iy, 0, len(Y) - 2)
    iz = np.clip(iz, 0, len(Z) - 2)
    x0, x1 = X[ix], X[ix + 1]; y0, y1 = Y[iy], Y[iy + 1]; z0, z1 = Z[iz], Z[iz + 1]
    tx = np.divide(x - x0, x1 - x0, out=np.zeros_like(x), where=(x1 > x0))
    ty = np.divide(y - y0, y1 - y0, out=np.zeros_like(y), where=(y1 > y0))
    tz = np.divide(z - z0, z1 - z0, out=np.zeros_like(z), where=(z1 > z0))
    g000 = G[ix, iy, iz]; g100 = G[ix+1, iy, iz]; g010 = G[ix, iy+1, iz]; g110 = G[ix+1, iy+1, iz]
    g001 = G[ix, iy, iz+1]; g101 = G[ix+1, iy, iz+1]; g011 = G[ix, iy+1, iz+1]; g111 = G[ix+1, iy+1, iz+1]
    return ((1-tx)*(1-ty)*(1-tz)*g000 + tx*(1-ty)*(1-tz)*g100 +
            (1-tx)*ty*(1-tz)*g010 + tx*ty*(1-tz)*g110 +
            (1-tx)*(1-ty)*tz*g001 + tx*(1-ty)*tz*g101 +
            (1-tx)*ty*tz*g011 + tx*ty*tz*g111)

# ---------- Metrics ----------
def metrics(y, yhat):
    resid = yhat - y
    return float(np.sqrt(np.mean(resid ** 2))), float(np.mean(np.abs(resid))), resid
# ---------- Online bias-only corrector ----------
class BiasLocalCorrector:
    def __init__(self, buffer_size=10000, k=64, radius=0.30, bandwidth=0.20,
                 half_life=50.0, alpha=0.6, W0=4.0, max_correction=np.inf,
                 x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1):
        self.N, self.k, self.radius, self.h2 = int(buffer_size), int(k), radius, bandwidth**2
        self.tau, self.alpha, self.W0, self.max_corr = half_life / np.log(2.0), alpha, W0, max_correction
        self.xmin, self.xmax = x_min, x_max; self.ymin, self.ymax = y_min, y_max; self.zmin, self.zmax = z_min, z_max
        self.xspan, self.yspan, self.zspan = max(x_max-x_min,1e-12), max(y_max-y_min,1e-12), max(z_max-z_min,1e-12)
        self.buf_x = np.empty(self.N, np.float32); self.buf_y = np.empty(self.N, np.float32)
        self.buf_z = np.empty(self.N, np.float32); self.buf_r = np.empty(self.N, np.float32)
        self.buf_t = np.empty(self.N, np.int32); self.size, self.head, self.t = 0, 0, 0

    def _norm(self, x, y, z):
        return (x-self.xmin)/self.xspan, (y-self.ymin)/self.yspan, (z-self.zmin)/self.zspan

    def update(self, x, y, z, residual):
        xn, yn, zn = self._norm(x, y, z)
        self.buf_x[self.head], self.buf_y[self.head], self.buf_z[self.head] = xn, yn, zn
        self.buf_r[self.head], self.buf_t[self.head] = residual, self.t
        self.head = (self.head + 1) % self.N; self.size = min(self.size+1, self.N); self.t += 1

    def correction(self, x, y, z):
        if self.size == 0: return 0.0, 0
        xn, yn, zn = self._norm(x, y, z)
        Xb, Yb, Zb, Rb, Tb = self.buf_x[:self.size], self.buf_y[:self.size], self.buf_z[:self.size], self.buf_r[:self.size], self.buf_t[:self.size]
        d2 = (Xb-xn)**2 + (Yb-yn)**2 + (Zb-zn)**2
        mask = d2 <= self.radius**2
        if not np.any(mask): return 0.0, 0
        d2, Rb, Tb = d2[mask], Rb[mask], Tb[mask]
        if d2.size > self.k:
            idx = np.argpartition(d2, self.k)[:self.k]
            d2, Rb, Tb = d2[idx], Rb[idx], Tb[idx]
        w = np.exp(-d2/(2*self.h2)) * np.exp(-(self.t-Tb)/self.tau)
        Wsum = float(w.sum()); 
        if Wsum <= 1e-9: return 0.0, len(Rb)
        local_bias = float(np.sum(w*Rb)/Wsum)
        alpha_eff = self.alpha * (Wsum / (Wsum + self.W0))
        return float(np.clip(alpha_eff * local_bias, -self.max_corr, self.max_corr)), len(Rb)

# ---------- Helpers ----------
def signed_percentiles(resid):
    p01, p10, p50, p90, p99 = np.percentile(resid, [1,10,50,90,99])
    return {"-p99":p01,"-p90":p10,"-p50":min(p50,0)," +p50":max(p50,0),"+p90":p90,"+p99":p99}

def maybe_plot(resid, out, bins=100, title="Residual Distribution"):
    if not out:
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Compute robust bounds to avoid extreme outliers stretching the axis
    p_lo, p_hi = np.percentile(resid, [0.5, 99.5])
    pad = 0.1 * (p_hi - p_lo)
    x_min, x_max = p_lo - pad, p_hi + pad
    resid_clipped = resid[(resid >= x_min) & (resid <= x_max)]
    
    # Histogram and lines
    ax.hist(resid_clipped, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Residual (pred - real) [ms]")
    ax.set_ylabel("Count")
    ax.set_title(title)
    
    # Draw percentile markers within the same range
    for v, lbl in zip(np.percentile(resid, [1, 10, 50, 90, 99]), 
                      ["-p99", "-p90", "p50", "+p90", "+p99"]):
        if x_min <= v <= x_max:
            ax.axvline(v, ls="--", lw=1.2, color="r")
            ax.text(v, ax.get_ylim()[1]*0.95, lbl, rotation=90, va="top", ha="right", color="r")
    
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Evaluate 3D grid with online bias correction + residual plot.")
    ap.add_argument("model_json")
    ap.add_argument("csv")
    ap.add_argument("--local-corr", choices=["none","bias"], default="bias")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--radius", type=float, default=0.30)
    ap.add_argument("--bandwidth", type=float, default=0.20)
    ap.add_argument("--half-life", type=float, default=50.0)
    ap.add_argument("--W0", type=float, default=4.0)
    ap.add_argument("--buffer-size", type=int, default=10000)
    ap.add_argument("--max-correction", type=float, default=1e9)
    ap.add_argument("--print-samples", type=int, default=10)
    ap.add_argument("--plot", type=str, default="")
    args = ap.parse_args()

    model = json.loads(Path(args.model_json).read_text())
    df = pd.read_csv(args.csv)
    xcol,ycol,zcol,pcol = model["axes"].values()
    Xk,Yk,Zk,G = map(np.array,[model["knots"]["X_knots"],model["knots"]["Y_knots"],model["knots"]["Z_knots"],model["grid"]])
    x,y,z,p = [df[c].to_numpy(float) for c in [xcol,ycol,zcol,pcol]]
    p_grid = trilinear_predict(x,y,z,Xk,Yk,Zk,G)

    if args.local_corr == "none":
        rmse,mae,resid = metrics(p,p_grid)
        print(f"RMSE={rmse:.4f}, MAE={mae:.4f} | local_corr=none")
    else:
        lc = BiasLocalCorrector(buffer_size=args.buffer_size,k=args.k,radius=args.radius,
                                bandwidth=args.bandwidth,half_life=args.half_life,alpha=args.alpha,
                                W0=args.W0,max_correction=args.max_correction,
                                x_min=Xk.min(),x_max=Xk.max(),y_min=Yk.min(),y_max=Yk.max(),
                                z_min=Zk.min(),z_max=Zk.max())
        p_hat = np.empty_like(p)
        for i in range(len(p)):
            corr,_ = lc.correction(x[i],y[i],z[i])
            p_hat[i] = p_grid[i] + corr
            lc.update(x[i],y[i],z[i],p[i]-p_grid[i])
        rmse,mae,resid = metrics(p,p_hat)
        print(f"RMSE={rmse:.4f}, MAE={mae:.4f} | local_corr=bias (α={args.alpha}, k={args.k}, r={args.radius}, h={args.bandwidth}, W0={args.W0})")

    sp = signed_percentiles(resid)
    print("\nSigned residual percentiles (ms):")
    print(f"+p99={sp['+p99']:.4f}, +p90={sp['+p90']:.4f}, +p50={sp[' +p50']:.4f} | "
          f"-p50={sp['-p50']:.4f}, -p90={sp['-p90']:.4f}, -p99={sp['-p99']:.4f}")
    maybe_plot(resid,args.plot,title=f"local_corr={args.local_corr}")

    for i in range(min(args.print_samples,len(p))):
        print(f"({x[i]:.0f}, {y[i]:.0f}, {z[i]:.0f}) → pred={p[i]+resid[i]:.3f} | real={p[i]:.3f} | resid={resid[i]:+.3f}")

if __name__ == "__main__":
    main()
