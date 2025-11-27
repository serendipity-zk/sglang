import numpy as np
import matplotlib.pyplot as plt

# === 模拟黑盒系统 ===
class MockSystem:
    def __init__(self, true_knee):
        self.true_knee = true_knee

    def get_attainment(self, rate):
        # Rate <= Knee: 100% Attainment
        if rate <= self.true_knee:
            return 1.0
        # Rate > Knee: 模拟排队效应导致的下降
        overload = rate / self.true_knee
        return 1.0 / (1.0 + 2.0 * (overload - 1.0)**1.5)

# === 核心算法：自适应三段式搜索 ===
def adaptive_search(system_func, estimate_rate, slo_target=0.99):
    history = [] # 记录 (rate, attainment, order)
    
    def probe(r):
        val = system_func(r)
        history.append((r, val, len(history)+1))
        return val

    # 1. 定界 (Bracketing)
    print(f"Searching... Start at {estimate_rate}")
    val = probe(estimate_rate)
    left, right = 0, 0
    
    if val >= slo_target: # 还在平原，向右冲
        left = estimate_rate
        curr = estimate_rate * 1.5
        while True:
            if probe(curr) < slo_target:
                right = curr; break
            left = curr; curr *= 1.5
    else: # 在坑里，向左撤
        right = estimate_rate
        curr = estimate_rate * 0.5
        while True:
            if probe(curr) >= slo_target:
                left = curr; break
            right = curr; curr *= 0.5
            
    # 2. 二分 (Binary Search)
    while (right - left) > (0.10 * left): # 精度 5%
        mid = (left + right) / 2
        if probe(mid) >= slo_target: left = mid
        else: right = mid
        
    # 3. 描尾 (Tail)
    knee = left
    # 如果1.2-1.5x 没有记录就多加一个
    if not any(h[0] > knee * 1.2 and h[0] < knee * 1.5 for h in history):
        probe(knee * 1.25)
        
    if not any(h[0] > knee * 1.5 and h[0] < knee * 2.0 for h in history):
        probe(knee * 1.60)
    # 如果0.7-1.0x 没有记录就多加一个
    if not any(h[0] > knee * 0.7 and h[0] < knee * 1.0 for h in history):
        probe(knee * 0.80)
    
    return history

# === 绘图主程序 ===
def run_demo():
    scenarios = [
        (1000, 1000, "Scenario A: Perfect Start"),
        (1000, 400,  "Scenario B: Under Estimate"),
        (1000, 2000, "Scenario C: Over Estimate")
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (knee, est, title) in zip(axes, scenarios):
        sys = MockSystem(knee)
        hist = adaptive_search(sys.get_attainment, est)
        
        # 画点和线
        rates = [h[0] for h in hist]
        vals = [h[1] for h in hist]
        orders = [h[2] for h in hist]
        colors = ['green' if v >= 0.99 else 'red' for v in vals]
        
        ax.set_title(f"{title}\nTotal Points: {len(hist)}")
        ax.scatter(rates, vals, c=colors, s=100, zorder=5) # 红绿点
        ax.plot(rates, vals, 'b--', alpha=0.3) # 连线路径
        
        # 标注顺序数字
        for r, v, o in zip(rates, vals, orders):
            ax.annotate(str(o), (r,v), xytext=(0,10), textcoords='offset points', ha='center', fontweight='bold')
            
        # 画真实曲线背景
        x_true = np.linspace(0, max(rates)*1.1, 100)
        y_true = [sys.get_attainment(x) for x in x_true]
        ax.plot(x_true, y_true, 'k-', alpha=0.2, label='True Curve')
        ax.axvline(knee, color='orange', linestyle=':', label='True Knee')
        ax.set_ylim(-0.05, 1.1)
        ax.legend()

    plt.tight_layout()
    plt.savefig("smart_perf.png")

if __name__ == "__main__":
    run_demo()