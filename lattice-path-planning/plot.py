import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely import offset_curve

def setup_plot_style():
    """设置符合论文发表标准的绘图风格"""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.size'] = 0
    plt.rcParams['ytick.major.size'] = 0
    plt.rcParams['xtick.labelsize'] = 0
    plt.rcParams['ytick.labelsize'] = 0

def draw_wavy_principle():
    """图1：绘制波浪晶格生成原理 (3个子图)"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 模拟一个多边形
    poly_coords = [(2, 1), (8, 1), (9, 8), (1, 8)]
    poly = Polygon(poly_coords)
    x_poly, y_poly = poly.exterior.xy

    # 网格参数
    grid_spacing = 0.6
    amplitude = 0.3
    period_scale = 2.0
    
    # --- 子图 (a): 初始网格采样 ---
    ax = axes[0]
    ax.plot(x_poly, y_poly, 'k-', linewidth=2.5, label='Boundary')
    ax.fill(x_poly, y_poly, color='lightgray', alpha=0.2)
    
    xs = np.arange(0, 10, grid_spacing)
    ys = np.arange(0, 10, grid_spacing)
    xx, yy = np.meshgrid(xs, ys)
    
    # 绘制网格点
    ax.scatter(xx, yy, c='gray', s=10, alpha=0.4)
    ax.set_title("(a) Initial Grid Sampling", fontsize=14, pad=10)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(5, -1, "Regular grid points $P_{ij}$", ha='center', fontsize=11)

    # --- 子图 (b): 正弦扰动变换 ---
    ax = axes[1]
    ax.plot(x_poly, y_poly, 'k:', linewidth=1, alpha=0.5)
    
    # 变换公式
    xx_shifted = xx + amplitude * np.sin(yy * np.pi / period_scale)
    
    # 画出原来的点（淡色）和移动后的点（红色）
    mask = (xx > 1) & (xx < 9) & (yy > 1) & (yy < 9) # 只显示中间区域的箭头以免太乱
    ax.scatter(xx[mask], yy[mask], c='gray', s=5, alpha=0.2)
    ax.scatter(xx_shifted[mask], yy[mask], c='red', s=15, zorder=5)
    
    # 画箭头展示移动
    for i in range(len(xs)):
        for j in range(len(ys)):
            if 2 < xx[j,i] < 8 and 2 < yy[j,i] < 8 and i % 2 == 0: # 抽样画箭头
                ax.arrow(xx[j,i], yy[j,i], xx_shifted[j,i]-xx[j,i], 0, 
                         head_width=0.15, color='red', alpha=0.5, length_includes_head=True)

    ax.set_title("(b) Sinusoidal Modulation", fontsize=14, pad=10)
    ax.text(5, 9, r"$x' = x + A \cdot \sin(\frac{y \cdot \pi}{S})$", fontsize=12, 
            ha='center', bbox=dict(facecolor='white', alpha=0.9, edgecolor='red'))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(5, -1, "Coordinate Transformation", ha='center', fontsize=11)

    # --- 子图 (c): 裁剪与图构建 ---
    ax = axes[2]
    ax.plot(x_poly, y_poly, 'k-', linewidth=3)
    
    valid_x, valid_y = [], []
    for x_val, y_val in zip(xx_shifted.flatten(), yy.flatten()):
        if poly.contains(Point(x_val, y_val)):
            valid_x.append(x_val)
            valid_y.append(y_val)
    
    # 简单的临近连接模拟
    from scipy.spatial import cKDTree
    if valid_x:
        pts = np.column_stack((valid_x, valid_y))
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r=grid_spacing * 1.5)
        for i, j in pairs:
            p1, p2 = pts[i], pts[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#1f77b4', linewidth=1.2)
            
    ax.scatter(valid_x, valid_y, c='black', s=20, zorder=10)
    ax.set_title("(c) Graph Construction", fontsize=14, pad=10)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(5, -1, "Clipped & Connected Graph $G(V,E)$", ha='center', fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

def draw_hybrid_strategy():
    """图2：混合策略演示 (左边波浪，右边轮廓)"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # --- 左图：大区域 (Wavy Mode) ---
    ax1 = axes[0]
    circle_big = Point(5, 5).buffer(4.5) # 大圆
    x, y = circle_big.exterior.xy
    ax1.plot(x, y, 'k-', linewidth=3)
    ax1.fill(x, y, color='#e6f3ff')
    
    # 画波浪填充
    ys = np.arange(1, 9, 0.4)
    for y_line in ys:
        xs = np.linspace(0, 10, 200)
        wave = 0.3 * np.sin(xs * 2) # 波浪
        line_coords = np.column_stack((xs, y_line + wave))
        line = LineString(line_coords)
        clipped = line.intersection(circle_big)
        if not clipped.is_empty:
            if clipped.geom_type == 'LineString':
                lx, ly = clipped.xy
                ax1.plot(lx, ly, 'b-', linewidth=1.5)
            elif clipped.geom_type == 'MultiLineString':
                for part in clipped.geoms:
                    lx, ly = part.xy
                    ax1.plot(lx, ly, 'b-', linewidth=1.5)
                    
    ax1.set_title("Mode A: Wavy Lattice\n(Large Area / High Connectivity)", fontsize=14, weight='bold')
    ax1.text(5, 5, "Threshold Check:\nArea > 200mm²\nNodes > 20\nPASS", ha='center', va='center', 
             bbox=dict(boxstyle="round", fc="white", ec="green", alpha=0.9))
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)

    # --- 右图：小/细长区域 (Contour Mode) ---
    ax2 = axes[1]
    # 创建一个哑铃形状（中间细，两头大，导致碎片多或面积小）
    p1 = Point(3, 3).buffer(1.5)
    p2 = Point(7, 7).buffer(1.0)
    # 用一个细矩形连接
    link = Polygon([(3,2.5), (7,6.5), (7,7.5), (3,3.5)])
    complex_poly = p1.union(p2) # 这里为了演示，故意让它分离或很怪
    # 实际上我们画一个简单的窄条或者两个分离的小岛代表“组件过多”
    
    poly_a = Point(3, 5).buffer(1.2)
    poly_b = Point(7, 5).buffer(0.8) # 两个分离的小岛
    
    for p in [poly_a, poly_b]:
        x, y = p.exterior.xy
        ax2.plot(x, y, 'k-', linewidth=3)
        ax2.fill(x, y, color='#fff0e6')
        
        # 画同心圆 (Contour)
        for dist in [-0.3, -0.6, -0.9]:
            off = p.buffer(dist)
            if not off.is_empty:
                ox, oy = off.exterior.xy
                ax2.plot(ox, oy, color='#d62728', linewidth=2, linestyle='-')

    ax2.set_title("Mode B: Contour (Fail-safe)\n(Small Area / Fragmentation)", fontsize=14, weight='bold')
    ax2.text(5, 5, "Threshold Check:\nComponents > Limit\nOR Area < Min\nFALLBACK TRIGGERED", ha='center', va='center', 
             fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="red", alpha=0.9))
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    setup_plot_style()
    print("正在生成图1：波浪生成原理...")
    draw_wavy_principle()
    print("正在生成图2：混合策略演示...")
    draw_hybrid_strategy()