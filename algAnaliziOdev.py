import math
import time
import random
import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# -------------------------
# Geometry helpers
# -------------------------
def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def polar_angle(p0, p1):
    return math.atan2(p1[1] - p0[1], p1[0] - p0[0])


def dist2(p0, p1):
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    return dx * dx + dy * dy


# -------------------------
# Graham Scan (O(n log n))
# -------------------------
def graham_scan(points):
    n = len(points)
    if n <= 1:
        return points[:]
    if n == 2:
        return points[:] if points[0] != points[1] else [points[0]]

    pivot = min(points, key=lambda p: (p[1], p[0]))

    pts = [p for p in points if p != pivot]
    pts.sort(key=lambda p: (polar_angle(pivot, p), dist2(pivot, p)))

    filtered = []
    i = 0
    while i < len(pts):
        j = i
        far = pts[i]
        ang = polar_angle(pivot, pts[i])
        while j < len(pts) and abs(polar_angle(pivot, pts[j]) - ang) < 1e-12:
            if dist2(pivot, pts[j]) >= dist2(pivot, far):
                far = pts[j]
            j += 1
        filtered.append(far)
        i = j

    if len(filtered) == 1:
        return [pivot, filtered[0]]

    stack = [pivot, filtered[0], filtered[1]]
    for p in filtered[2:]:
        while len(stack) >= 2 and cross(stack[-2], stack[-1], p) <= 0:
            stack.pop()
        stack.append(p)

    return stack


# -------------------------
# Brute Force (Worst-case O(n^3))
# -------------------------
def brute_force_hull(points):
    n = len(points)
    if n <= 1:
        return points[:]
    if n == 2:
        return points[:] if points[0] != points[1] else [points[0]]

    edges = set()

    for i in range(n):
        p = points[i]
        for j in range(i + 1, n):
            q = points[j]

            pos = neg = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                r = points[k]
                c = cross(p, q, r)
                if c > 0:
                    pos += 1
                elif c < 0:
                    neg += 1
                if pos and neg:
                    break

            if not (pos and neg):
                edges.add(p)
                edges.add(q)

    hull_pts = list(edges)
    if len(hull_pts) <= 2:
        return hull_pts

    cx = sum(p[0] for p in hull_pts) / len(hull_pts)
    cy = sum(p[1] for p in hull_pts) / len(hull_pts)
    hull_pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    return hull_pts


# -------------------------
# GUI App
# -------------------------
class ConvexHullApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Convex Hull: Brute Force vs Graham Scan")
        self.root.geometry("1200x800")

        self.points = []
        self.hull = None
        self.hull_label = None

        self.brute_data = []   # list of (N, time)
        self.graham_data = []  # list of (N, time)

        self.view_mode = "points"  # "points" or "graph"

        ctrl = ttk.Frame(root, padding=10)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(ctrl, text="N (nokta sayısı):").pack(side=tk.LEFT)
        self.n_var = tk.StringVar(value="100")
        ttk.Entry(ctrl, textvariable=self.n_var, width=10).pack(side=tk.LEFT, padx=8)

        ttk.Button(ctrl, text="Rastgele Nokta Üret", command=self.generate_points).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Brute Force Hull (çiz + süre ölç)", command=self.run_bruteforce).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Graham Scan Hull (çiz + süre ölç)", command=self.run_graham).pack(side=tk.LEFT, padx=6)

        ttk.Separator(ctrl, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Button(ctrl, text="Grafiği Göster", command=self.show_graph).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Noktalara Geri Dön", command=self.show_points).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Grafik Verisini Temizle", command=self.clear_graph_data).pack(side=tk.LEFT, padx=6)

        self.status = tk.StringVar(value="Hazır.")
        ttk.Label(ctrl, textvariable=self.status).pack(side=tk.RIGHT)

        self.fig = Figure(figsize=(9.5, 6.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.draw_points()

    def get_n(self):
        try:
            n = int(self.n_var.get())
            if n <= 0:
                raise ValueError
            return n
        except Exception:
            messagebox.showerror("Hata", "N pozitif bir tamsayı olmalı.")
            return None

    def generate_points(self):
        n = self.get_n()
        if n is None:
            return
        self.points = [(random.random() * 1000, random.random() * 700) for _ in range(n)]
        self.hull = None
        self.hull_label = None
        self.status.set(f"{n} nokta üretildi.")
        self.view_mode = "points"
        self.draw_points()

    def draw_points(self):
        self.ax.clear()

        if self.points:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            self.ax.scatter(xs, ys, s=10)

        if self.hull and len(self.hull) >= 2:
            hx = [p[0] for p in self.hull] + [self.hull[0][0]]
            hy = [p[1] for p in self.hull] + [self.hull[0][1]]
            self.ax.plot(hx, hy, linewidth=2)
            if self.hull_label:
                self.ax.set_title(self.hull_label)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        # POINTS VIEW: equal aspect OK here
        self.ax.set_aspect("equal", adjustable="datalim")

        self.canvas.draw_idle()

    # ---------- Performance graph (robust) ----------
    def draw_performance_graph(self):
        self.ax.clear()

        # GRAPH VIEW: MUST disable equal aspect to avoid log+aspect issues
        self.ax.set_aspect("auto")

        def aggregate_by_n_avg(data):
            if not data:
                return []
            bucket = {}
            for n, t in data:
                if not isinstance(n, int) or n <= 0:
                    continue
                if t is None:
                    continue
                try:
                    t = float(t)
                except Exception:
                    continue
                if not math.isfinite(t) or t <= 0:
                    continue
                bucket.setdefault(n, []).append(t)

            out = [(n, sum(ts) / len(ts)) for n, ts in bucket.items() if ts]
            out.sort(key=lambda x: x[0])
            return out

        bd = aggregate_by_n_avg(self.brute_data)
        gd = aggregate_by_n_avg(self.graham_data)

        if not bd and not gd:
            self.ax.set_title("Performans Karşılaştırması")
            self.ax.set_xlabel("N (nokta sayısı)")
            self.ax.set_ylabel("Süre (saniye)")
            self.ax.text(
                0.5, 0.5,
                "Henüz geçerli veri yok.\nÖnce Brute/Graham çalıştır, sonra Grafiği Göster.",
                ha="center", va="center",
                transform=self.ax.transAxes
            )
            self.canvas.draw_idle()
            return

        # Plot series
        all_y = []
        if bd:
            bx, by = zip(*bd)
            self.ax.plot(bx, by, marker="o", label="Brute Force (avg)")
            all_y += list(by)
        if gd:
            gx, gy = zip(*gd)
            self.ax.plot(gx, gy, marker="o", label="Graham Scan (avg)")
            all_y += list(gy)

        self.ax.set_xlabel("N (nokta sayısı)")
        self.ax.set_title("Performans Karşılaştırması")
        self.ax.legend()

        # LOG scale only if safe
        if all_y and all(t > 0 and math.isfinite(t) for t in all_y):
            self.ax.set_yscale("log")
            self.ax.set_ylabel("Süre (saniye) - log")

            # Safe, finite ylim to prevent overflow in tick generation
            ymin = min(all_y)
            ymax = max(all_y)
            if ymin == ymax:
                ymin = ymin / 10.0
                ymax = ymax * 10.0
            else:
                ymin = ymin * 0.8
                ymax = ymax * 1.2

            # clamp to avoid extreme log ranges
            ymin = max(ymin, 1e-12)
            ymax = min(ymax, 1e12)

            self.ax.set_ylim(ymin, ymax)
        else:
            self.ax.set_ylabel("Süre (saniye)")

        self.canvas.draw_idle()

    # ---------- Actions ----------
    def run_bruteforce(self):
        if not self.points:
            self.generate_points()
            if not self.points:
                return

        n = len(self.points)
        self.status.set("Brute Force çalışıyor...")
        self.root.update_idletasks()

        t0 = time.perf_counter()
        hull = brute_force_hull(self.points)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        elapsed = max(elapsed, 1e-9)

        self.hull = hull
        self.hull_label = f"Brute Force Hull | N={n} | süre={elapsed:.6f} s"
        self.status.set(f"Brute Force bitti: {elapsed:.6f} s")

        self.brute_data.append((n, elapsed))

        if self.view_mode == "graph":
            self.draw_performance_graph()
        else:
            self.draw_points()

    def run_graham(self):
        if not self.points:
            self.generate_points()
            if not self.points:
                return

        n = len(self.points)
        self.status.set("Graham Scan çalışıyor...")
        self.root.update_idletasks()

        t0 = time.perf_counter()
        hull = graham_scan(self.points)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        elapsed = max(elapsed, 1e-9)

        self.hull = hull
        self.hull_label = f"Graham Scan Hull | N={n} | süre={elapsed:.6f} s"
        self.status.set(f"Graham Scan bitti: {elapsed:.6f} s")

        self.graham_data.append((n, elapsed))

        if self.view_mode == "graph":
            self.draw_performance_graph()
        else:
            self.draw_points()

    def show_graph(self):
        self.view_mode = "graph"
        self.draw_performance_graph()
        self.status.set("Grafik modu açık. (Log ölçek uygunsa kullanılır.)")

    def show_points(self):
        self.view_mode = "points"
        self.draw_points()
        self.status.set("Nokta modu açık.")

    def clear_graph_data(self):
        self.brute_data.clear()
        self.graham_data.clear()
        if self.view_mode == "graph":
            self.draw_performance_graph()
        else:
            self.draw_points()
        self.status.set("Grafik verisi temizlendi.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ConvexHullApp(root)
    root.mainloop()
