import math
import time
import random
import tkinter as tk
from tkinter import ttk, messagebox

import multiprocessing as mp

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# =========================================================
# 1) GEOMETRİ YARDIMCI FONKSİYONLARI
#    (Algoritmalarda tekrar tekrar kullanıyoruz)
# =========================================================
def cross(o, a, b):
    """
    o->a ve o->b vektörlerinin 2B cross product değeri.
    >0 ise sola dönüş, <0 ise sağa dönüş, 0 ise koliner (aynı doğru üzerinde).
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def polar_angle(p0, p1):
    """p0 referans alınarak p1 noktasının polar açısını döndürür (radyan)."""
    return math.atan2(p1[1] - p0[1], p1[0] - p0[0])


def dist2(p0, p1):
    """
    İki nokta arasındaki Öklid uzaklığının karesi.
    sqrt almadan karşılaştırma yapabildiğimiz için daha hızlıdır.
    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    return dx * dx + dy * dy


# =========================================================
# 2) GRAHAM SCAN (O(n log n))
#    Adımlar:
#    - Pivot seç (en düşük y, eşitse en düşük x)
#    - Noktaları polar açıya göre sırala
#    - Stack ile sağa dönüşleri (veya kolinerleri) ele
# =========================================================
def graham_scan(points):
    n = len(points)

    # Çok az nokta varsa hull zaten kendisi olur (basit durumlar)
    if n <= 1:
        return points[:]
    if n == 2:
        return points[:] if points[0] != points[1] else [points[0]]

    # Pivot: en aşağıdaki nokta (y küçük), eşitse x küçük
    pivot = min(points, key=lambda p: (p[1], p[0]))

    # Pivot hariç diğer noktaları al
    pts = [p for p in points if p != pivot]

    # Önce açıya göre, aynı açıysa pivot'a uzak olana göre sırala
    pts.sort(key=lambda p: (polar_angle(pivot, p), dist2(pivot, p)))

    # Aynı açıya sahip noktalar varsa sadece en uzaktakini tutuyoruz
    # (yakındaki nokta hull üzerinde olmaz, çizgiyi boşa şişirir)
    filtered = []
    i = 0
    while i < len(pts):
        j = i
        far = pts[i]
        ang = polar_angle(pivot, pts[i])

        # Aynı açıdaki grubu tarayıp en uzaktakini seç
        while j < len(pts) and abs(polar_angle(pivot, pts[j]) - ang) < 1e-12:
            if dist2(pivot, pts[j]) >= dist2(pivot, far):
                far = pts[j]
            j += 1

        filtered.append(far)
        i = j

    # Filtre sonrası sadece 1 nokta kaldıysa hull 2 noktadır
    if len(filtered) == 1:
        return [pivot, filtered[0]]

    # Stack başlangıcı (ilk 3 nokta ile)
    stack = [pivot, filtered[0], filtered[1]]

    # Diğer noktaları tek tek ekle
    for p in filtered[2:]:
        # Sağ dönüş / koliner durumunda üstteki noktayı çıkar (hull dışı)
        while len(stack) >= 2 and cross(stack[-2], stack[-1], p) <= 0:
            stack.pop()
        stack.append(p)

    return stack


# =========================================================
# 3) PARALEL BRUTE FORCE (ÇOK ÇEKİRDEK)
#    Brute force mantığı aynı:
#    - Her (i,j) çifti kenar olabilir mi?
#    - Tüm diğer noktalar aynı yarı düzlemde mi?
#    Optimizasyon:
#    - points yerine xs/ys listeleriyle çalışıyoruz (daha hızlı)
# =========================================================
def _bf_worker_fast(args):
    """
    Worker fonksiyonu:
    - Kendisine verilen (i,j) çiftlerini kontrol eder.
    - Eğer (i,j) bir hull kenarıysa, i ve j indexlerini set'e ekler.
    - Sonunda "hull üzerinde olma ihtimali yüksek" köşe indexlerini döndürür.
    """
    xs, ys, pairs = args
    n = len(xs)
    edges_idx = set()

    for (i, j) in pairs:
        px, py = xs[i], ys[i]
        qx, qy = xs[j], ys[j]

        # pq vektörü (cross hesabını hızlandırmak için)
        vx = qx - px
        vy = qy - py

        # pos/neg: Noktalar çizginin iki tarafına düşüyor mu?
        # İkisi de 1 olursa -> bu (i,j) kesin hull kenarı değildir
        pos = neg = 0

        for k in range(n):
            if k == i or k == j:
                continue

            rx, ry = xs[k], ys[k]

            # cross(p,q,r) işareti r'nin pq doğrusunun hangi tarafında kaldığını söyler
            c = vx * (ry - py) - vy * (rx - px)

            if c > 0:
                pos = 1
            elif c < 0:
                neg = 1

            # Hem + hem - gördüysek bu çizgi iki tarafı da kesiyor -> hull kenarı olamaz
            if pos and neg:
                break

        # Eğer tüm noktalar tek taraftaysa (pos^neg), (i,j) hull kenarı olabilir
        if not (pos and neg):
            edges_idx.add(i)
            edges_idx.add(j)

    return edges_idx


def brute_force_hull_parallel(points, processes=6, chunk_pairs=12000):
    """
    Paralel brute force convex hull:
    - Tüm (i,j) çiftlerini parçalara böl (chunk)
    - Her chunk'ı farklı çekirdekte çalıştır
    - Gelen aday köşe noktalarını birleştir
    - Çizim için noktaları merkez etrafında sıralayıp döndür

    Not: Burada "algoritmanın mantığı" aynı kalır (hala brute force).
    Sadece işi çok çekirdeğe bölüp sabit çarpanı düşürüyoruz.
    """
    n = len(points)

    # Basit durumlar
    if n <= 1:
        return points[:]
    if n == 2:
        return points[:] if points[0] != points[1] else [points[0]]

    # Noktaları x ve y listelerine ayırmak daha hızlı (tuple overhead azalıyor)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    # Süreç sayısı güvenliği
    processes = max(1, int(processes))

    # (i,j) çiftlerini chunk_pairs büyüklüğünde parçalara bölüp task listesi oluştur
    tasks = []
    pairs_chunk = []

    for i in range(n):
        for j in range(i + 1, n):
            pairs_chunk.append((i, j))
            if len(pairs_chunk) >= chunk_pairs:
                tasks.append((xs, ys, pairs_chunk))
                pairs_chunk = []
    if pairs_chunk:
        tasks.append((xs, ys, pairs_chunk))

    # Paralel çalıştırıp çıkan indexleri birleştiriyoruz
    edges_idx = set()
    with mp.Pool(processes=processes) as pool:
        # imap_unordered: sonuçlar biten işten geldikçe toparlanır (genelde daha hızlı hissettirir)
        for part in pool.imap_unordered(_bf_worker_fast, tasks, chunksize=1):
            edges_idx |= part

    # Indexlerden gerçek hull noktalarını çıkar
    hull_pts = [(xs[i], ys[i]) for i in edges_idx]
    if len(hull_pts) <= 2:
        return hull_pts

    # Çizim için noktaları saat yönünde sıralamak lazım:
    # Merkez (centroid) hesapla, atan2 ile açıya göre sırala
    cx = sum(p[0] for p in hull_pts) / len(hull_pts)
    cy = sum(p[1] for p in hull_pts) / len(hull_pts)
    hull_pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    return hull_pts


# =========================================================
# 4) GUI UYGULAMASI (Tkinter + Matplotlib)
#    - Nokta üretme
#    - Brute Force çalıştırma (paralel)
#    - Graham Scan çalıştırma
#    - Performans grafiği çizme
# =========================================================
class ConvexHullApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Convex Hull: Brute Force vs Graham Scan")
        self.root.geometry("1200x800")

        # Noktalar ve hull
        self.points = []
        self.hull = None
        self.hull_label = None

        # Performans ölçümleri (N, süre)
        self.brute_data = []
        self.graham_data = []

        # Görünüm modu: nokta çizimi veya performans grafiği
        self.view_mode = "points"  # "points" or "graph"

        # Üst kontrol paneli (butonlar, giriş alanı)
        ctrl = ttk.Frame(root, padding=10)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(ctrl, text="N (nokta sayısı):").pack(side=tk.LEFT)
        self.n_var = tk.StringVar(value="1000")
        ttk.Entry(ctrl, textvariable=self.n_var, width=10).pack(side=tk.LEFT, padx=8)

        ttk.Button(ctrl, text="Rastgele Nokta Üret", command=self.generate_points).pack(side=tk.LEFT, padx=6)

        ttk.Button(ctrl, text="Brute Force Hull (çiz + süre ölç)", command=self.run_bruteforce).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Graham Scan Hull (çiz + süre ölç)", command=self.run_graham).pack(side=tk.LEFT, padx=6)

        ttk.Separator(ctrl, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=10)

        ttk.Button(ctrl, text="Grafiği Göster", command=self.show_graph).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Noktalara Geri Dön", command=self.show_points).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Grafik Verisini Temizle", command=self.clear_graph_data).pack(side=tk.LEFT, padx=6)

        # Sağ tarafta kısa durum mesajı gösteriyoruz (hazır, çalışıyor, bitti vs.)
        self.status = tk.StringVar(value="Hazır.")
        ttk.Label(ctrl, textvariable=self.status).pack(side=tk.RIGHT)

        # Matplotlib figürü Tkinter içine gömüyoruz
        self.fig = Figure(figsize=(9.5, 6.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.draw_points()

    def get_n(self):
        """Textbox'tan N değerini alır ve geçerliyse int döndürür."""
        try:
            n = int(self.n_var.get())
            if n <= 0:
                raise ValueError
            return n
        except Exception:
            messagebox.showerror("Hata", "N pozitif bir tamsayı olmalı.")
            return None

    def generate_points(self):
        """N tane rastgele nokta üretir (0..1000, 0..700 aralığında)."""
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
        """Noktaları ve varsa hull çizgisini ekrana çizer."""
        self.ax.clear()

        # Noktaları çiz
        if self.points:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            self.ax.scatter(xs, ys, s=10)

        # Hull varsa polygon gibi kapatıp çiz
        if self.hull and len(self.hull) >= 2:
            hx = [p[0] for p in self.hull] + [self.hull[0][0]]
            hy = [p[1] for p in self.hull] + [self.hull[0][1]]
            self.ax.plot(hx, hy, linewidth=2)
            if self.hull_label:
                self.ax.set_title(self.hull_label)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        # Nokta çiziminde görüntü bozulmasın diye ölçeği eşit tutuyoruz
        self.ax.set_aspect("equal", adjustable="datalim")
        self.canvas.draw_idle()

    def draw_performance_graph(self):
        """Toplanan süre verilerine göre performans grafiğini çizer."""
        self.ax.clear()

        # Log ölçek + equal aspect çakışmasın diye grafikte auto yapıyoruz
        self.ax.set_aspect("auto")

        def aggregate_by_n_avg(data):
            """Aynı N için birden fazla ölçüm varsa ortalamasını alır."""
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

        self.ax.set_title("Performans Karşılaştırması")

        # Veri yoksa kullanıcıya bilgi yazısı göster
        if not bd and not gd:
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

        # Çizilecek y değerlerini toplayıp log'a uygun mu kontrol edeceğiz
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
        self.ax.legend()

        # Tüm süreler pozitifse log ölçek kullanıyoruz (farkı daha net gösteriyor)
        if all_y and all(t > 0 and math.isfinite(t) for t in all_y):
            self.ax.set_yscale("log")
            self.ax.set_ylabel("Süre (saniye) - log")

            # Grafik sıkışmasın diye biraz marj ekle
            ymin = min(all_y) * 0.8
            ymax = max(all_y) * 1.2

            ymin = max(ymin, 1e-12)
            ymax = min(ymax, 1e12)

            # Limitler aynı çıkarsa düzelt
            if ymin == ymax:
                ymin /= 10.0
                ymax *= 10.0

            self.ax.set_ylim(ymin, ymax)
        else:
            self.ax.set_ylabel("Süre (saniye)")

        self.canvas.draw_idle()

    def run_bruteforce(self):
        """Brute Force (paralel/optimize) çalıştırır, süreyi ölçer ve sonucu çizer."""
        if not self.points:
            self.generate_points()
            if not self.points:
                return

        n = len(self.points)

        # Brute Force çok pahalı olduğu için çok büyük N’de kullanıcıyı uyarıyoruz
        if n > 15000:
            ok = messagebox.askyesno(
                "Uyarı",
                f"N={n} için Brute Force çok uzun sürebilir (O(n^3)).\n"
                "Devam etmek istiyor musun?"
            )
            if not ok:
                self.status.set("Brute Force iptal edildi (N çok büyük).")
                return

        self.status.set("Brute Force (çok çekirdek, optimize) çalışıyor...")
        self.root.update_idletasks()

        # Zaman ölçümü (perf_counter daha hassas)
        t0 = time.perf_counter()

        # İyileştirilmiş paralel brute force çalıştırma
        hull = brute_force_hull_parallel(self.points, processes=6, chunk_pairs=12000)

        t1 = time.perf_counter()
        elapsed = max(t1 - t0, 1e-9)

        # Sonucu kaydet ve ekrana bas
        self.hull = hull
        self.hull_label = f"Brute Force Hull | N={n} | süre={elapsed:.6f} s"
        self.status.set(f"Brute Force bitti: {elapsed:.6f} s")

        # Performans datasına ekle
        self.brute_data.append((n, elapsed))

        # Hangi moddaysak ona göre çiz
        if self.view_mode == "graph":
            self.draw_performance_graph()
        else:
            self.draw_points()

    def run_graham(self):
        """Graham Scan çalıştırır, süreyi ölçer ve sonucu çizer."""
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

        elapsed = max(t1 - t0, 1e-9)

        self.hull = hull
        self.hull_label = f"Graham Scan Hull | N={n} | süre={elapsed:.6f} s"
        self.status.set(f"Graham Scan bitti: {elapsed:.6f} s")

        self.graham_data.append((n, elapsed))

        if self.view_mode == "graph":
            self.draw_performance_graph()
        else:
            self.draw_points()

    def show_graph(self):
        """Grafik moduna geçer."""
        self.view_mode = "graph"
        self.draw_performance_graph()
        self.status.set("Grafik modu açık. (Log ölçek uygunsa kullanılır.)")

    def show_points(self):
        """Nokta/hull görüntüsüne geri döner."""
        self.view_mode = "points"
        self.draw_points()
        self.status.set("Nokta modu açık.")

    def clear_graph_data(self):
        """Toplanan performans verilerini temizler."""
        self.brute_data.clear()
        self.graham_data.clear()

        if self.view_mode == "graph":
            self.draw_performance_graph()
        else:
            self.draw_points()

        self.status.set("Grafik verisi temizlendi.")


# =========================================================
# 5) PROGRAM BAŞLANGICI
#    Windows'ta multiprocessing sorunsuz çalışsın diye:
#    - freeze_support()
#    - spawn method
# =========================================================
if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)

    root = tk.Tk()
    app = ConvexHullApp(root)
    root.mainloop()
