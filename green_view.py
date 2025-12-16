# green_view.py (Frame/Screen 버전)
#
# - Toplevel 제거 → GreenViewScreen(tk.Frame)
# - STOP 버튼으로 on_back() 호출(보통 DistanceScreen으로 복귀)
# - 클릭으로 pin_EN 선택(그린 내부에서만)
# - 회전 B안(각도 기준: 현재→center), 기준점 center_EN
# - entry_distance/길이/폭/엔트리 삼각형은 pin_EN 기준
# - entry_display <= 20이면 "그린 근처입니다" 1초 표시 후 자동 복귀
# - 단독 실행 코드 없음
#
# [반영]
# - DistanceScreen에서 전달된 selected_green(L/R) 표시
# - 그린 contour를 더 부드럽게: (각도 정렬) + Chaikin 스무딩(iterations=2)

import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable

import tkinter as tk
from PIL import Image, ImageTk

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

from config import LCD_WIDTH, LCD_HEIGHT
from setting import app_settings


def make_transformer(lat0: float, lon0: float) -> Optional[Transformer]:
    if np.isnan(lat0) or np.isnan(lon0):
        return None
    wgs84 = CRS.from_epsg(4326)
    aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m"
    )
    return Transformer.from_crs(wgs84, aeqd, always_xy=True)


def dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def segment_ray_intersection(
    A: Tuple[float, float], B: Tuple[float, float],
    P: Tuple[float, float], d: Tuple[float, float],
) -> Optional[Tuple[Tuple[float, float], float]]:
    Ax, Ay = A
    Bx, By = B
    Px, Py = P
    dx, dy = d

    rx = Bx - Ax
    ry = By - Ay

    denom = rx * dy - ry * dx
    if abs(denom) < 1e-9:
        return None

    t_num = (Px - Ax) * ry - (Py - Ay) * rx
    s_num = (Px - Ax) * dy - (Py - Ay) * dx

    s = s_num / denom
    t = t_num / denom

    if s < 0.0 or s > 1.0 or t < 0.0:
        return None

    ix = Ax + s * rx
    iy = Ay + s * ry
    return (ix, iy), t


def segment_line_intersection(
    A: Tuple[float, float], B: Tuple[float, float],
    P: Tuple[float, float], d: Tuple[float, float],
) -> Optional[Tuple[Tuple[float, float], float]]:
    Ax, Ay = A
    Bx, By = B
    Px, Py = P
    dx, dy = d

    rx = Bx - Ax
    ry = By - Ay

    denom = rx * dy - ry * dx
    if abs(denom) < 1e-9:
        return None

    t_num = (Px - Ax) * ry - (Py - Ay) * rx
    s_num = (Px - Ax) * dy - (Py - Ay) * dx

    s = s_num / denom
    t = t_num / denom

    if s < 0.0 or s > 1.0:
        return None

    ix = Ax + s * rx
    iy = Ay + s * ry
    return (ix, iy), t


def is_point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            denom = (y2 - y1)
            if abs(denom) < 1e-12:
                continue
            x_int = (x2 - x1) * (y - y1) / denom + x1
            if x_int > x:
                inside = not inside
    return inside


def sort_points_by_angle(points: List[Tuple[float, float]], center: Tuple[float, float]) -> List[Tuple[float, float]]:
    cx, cy = center
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def chaikin_smooth_closed(points: List[Tuple[float, float]], iterations: int = 2) -> List[Tuple[float, float]]:
    """
    Chaikin corner-cutting for a closed polyline.
    - iterations=2 적용
    """
    if len(points) < 3:
        return points

    pts = points[:]
    for _ in range(iterations):
        new_pts: List[Tuple[float, float]] = []
        n = len(pts)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            new_pts.append(q)
            new_pts.append(r)
        pts = new_pts
    return pts


class GreenViewScreen(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        on_back: Optional[Callable[[], None]] = None,
    ):
        super().__init__(master, width=LCD_WIDTH, height=LCD_HEIGHT, bg="black")
        self.pack_propagate(False)

        self.on_back = on_back

        self.base_dir = Path(__file__).parent
        self.asset_dir = self.base_dir / "assets_image"
        self.images: Dict[Tuple[str, Tuple[int, int]], ImageTk.PhotoImage] = {}

        self.canvas = tk.Canvas(self, width=LCD_WIDTH, height=LCD_HEIGHT, highlightthickness=0, bg="black")
        self.canvas.pack()

        self.stop_button = tk.Button(self, text="STOP", command=self._on_stop)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # context
        self.parent_window: Optional[tk.Misc] = None
        self.hole_row: Optional[pd.Series] = None
        self.gc_center_lat: float = float("nan")
        self.gc_center_lng: float = float("nan")
        self.cur_lat: float = float("nan")
        self.cur_lng: float = float("nan")
        self.selected_green: str = "L"

        # transformer
        self.tf: Optional[Transformer] = None
        self.E_cur: float = 0.0
        self.N_cur: float = 0.0

        # green data
        self.center_EN: Tuple[float, float] = (0.0, 0.0)
        self.center_alt: float = 0.0
        self.green_points: List[Tuple[float, float]] = []
        self.pin_EN: Tuple[float, float] = (0.0, 0.0)

        # inverse transform cache
        self._last_cos = 1.0
        self._last_sin = 0.0
        self._last_scale = 1.0
        self._last_cx = LCD_WIDTH / 2
        self._last_cy = LCD_HEIGHT / 2

        self._after_id: Optional[str] = None

    # ---------- public ---------- #

    def set_context(
        self,
        parent_window: tk.Misc,
        hole_row: pd.Series,
        gc_center_lat: float,
        gc_center_lng: float,
        cur_lat: float,
        cur_lng: float,
        selected_green: str,
    ):
        self.parent_window = parent_window
        self.hole_row = hole_row
        self.gc_center_lat = gc_center_lat
        self.gc_center_lng = gc_center_lng
        self.cur_lat = cur_lat
        self.cur_lng = cur_lng
        self.selected_green = selected_green

        self.tf = make_transformer(self.gc_center_lat, self.gc_center_lng)
        if self.tf is None:
            print("[GreenViewScreen] transformer 생성 실패")
            return

        self.E_cur, self.N_cur = self.tf.transform(self.cur_lng, self.cur_lat)

        self.center_EN, self.center_alt = self._get_green_center_and_alt()
        self.green_points = self._get_green_points()

        self.pin_EN = (self.center_EN[0], self.center_EN[1])  # 기본 pin = center

    def start(self):
        self.stop()
        self._compute_geometry_and_draw()
        self._after_id = self.after(1000, self._auto_update_loop)

    def stop(self):
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    # ---------- loop ---------- #

    def _update_current_position(self):
        if self.parent_window is None:
            return
        lat = getattr(self.parent_window, "latitude", self.cur_lat)
        lng = getattr(self.parent_window, "longitude", self.cur_lng)
        self.cur_lat = lat
        self.cur_lng = lng
        if self.tf is not None:
            self.E_cur, self.N_cur = self.tf.transform(self.cur_lng, self.cur_lat)

    def _auto_update_loop(self):
        self._update_current_position()
        self._compute_geometry_and_draw()
        self._after_id = self.after(1000, self._auto_update_loop)

    # ---------- data ---------- #

    def _get_EN_point(self, label: str) -> Optional[Tuple[float, float]]:
        assert self.hole_row is not None
        e = self.hole_row.get(f"{label}_E", np.nan)
        n = self.hole_row.get(f"{label}_N", np.nan)
        if pd.isna(e) or pd.isna(n):
            return None
        return float(e), float(n)

    def _get_green_center_and_alt(self) -> Tuple[Tuple[float, float], float]:
        label = "LG" if self.selected_green == "L" else "RG"
        p = self._get_EN_point(label)
        if p is None:
            pts = self._get_green_points()
            if pts:
                xs = [q[0] for q in pts]
                ys = [q[1] for q in pts]
                p = (sum(xs) / len(xs), sum(ys) / len(ys))
            else:
                p = (self.E_cur, self.N_cur)

        assert self.hole_row is not None
        a = self.hole_row.get("LG_dAlt" if self.selected_green == "L" else "RG_dAlt", np.nan)
        if pd.isna(a):
            a = 0.0
        return (float(p[0]), float(p[1])), float(a)

    def _get_green_points(self) -> List[Tuple[float, float]]:
        assert self.hole_row is not None
        pts: List[Tuple[float, float]] = []
        prefix = "LG" if self.selected_green == "L" else "RG"
        for k in range(1, 21):
            p = self._get_EN_point(f"{prefix}{k}")
            if p is not None:
                pts.append(p)
        return pts

    def _check_rg_exists(self) -> bool:
        if self.hole_row is None:
            return False
        rg_e = self.hole_row.get("RG_E", np.nan)
        rg_n = self.hole_row.get("RG_N", np.nan)
        return not (pd.isna(rg_e) or pd.isna(rg_n))

    # ---------- images ---------- #

    def load_image(self, filename: str, size: Optional[Tuple[int, int]] = None):
        key = (filename, size if size is not None else (-1, -1))
        if key in self.images:
            return self.images[key]
        path = self.asset_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"[GreenView] 이미지 없음: {path}")
        img = Image.open(path).convert("RGBA")
        if size is not None:
            img = img.resize(size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.images[key] = photo
        return photo

    # ---------- L/R indicator ---------- #

    def _draw_green_indicator(self):
        """
        DistanceScreen에서 전달된 selected_green(L/R)을 표시한다.
        RG가 없는 홀은 의미상 L만 표시.
        """
        has_rg = self._check_rg_exists()
        if not has_rg:
            return

        label = self.selected_green if has_rg else "L"

        # 표시 위치/크기 (필요 시 조정)
        cx, cy = 105, 145
        r = 22
        self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill="#2E7D32", outline="#1B5E20", width=3
        )
        self.canvas.create_text(
            cx, cy, text=label,
            fill="white", font=("Helvetica", 18, "bold")
        )

    # ---------- click pin ---------- #

    def _on_canvas_click(self, event):
        # STOP 버튼 클릭 영역은 버튼 위젯이 처리
        if self._last_scale <= 0:
            return

        sx, sy = float(event.x), float(event.y)

        x_rot = (sx - self._last_cx) / self._last_scale
        y_rot = -(sy - self._last_cy) / self._last_scale

        cos_t = self._last_cos
        sin_t = self._last_sin
        x_rel = x_rot * cos_t + y_rot * sin_t
        y_rel = -x_rot * sin_t + y_rot * cos_t

        Cx, Cy = self.center_EN
        x_world = x_rel + Cx
        y_world = y_rel + Cy

        if not is_point_in_polygon(x_world, y_world, self.green_points):
            return

        self.pin_EN = (x_world, y_world)
        self._compute_geometry_and_draw()

    # ---------- draw ---------- #

    def _compute_geometry_and_draw(self):
        self.canvas.delete("all")
        try:
            bg = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
            self.canvas.create_image(0, 0, anchor="nw", image=bg)
        except FileNotFoundError:
            pass

        # L/R 표시 (배경 위)
        self._draw_green_indicator()

        if not self.green_points:
            return

        # world -> rel(center)
        Cx, Cy = self.center_EN
        G_rel = [(x - Cx, y - Cy) for (x, y) in self.green_points]
        cur_rel = (self.E_cur - Cx, self.N_cur - Cy)
        pin_rel = (self.pin_EN[0] - Cx, self.pin_EN[1] - Cy)

        # B안 회전각: 현재→center가 12시
        v = (-cur_rel[0], -cur_rel[1])
        angle_v = 0.0 if v == (0.0, 0.0) else math.atan2(v[1], v[0])
        theta = (math.pi / 2) - angle_v
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        self._last_cos = cos_t
        self._last_sin = sin_t

        def rot(p):
            x, y = p
            return (x * cos_t - y * sin_t, x * sin_t + y * cos_t)

        # 회전 좌표계로 변환
        G_rot_raw = [rot(p) for p in G_rel]
        cur_rot = rot(cur_rel)
        pin_rot = rot(pin_rel)

        # contour 부드럽게: 각도 정렬 + Chaikin(iterations=2)
        G_rot_sorted = sort_points_by_angle(G_rot_raw, center=(0.0, 0.0))
        G_rot = chaikin_smooth_closed(G_rot_sorted, iterations=2)

        # scale
        xs = [p[0] for p in G_rot]
        ys = [p[1] for p in G_rot]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        scale = 1.0 if width <= 0 or height <= 0 else min(234.0 / width, 234.0 / height)

        cx = LCD_WIDTH / 2
        cy = LCD_HEIGHT / 2
        self._last_scale = scale
        self._last_cx = cx
        self._last_cy = cy

        def to_screen(p):
            x, y = p
            return (cx + x * scale, cy - y * scale)

        # entry via ray (cur->pin)
        P0 = cur_rot
        Pin = pin_rot
        dir_vec = (Pin[0] - P0[0], Pin[1] - P0[1])
        norm = math.hypot(dir_vec[0], dir_vec[1])
        dir_vec = (0.0, 1.0) if norm == 0 else (dir_vec[0] / norm, dir_vec[1] / norm)

        intersections: List[Tuple[Tuple[float, float], float]] = []
        for i in range(len(G_rot)):
            A = G_rot[i]
            B = G_rot[(i + 1) % len(G_rot)]
            res = segment_ray_intersection(A, B, P0, dir_vec)
            if res is not None:
                intersections.append(res)

        if len(intersections) >= 2:
            intersections.sort(key=lambda x: x[1])
            p_front = intersections[0][0]
            p_back = intersections[-1][0]
        elif len(intersections) == 1:
            p_front = intersections[0][0]
            p_back = intersections[0][0]
        else:
            dists = [(dist(Pin, p), p) for p in G_rot]
            dists.sort(key=lambda x: x[0])
            p_front = dists[0][1]
            p_back = dists[-1][1]

        flat_entry = dist(P0, p_front)
        flat_center_to_pin = dist(P0, Pin)

        # entry_distance same logic as measure_distance
        parent_alt = getattr(self.parent_window, "altitude", 0.0) if self.parent_window else 0.0
        alt_offset = getattr(self.parent_window, "alt_offset", 0.0) if self.parent_window else 0.0
        current_alt = parent_alt - alt_offset
        diff_h = self.center_alt - current_alt

        dist_mode = getattr(app_settings, "dist_mode", "직선")
        if dist_mode == "보정":
            eDist = flat_center_to_pin + diff_h
            eDist_clamped = 80.0 if eDist < 80 else 220.0 if eDist > 220 else eDist
            landing = -0.11 * eDist_clamped + 64.0
            landing_rad = math.radians(landing) if landing != 0 else 1e-6
            extended = diff_h / math.tan(landing_rad)
            entry_distance = flat_entry + extended
        else:
            entry_distance = math.sqrt(flat_entry * flat_entry + diff_h * diff_h)

        unit = getattr(app_settings, "unit", "M")
        conv = 1.0 if unit == "M" else 1.09361
        entry_display = int(round(entry_distance * conv))

        # near-green auto return
        if entry_display <= 20:
            self.canvas.create_text(LCD_WIDTH // 2, LCD_HEIGHT // 2, text="그린 근처입니다",
                                    fill="white", font=("Helvetica", 30, "bold"))
            self.after(1000, self._on_stop)
            return

        # length/width via lines through pin
        a = (p_back[0] - p_front[0], p_back[1] - p_front[1])
        a_norm = math.hypot(a[0], a[1])
        if a_norm == 0:
            green_pin_length = 0.0
            width_val = 0.0
            len_p1 = len_p2 = Pin
            p3 = p4 = Pin
        else:
            a_dir = (a[0] / a_norm, a[1] / a_norm)
            perp = (-a_dir[1], a_dir[0])

            length_inters: List[Tuple[Tuple[float, float], float]] = []
            width_inters: List[Tuple[Tuple[float, float], float]] = []

            for i in range(len(G_rot)):
                A = G_rot[i]
                B = G_rot[(i + 1) % len(G_rot)]
                r1 = segment_line_intersection(A, B, Pin, a_dir)
                if r1 is not None:
                    length_inters.append(r1)
                r2 = segment_line_intersection(A, B, Pin, perp)
                if r2 is not None:
                    width_inters.append(r2)

            if len(length_inters) >= 2:
                length_inters.sort(key=lambda x: x[1])
                len_p1 = length_inters[0][0]
                len_p2 = length_inters[-1][0]
            else:
                len_p1, len_p2 = p_front, p_back
            green_pin_length = dist(len_p1, len_p2)

            if len(width_inters) >= 2:
                width_inters.sort(key=lambda x: x[1])
                p3 = width_inters[0][0]
                p4 = width_inters[-1][0]
                width_val = dist(p3, p4)
            else:
                p3 = p4 = Pin
                width_val = 0.0

        length_display = int(round(green_pin_length * conv))
        width_display = int(round(width_val * conv))

        # draw contour (smooth)
        contour = []
        for p in G_rot:
            sx, sy = to_screen(p)
            contour.extend([sx, sy])
        self.canvas.create_polygon(contour, fill="#90EE90", outline="white", width=3)

        # pin dot
        pin_s = to_screen(pin_rot)
        self.canvas.create_oval(pin_s[0] - 4, pin_s[1] - 4, pin_s[0] + 4, pin_s[1] + 4,
                                fill="red", outline="red")

        # cross lines
        l1 = to_screen(len_p1)
        l2 = to_screen(len_p2)
        self.canvas.create_line(l1[0], l1[1], l2[0], l2[1], fill="red", dash=(4, 4), width=2)

        w1 = to_screen(p3)
        w2 = to_screen(p4)
        self.canvas.create_line(w1[0], w1[1], w2[0], w2[1], fill="red", dash=(4, 4), width=2)

        # entry triangle
        pf_s = to_screen(p_front)
        tip = pf_s
        dir_tri = (pin_s[0] - tip[0], pin_s[1] - tip[1])
        ntri = math.hypot(dir_tri[0], dir_tri[1]) or 1.0
        dtx, dty = dir_tri[0] / ntri, dir_tri[1] / ntri
        pxv, pyv = -dty, dtx
        tri_len, tri_width = 18, 16
        base_center = (tip[0] - dtx * tri_len, tip[1] - dty * tri_len)
        left = (base_center[0] + pxv * (tri_width / 2), base_center[1] + pyv * (tri_width / 2))
        right = (base_center[0] - pxv * (tri_width / 2), base_center[1] - pyv * (tri_width / 2))
        self.canvas.create_polygon([tip[0], tip[1], left[0], left[1], right[0], right[1]], fill="red", outline="red")

        # entry text (fixed HUD position)
        entry_text_x = cx
        entry_text_y = 400  # 원하는 고정 위치로 조정 (예: 상단 바 아래)
        self.canvas.create_text(
            entry_text_x, entry_text_y,
            text=f"{entry_display}",
            fill="white",
            font=("Helvetica", 32, "bold")
        )

        # top/right numbers
        radius = LCD_WIDTH / 2
        self.canvas.create_text(cx, cy - radius + 80, text=f"{length_display}", fill="white", font=("Helvetica", 26, "bold"))
        self.canvas.create_text(cx + radius - 80, cy, text=f"{width_display}", fill="white", font=("Helvetica", 26, "bold"))

        # hole/par
        hole = int(self.hole_row.get("Hole", 0)) if self.hole_row is not None else 0
        par = int(self.hole_row.get("PAR", 0)) if self.hole_row is not None else 0
        self.canvas.create_text(cx - radius + 80, cy - 20, text=f"H{hole}", fill="white", font=("Helvetica", 22, "bold"))
        self.canvas.create_text(cx - radius + 80, cy + 20, text=f"P{par}", fill="white", font=("Helvetica", 22, "bold"))

        # STOP button
        self.canvas.create_window(400, 400, window=self.stop_button)

    def _on_stop(self):
        self.stop()
        if callable(self.on_back):
            self.on_back()
