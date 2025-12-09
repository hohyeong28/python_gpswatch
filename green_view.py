# green_view.py
#
# - 현재 홀의 그린 컨투어(LG/RG)와 pin_position을 회전/스케일링해서 보여주는 창
# - DistanceWindow 에서 open_green_view_window(...) 로 호출하여 사용
# - 요구사항:
#   1) 골퍼의 위치는 화면 기준 6시 방향(아래) 쪽에 위치하도록 회전
#   2) entry(front) 포인트는 골퍼 → 핀 방향으로 쏜 직선이 그린과 처음 만나는 점
#      → 화면에서는 6시 방향 에지에 삼각형 표시
#   3) entry_distance 는 measure_distance.py 의 front 와 “동일 로직”으로 계산
#      (dist_mode = 직선/보정, 고도 보정, 단위 변환까지 동일)
#   4) green_pin_length(길이) / width(폭)는 각각 상단, 우측에 표시
#   5) Hole / PAR 정보는 좌측에 표시
#   6) STOP 버튼:
#        - (400, 400) 좌표에 "STOP" 버튼
#        - 클릭 시 GreenViewWindow 를 닫고, 부모(DistanceWindow)를 다시 포커스
#   7) 현재 위치(위도/경도)는 DistanceWindow 에서 계속 갱신되며,
#      GreenViewWindow 는 1초마다 부모로부터 위치를 읽어와 화면을 재계산한다.

import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import tkinter as tk
from PIL import Image, ImageTk

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

from config import LCD_WIDTH, LCD_HEIGHT
from setting import app_settings


# ---------------- 공용 유틸 ---------------- #

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
    A: Tuple[float, float],
    B: Tuple[float, float],
    P: Tuple[float, float],
    d: Tuple[float, float],
) -> Optional[Tuple[Tuple[float, float], float]]:
    """
    선분 AB 와 반직선 P + t*d (t>=0)의 교차.
    반환값: (교차점, t)  (t는 ray 방향으로의 파라미터)
    """
    Ax, Ay = A
    Bx, By = B
    Px, Py = P
    dx, dy = d

    rx = Bx - Ax
    ry = By - Ay

    denom = rx * dy - ry * dx
    if abs(denom) < 1e-9:
        return None  # 평행

    t_num = (Px - Ax) * ry - (Py - Ay) * rx
    s_num = (Px - Ax) * dy - (Py - Ay) * dx

    s = s_num / denom  # segment parameter
    t = t_num / denom  # ray parameter

    if s < 0.0 or s > 1.0 or t < 0.0:
        return None

    ix = Ax + s * rx
    iy = Ay + s * ry
    return (ix, iy), t


def segment_line_intersection(
    A: Tuple[float, float],
    B: Tuple[float, float],
    P: Tuple[float, float],
    d: Tuple[float, float],
) -> Optional[Tuple[Tuple[float, float], float]]:
    """
    선분 AB 와 직선 P + t*d (t ∈ R)의 교차.
    반환값: (교차점, t)  (t는 line 파라미터)
    """
    Ax, Ay = A
    Bx, By = B
    Px, Py = P
    dx, dy = d

    rx = Bx - Ax
    ry = By - Ay

    denom = rx * dy - ry * dx
    if abs(denom) < 1e-9:
        return None  # 평행

    t_num = (Px - Ax) * ry - (Py - Ay) * rx
    s_num = (Px - Ax) * dy - (Py - Ay) * dx

    s = s_num / denom  # segment parameter
    t = t_num / denom  # line parameter

    if s < 0.0 or s > 1.0:
        return None

    ix = Ax + s * rx
    iy = Ay + s * ry
    return (ix, iy), t


# ---------------- Green View Window ---------------- #

class GreenViewWindow(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Tk,
        hole_row: pd.Series,
        gc_center_lat: float,
        gc_center_lng: float,
        cur_lat: float,
        cur_lng: float,
        selected_green: str = "L",
    ):
        super().__init__(parent)

        self.title("Green View")
        self.resizable(False, False)

        self.geometry(
            f"{LCD_WIDTH}x{LCD_HEIGHT}+"
            f"{parent.winfo_rootx() + 40}+{parent.winfo_rooty() + 40}"
        )

        self.parent_window = parent
        self.hole_row = hole_row
        self.gc_center_lat = gc_center_lat
        self.gc_center_lng = gc_center_lng
        self.cur_lat = cur_lat
        self.cur_lng = cur_lng
        self.selected_green = selected_green  # 'L' or 'R'

        self.base_dir = Path(__file__).parent
        self.asset_dir = self.base_dir / "assets_image"
        self.images: Dict[Tuple[str, Tuple[int, int]], ImageTk.PhotoImage] = {}

        # 캔버스
        self.canvas = tk.Canvas(
            self,
            width=LCD_WIDTH,
            height=LCD_HEIGHT,
            highlightthickness=0,
            bg="black",
        )
        self.canvas.pack()

        # STOP 버튼 (400, 400 위치에 표시 예정)
        self.stop_button = tk.Button(
            self,
            text="STOP",
            command=self._on_stop,
        )

        # ENU 변환
        self.tf = make_transformer(self.gc_center_lat, self.gc_center_lng)
        if self.tf is None:
            print("[GreenView] transformer 생성 실패")
            return

        # 현재 위치 ENU (초기값)
        self.E_cur, self.N_cur = self.tf.transform(self.cur_lng, self.cur_lat)

        # 데이터 추출 (DB 기반)
        self.center_EN, self.center_alt = self._get_green_center_and_alt()
        self.pin_EN = self._get_pin_position()  # 현재는 center와 동일
        self.green_points = self._get_green_points()
        self.origin_EN = self._get_origin_point()

        # 첫 렌더
        self._compute_geometry_and_draw()

        # 주기적 위치/화면 갱신 (1초마다)
        self.after(1000, self._auto_update_loop)

    # --------- DB 필드 접근 --------- #

    def _get_EN_point(self, label: str) -> Optional[Tuple[float, float]]:
        e = self.hole_row.get(f"{label}_E", np.nan)
        n = self.hole_row.get(f"{label}_N", np.nan)
        if pd.isna(e) or pd.isna(n):
            return None
        return float(e), float(n)

    def _get_green_center_and_alt(self) -> Tuple[Tuple[float, float], float]:
        if self.selected_green == "L":
            label = "LG"
        else:
            label = "RG"

        # center ENU
        p = self._get_EN_point(label)
        if p is not None:
            center_en = p
        else:
            pts = self._get_green_points()
            if pts:
                xs = [q[0] for q in pts]
                ys = [q[1] for q in pts]
                center_en = (sum(xs) / len(xs), sum(ys) / len(ys))
            else:
                center_en = (self.E_cur, self.N_cur)

        # center altitude
        if self.selected_green == "L":
            a = self.hole_row.get("LG_dAlt", np.nan)
        else:
            a = self.hole_row.get("RG_dAlt", np.nan)
        if pd.isna(a):
            a = 0.0

        return center_en, float(a)

    def _get_pin_position(self) -> Tuple[float, float]:
        # 나중에 핀 위치가 별도 필드/설정으로 주어지면 여기서 변경
        return self.center_EN

    def _get_green_points(self) -> List[Tuple[float, float]]:
        pts = []
        prefix = "LG" if self.selected_green == "L" else "RG"
        for k in range(1, 21):
            p = self._get_EN_point(f"{prefix}{k}")
            if p is not None:
                pts.append(p)
        return pts

    def _get_origin_point(self) -> Tuple[float, float]:
        for label in ["IP2", "IP1", "T3"]:
            p = self._get_EN_point(label)
            if p is not None:
                return p
        return self.E_cur, self.N_cur

    # --------- 위치 갱신 --------- #

    def _update_current_position(self):
        """
        부모(DistanceWindow 또는 상위 객체)에서 현재 위치(lat/lng)를 읽어와
        self.cur_lat / self.cur_lng 및 ENU(self.E_cur / self.N_cur)를 갱신.
        """
        lat = None
        lng = None

        # 1순위: DistanceWindow 가 cur_lat / cur_lng 를 유지하는 경우
        if hasattr(self.parent_window, "cur_lat") and hasattr(self.parent_window, "cur_lng"):
            lat = getattr(self.parent_window, "cur_lat")
            lng = getattr(self.parent_window, "cur_lng")

        # 2순위: DistanceWindow.parent_window(PlayGolfWindow)의 latitude/longitude
        if (lat is None or lng is None) and hasattr(self.parent_window, "parent_window"):
            pg = getattr(self.parent_window, "parent_window")
            if hasattr(pg, "latitude") and hasattr(pg, "longitude"):
                lat = getattr(pg, "latitude")
                lng = getattr(pg, "longitude")

        # 3순위: 기존 값 유지
        if lat is None or lng is None:
            lat = self.cur_lat
            lng = self.cur_lng

        self.cur_lat = lat
        self.cur_lng = lng

        if self.tf is not None:
            self.E_cur, self.N_cur = self.tf.transform(self.cur_lng, self.cur_lat)

    def _auto_update_loop(self):
        if not self.winfo_exists():
            return
        self._update_current_position()
        self._compute_geometry_and_draw()
        self.after(1000, self._auto_update_loop)

    # --------- 이미지 로드 --------- #

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

    # --------- 주 계산 & 그리기 --------- #

    def _compute_geometry_and_draw(self):
        self.canvas.delete("all")

        # 배경
        try:
            bg = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
            self.canvas.create_image(0, 0, anchor="nw", image=bg)
        except FileNotFoundError:
            self.canvas.create_oval(
                0, 0, LCD_WIDTH, LCD_HEIGHT, fill="black", outline=""
            )

        if not self.green_points:
            print("[GreenView] 그린 포인트 없음")
            return

        # 1) 월드 좌표 → center 기준 상대좌표
        Cx, Cy = self.center_EN
        Px, Py = self.pin_EN
        Gpts_rel = [(x - Cx, y - Cy) for (x, y) in self.green_points]
        origin_rel = (self.origin_EN[0] - Cx, self.origin_EN[1] - Cy)
        cur_rel = (self.E_cur - Cx, self.N_cur - Cy)
        pin_rel = (Px - Cx, Py - Cy)

        # 2) 회전 각도: 현재 위치 → pin_position 방향이 화면에서 "위쪽(12시)"을 향하도록
        v = (pin_rel[0] - cur_rel[0], pin_rel[1] - cur_rel[1])
        if v == (0.0, 0.0):
            angle_v = 0.0
        else:
            angle_v = math.atan2(v[1], v[0])  # y, x (수학 좌표, y↑)
        target_angle = math.pi / 2  # (0, 1) 방향 = 12시
        theta = target_angle - angle_v

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        def rot(p):
            x, y = p
            return (x * cos_t - y * sin_t, x * sin_t + y * cos_t)

        G_rot = [rot(p) for p in Gpts_rel]
        origin_rot = rot(origin_rel)
        cur_rot = rot(cur_rel)
        pin_rot = rot(pin_rel)

        # 3) 스케일: 그린 컨투어 bounding box → 최대 234x234 px
        xs = [p[0] for p in G_rot]
        ys = [p[1] for p in G_rot]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        width = maxx - minx
        height = maxy - miny
        if width <= 0 or height <= 0:
            scale = 1.0
        else:
            scale = min(234.0 / width, 234.0 / height)

        cx = LCD_WIDTH / 2
        cy = LCD_HEIGHT / 2

        def world_to_screen(p):
            x, y = p
            sx = cx + x * scale
            sy = cy - y * scale
            return sx, sy

        # 4) entry/front/back 계산
        P0 = cur_rot
        Pin = pin_rot

        dir_vec = (Pin[0] - P0[0], Pin[1] - P0[1])
        norm = math.hypot(dir_vec[0], dir_vec[1])
        if norm == 0:
            dir_vec = (0.0, 1.0)
        else:
            dir_vec = (dir_vec[0] / norm, dir_vec[1] / norm)

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
            distances = [(dist(Pin, p), p) for p in G_rot]
            distances.sort(key=lambda x: x[0])
            p_front = distances[0][1]
            p_back = distances[-1][1]

        # 평면 거리 (회전 후이지만 길이는 동일)
        flat_entry = dist(P0, p_front)
        flat_center = dist(P0, Pin)

        # 5) measure_distance 와 동일 로직으로 entry_distance 계산
        # 고도 정보: center_alt vs current_alt
        # DistanceWindow.parent_window (PlayGolfWindow)에 altitude / alt_offset 이 있다고 가정
        pg = getattr(self.parent_window, "parent_window", self.parent_window)
        parent_alt = getattr(pg, "altitude", 0.0)
        alt_offset = getattr(pg, "alt_offset", 0.0)
        current_alt = parent_alt - alt_offset

        diff_h = self.center_alt - current_alt

        dist_mode = getattr(app_settings, "dist_mode", "직선")

        if dist_mode == "보정":
            eDist = flat_center + diff_h
            if eDist < 80:
                eDist_clamped = 80.0
            elif eDist > 220:
                eDist_clamped = 220.0
            else:
                eDist_clamped = eDist

            landing = -0.11 * eDist_clamped + 64.0
            landing_rad = math.radians(landing) if landing != 0 else 1e-6

            extended_dist = diff_h / math.tan(landing_rad)

            entry_distance = flat_entry + extended_dist
        else:
            entry_distance = math.sqrt(flat_entry * flat_entry + diff_h * diff_h)

        # green_pin_length / width 는 평면 거리 그대로 사용
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
            for i in range(len(G_rot)):
                A = G_rot[i]
                B = G_rot[(i + 1) % len(G_rot)]
                res = segment_line_intersection(A, B, Pin, a_dir)
                if res is not None:
                    length_inters.append(res)

            if len(length_inters) >= 2:
                length_inters.sort(key=lambda x: x[1])
                len_p1 = length_inters[0][0]
                len_p2 = length_inters[-1][0]
            else:
                len_p1 = p_front
                len_p2 = p_back

            green_pin_length = dist(len_p1, len_p2)

            width_inters: List[Tuple[Tuple[float, float], float]] = []
            for i in range(len(G_rot)):
                A = G_rot[i]
                B = G_rot[(i + 1) % len(G_rot)]
                res = segment_line_intersection(A, B, Pin, perp)
                if res is not None:
                    width_inters.append(res)

            if len(width_inters) >= 2:
                width_inters.sort(key=lambda x: x[1])
                p3 = width_inters[0][0]
                p4 = width_inters[-1][0]
                width_val = dist(p3, p4)
            else:
                width_val = 0.0
                p3 = p4 = Pin

        # 6) 단위 변환 (M / Yd) – measure_distance 와 동일
        unit = getattr(app_settings, "unit", "M")
        conv = 1.0
        if unit == "Yd":
            conv = 1.09361

        entry_display = int(round(entry_distance * conv))
        length_display = int(round(green_pin_length * conv))
        width_display = int(round(width_val * conv))

        # 6.5) 그린 근처 자동 복귀 처리
        # entry_display 는 단위 변환이 반영된 값 (M 또는 Yd)
        # → 20 이하면 "그린 근처입니다" 1초 표시 후 distance 화면으로 복귀
        if entry_display <= 20:
            # 배경 정리
            self.canvas.delete("all")
            try:
                bg = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
                self.canvas.create_image(0, 0, anchor="nw", image=bg)
            except FileNotFoundError:
                self.canvas.create_oval(
                    0, 0, LCD_WIDTH, LCD_HEIGHT, fill="black", outline=""
                )

            # 안내 문구 표시
            self.canvas.create_text(
                LCD_WIDTH // 2,
                LCD_HEIGHT // 2,
                text="그린 근처입니다",
                fill="white",
                font=("Helvetica", 30, "bold"),
            )

            # 1초 후 자동 종료(Measure Distance로 복귀)
            self.after(1000, self._on_stop)
            return  # 이후 그린/삼각형 등은 그리지 않음

        # 7) 화면에 그리기

        contour_screen = []
        for p in G_rot:
            contour_screen.extend(world_to_screen(p))
        self.canvas.create_polygon(
            contour_screen,
            fill="#90EE90",
            outline="white",
            width=3,
        )

        pin_s = world_to_screen(pin_rot)
        self.canvas.create_oval(
            pin_s[0] - 4,
            pin_s[1] - 4,
            pin_s[0] + 4,
            pin_s[1] + 4,
            fill="red",
            outline="red",
        )

        len1_s = world_to_screen(len_p1)
        len2_s = world_to_screen(len_p2)
        self.canvas.create_line(
            len1_s[0],
            len1_s[1],
            len2_s[0],
            len2_s[1],
            fill="red",
            dash=(4, 4),
            width=2,
        )

        p3_s = world_to_screen(p3)
        p4_s = world_to_screen(p4)
        self.canvas.create_line(
            p3_s[0],
            p3_s[1],
            p4_s[0],
            p4_s[1],
            fill="red",
            dash=(4, 4),
            width=2,
        )

        pf_s = world_to_screen(p_front)
        tip = pf_s

        dir_tri = (pin_s[0] - tip[0], pin_s[1] - tip[1])
        norm_tri = math.hypot(dir_tri[0], dir_tri[1])
        if norm_tri == 0:
            dir_tri = (0.0, -1.0)
            norm_tri = 1.0
        dtx = dir_tri[0] / norm_tri
        dty = dir_tri[1] / norm_tri

        pxv = -dty
        pyv = dtx

        tri_len = 18
        tri_width = 16

        base_center = (tip[0] - dtx * tri_len, tip[1] - dty * tri_len)

        left = (
            base_center[0] + pxv * (tri_width / 2),
            base_center[1] + pyv * (tri_width / 2),
        )
        right = (
            base_center[0] - pxv * (tri_width / 2),
            base_center[1] - pyv * (tri_width / 2),
        )
        self.canvas.create_polygon(
            [tip[0], tip[1], left[0], left[1], right[0], right[1]],
            fill="red",
            outline="red",
        )

        text_y = max(tip[1], left[1], right[1]) + 30
        self.canvas.create_text(
            cx,
            text_y,
            text=f"{entry_display}",
            fill="white",
            font=("Helvetica", 32, "bold"),
        )

        radius = LCD_WIDTH / 2
        self.canvas.create_text(
            cx,
            cy - radius + 80,
            text=f"{length_display}",
            fill="white",
            font=("Helvetica", 26, "bold"),
        )
        self.canvas.create_text(
            cx + radius - 80,
            cy,
            text=f"{width_display}",
            fill="white",
            font=("Helvetica", 26, "bold"),
        )

        hole = int(self.hole_row.get("Hole", 0))
        par = int(self.hole_row.get("PAR", 0))
        self.canvas.create_text(
            cx - radius + 80,
            cy - 20,
            text=f"H{hole}",
            fill="white",
            font=("Helvetica", 22, "bold"),
        )
        self.canvas.create_text(
            cx - radius + 80,
            cy + 20,
            text=f"P{par}",
            fill="white",
            font=("Helvetica", 22, "bold"),
        )

        self.canvas.create_window(
            400,
            400,
            window=self.stop_button,
        )

    # --------- STOP 버튼 핸들러 --------- #

    def _on_stop(self):
        try:
            parent = self.parent_window
            self.destroy()
            if parent is not None:
                try:
                    parent.lift()
                    parent.focus_set()
                except Exception:
                    pass
        except Exception as e:
            print("[GreenView] STOP 처리 중 오류:", e)


# ---------------- DistanceWindow 에서 사용할 엔트리 함수 ---------------- #

def open_green_view_window(
    parent: tk.Tk,
    hole_row: pd.Series,
    gc_center_lat: float,
    gc_center_lng: float,
    cur_lat: float,
    cur_lng: float,
    selected_green: str = "L",
) -> GreenViewWindow:
    return GreenViewWindow(
        parent=parent,
        hole_row=hole_row,
        gc_center_lat=gc_center_lat,
        gc_center_lng=gc_center_lng,
        cur_lat=cur_lat,
        cur_lng=cur_lng,
        selected_green=selected_green,
    )
