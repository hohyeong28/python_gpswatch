# measure_distance.py (Frame/Screen 버전)
#
# - 기존 DistanceWindow(Toplevel) 제거 → DistanceScreen(tk.Frame)
# - 단일 윈도우에서 ScreenManager로 전환되는 구조 지원
# - 외부에서 set_context(...)로 초기화 후 start() 호출 형태
# - GreenView / Putt 전환은 콜백(on_open_green_view, on_open_putt)으로 처리
# - 단독 실행 코드 제거

import math
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Callable

import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

from config import LCD_WIDTH, LCD_HEIGHT
from setting import app_settings


# -------------------- 좌표 변환 / 거리 유틸 -------------------- #

def make_transformer(lat0: float, lon0: float) -> Optional[Transformer]:
    if np.isnan(lat0) or np.isnan(lon0):
        return None
    wgs84 = CRS.from_epsg(4326)
    aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m"
    )
    return Transformer.from_crs(wgs84, aeqd, always_xy=True)  # (lon, lat) → (E, N)


def euclidean_dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


# -------------------- Distance Screen -------------------- #

class DistanceScreen(tk.Frame):
    """
    단일 윈도우용 거리 측정 화면(Frame).

    외부 연결(필수):
      - on_back(): 이전 화면(예: PlayGolf)로 복귀
      - on_open_green_view(context): GreenView 화면으로 전환
      - on_open_putt(context): PuttDistance 화면으로 전환

    context 전달 내용:
      - hole_row, gc_center_lat/lng, cur_lat/lng, selected_green, parent_window(PlayGolfFrame)
    """

    def __init__(
        self,
        master: tk.Misc,
        on_back: Optional[Callable[[], None]] = None,
        on_open_green_view: Optional[Callable[[dict], None]] = None,
        on_open_putt: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(master, width=LCD_WIDTH, height=LCD_HEIGHT, bg="black")
        self.pack_propagate(False)

        self.on_back = on_back
        self.on_open_green_view = on_open_green_view
        self.on_open_putt = on_open_putt

        # 외부에서 주입되는 컨텍스트
        self.parent_window: Optional[tk.Misc] = None  # PlayGolfFrame
        self.hole_row: Optional[pd.Series] = None
        self.gc_center_lat: float = float("nan")
        self.gc_center_lng: float = float("nan")

        self.cur_lat: float = float("nan")
        self.cur_lng: float = float("nan")

        # SHOT
        self.start_E: Optional[float] = None
        self.start_N: Optional[float] = None
        self.start_alt: Optional[float] = None

        self.shot_active: bool = False

        # on-green
        self.on_green_candidate_since: Optional[float] = None
        self.on_green_confirmed: bool = False

        # ENU transformer
        self.tf: Optional[Transformer] = None
        self.E_cur: float = 0.0
        self.N_cur: float = 0.0

        # fix_point
        self.fix_point_name: str = "CUR"
        self.fix_E: float = 0.0
        self.fix_N: float = 0.0
        self.fix_alt: float = 0.0

        # green selection
        self.has_rg: bool = False
        self.selected_green: str = "L"

        # assets/images
        self.base_dir = Path(__file__).parent
        self.asset_dir = self.base_dir / "assets_image"
        self.images: Dict[Tuple[str, Optional[Tuple[int, int]]], ImageTk.PhotoImage] = {}

        # canvas
        self.canvas = tk.Canvas(self, width=LCD_WIDTH, height=LCD_HEIGHT, highlightthickness=0, bg="black")
        self.canvas.pack()

        # Back 터치(좌상단)
        self._draw_back_hitbox()

        # GreenView 버튼(요구 유지: 400x400)
        self.greenview_button = tk.Button(self, text="GreenView", command=self._on_green_view)

        # 루프 핸들
        self._after_id: Optional[str] = None

    # ---------- public API ---------- #

    def set_context(
        self,
        parent_window: tk.Misc,
        hole_row: pd.Series,
        gc_center_lat: float,
        gc_center_lng: float,
        cur_lat: float,
        cur_lng: float,
    ):
        self.parent_window = parent_window
        self.hole_row = hole_row
        self.gc_center_lat = gc_center_lat
        self.gc_center_lng = gc_center_lng
        self.cur_lat = cur_lat
        self.cur_lng = cur_lng

        self.tf = make_transformer(self.gc_center_lat, self.gc_center_lng)
        if self.tf is None:
            print("[DistanceScreen] transformer 생성 실패")
            return

        self._update_position_from_parent()

        # fix_point는 화면 진입 시점 1회 선정
        self.fix_point_name, self.fix_E, self.fix_N, self.fix_alt = self._select_fix_point()

        # RG 존재 여부
        self.has_rg = self._check_rg_exists()

        # 선택 그린 초기화: parent_window.last_green 사용
        if self.has_rg:
            self.selected_green = getattr(self.parent_window, "last_green", "L")
        else:
            self.selected_green = "L"
        setattr(self.parent_window, "last_green", self.selected_green)

        # 상태 리셋
        self.start_E = self.start_N = self.start_alt = None
        self.shot_active = False
        self.on_green_candidate_since = None
        self.on_green_confirmed = False

    def start(self):
        """1초 루프 시작"""
        self.stop()
        self._render_screen()
        self._after_id = self.after(1000, self._auto_update_loop)

    def stop(self):
        """1초 루프 정지"""
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    # ---------- Back ---------- #

    def _draw_back_hitbox(self):
        self.canvas.create_text(60, 40, text="BACK", fill="white", font=("Helvetica", 14, "bold"))
        back_region = self.canvas.create_rectangle(0, 0, 120, 80, outline="", fill="")
        self.canvas.tag_bind(back_region, "<Button-1>", lambda e: self._on_back())

    def _on_back(self):
        self.stop()
        if callable(self.on_back):
            self.on_back()

    # ---------- 위치 갱신 ---------- #

    def _update_position_from_parent(self):
        if self.parent_window is None:
            return
        lat = getattr(self.parent_window, "latitude", self.cur_lat)
        lng = getattr(self.parent_window, "longitude", self.cur_lng)
        self.cur_lat = lat
        self.cur_lng = lng
        if self.tf is not None:
            self.E_cur, self.N_cur = self.tf.transform(self.cur_lng, self.cur_lat)

    def _auto_update_loop(self):
        self._update_position_from_parent()
        self._render_screen()
        self._after_id = self.after(1000, self._auto_update_loop)

    # ---------- 이미지 ---------- #

    def load_image(self, filename: str, size=None) -> ImageTk.PhotoImage:
        key = (filename, size)
        if key in self.images:
            return self.images[key]

        path = self.asset_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")

        img = Image.open(path).convert("RGBA")
        if size is not None:
            img = img.resize(size, Image.LANCZOS)

        photo = ImageTk.PhotoImage(img)
        self.images[key] = photo
        return photo

    # ---------- 데이터 ---------- #

    def _get_label_point(self, label: str) -> Optional[Tuple[float, float]]:
        assert self.hole_row is not None
        e = self.hole_row.get(f"{label}_E", np.nan)
        n = self.hole_row.get(f"{label}_N", np.nan)
        if pd.isna(e) or pd.isna(n):
            return None
        return float(e), float(n)

    def _get_label_alt(self, label: str) -> Optional[float]:
        assert self.hole_row is not None
        a = self.hole_row.get(f"{label}_dAlt", np.nan)
        if pd.isna(a):
            return None
        return float(a)

    def _select_fix_point(self) -> Tuple[str, float, float, float]:
        candidates = []
        for lab in ["T1", "T2", "T3", "T4", "T5", "T6", "IP1", "IP2"]:
            p = self._get_label_point(lab)
            if p is None:
                continue
            d = euclidean_dist((self.E_cur, self.N_cur), p)
            alt = self._get_label_alt(lab)
            if alt is None:
                alt = 0.0
            candidates.append((lab, p[0], p[1], alt, d))

        if not candidates:
            return "CUR", self.E_cur, self.N_cur, 0.0

        lab, e, n, alt, _ = min(candidates, key=lambda x: x[4])
        return lab, e, n, alt

    def _check_rg_exists(self) -> bool:
        assert self.hole_row is not None
        rg_e = self.hole_row.get("RG_E", np.nan)
        rg_n = self.hole_row.get("RG_N", np.nan)
        return not (pd.isna(rg_e) or pd.isna(rg_n))

    def _get_green_center_and_points(
        self, which: str
    ) -> Tuple[Tuple[float, float], float, List[Tuple[float, float, float]]]:
        assert self.hole_row is not None
        points: List[Tuple[float, float, float]] = []

        if which == "L":
            center_e = self.hole_row.get("LG_E", np.nan)
            center_n = self.hole_row.get("LG_N", np.nan)
            center_alt = self.hole_row.get("LG_dAlt", np.nan)
            for k in range(1, 21):
                e = self.hole_row.get(f"LG{k}_E", np.nan)
                n = self.hole_row.get(f"LG{k}_N", np.nan)
                a = self.hole_row.get(f"LG{k}_dAlt", np.nan)
                if pd.isna(e) or pd.isna(n):
                    continue
                if pd.isna(a):
                    a = center_alt if not pd.isna(center_alt) else 0.0
                points.append((float(e), float(n), float(a)))
        else:
            center_e = self.hole_row.get("RG_E", np.nan)
            center_n = self.hole_row.get("RG_N", np.nan)
            center_alt = self.hole_row.get("RG_dAlt", np.nan)
            for k in range(1, 21):
                e = self.hole_row.get(f"RG{k}_E", np.nan)
                n = self.hole_row.get(f"RG{k}_N", np.nan)
                a = self.hole_row.get(f"RG{k}_dAlt", np.nan)
                if pd.isna(e) or pd.isna(n):
                    continue
                if pd.isna(a):
                    a = center_alt if not pd.isna(center_alt) else 0.0
                points.append((float(e), float(n), float(a)))

        if pd.isna(center_e) or pd.isna(center_n):
            if points:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                center_e = float(sum(xs) / len(xs))
                center_n = float(sum(ys) / len(ys))
            else:
                center_e, center_n = self.E_cur, self.N_cur

        if pd.isna(center_alt):
            center_alt = 0.0

        return (float(center_e), float(center_n)), float(center_alt), points

    def _select_front_back(
        self, center: Tuple[float, float], points: List[Tuple[float, float, float]]
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        if not points:
            return (center[0], center[1], 0.0), (center[0], center[1], 0.0)

        P = (self.E_cur, self.N_cur)
        C = center
        vx = C[0] - P[0]
        vy = C[1] - P[1]
        v_norm2 = vx * vx + vy * vy
        if v_norm2 == 0:
            pts_sorted = sorted(points, key=lambda p: euclidean_dist(P, (p[0], p[1])))
            return pts_sorted[0], pts_sorted[-1]

        v_len = math.sqrt(v_norm2)

        best_front = None
        best_back = None
        min_along = float("inf")
        max_along = -float("inf")

        for (x, y, alt) in points:
            wx = x - P[0]
            wy = y - P[1]
            t = (wx * vx + wy * vy) / v_norm2
            along = t * v_len
            if along < min_along:
                min_along = along
                best_front = (x, y, alt)
            if along > max_along:
                max_along = along
                best_back = (x, y, alt)

        return best_front or points[0], best_back or points[-1]

    # ---------- on-green ---------- #

    def _is_point_in_polygon(self, x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
        inside = False
        n = len(polygon)
        if n < 3:
            return False
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

    def _check_ongreen(self, dist_info: Dict):
        if self.on_green_confirmed:
            return

        distance_front = dist_info.get("front")
        if distance_front is None or distance_front > 15:
            self.on_green_candidate_since = None
            return

        _, _, points = self._get_green_center_and_points(self.selected_green)
        polygon = [(e, n) for (e, n, _alt) in points]
        if len(polygon) < 3:
            self.on_green_candidate_since = None
            return

        inside = self._is_point_in_polygon(self.E_cur, self.N_cur, polygon)
        if not inside:
            self.on_green_candidate_since = None
            return

        now = time.time()
        if self.on_green_candidate_since is None:
            self.on_green_candidate_since = now
            return

        if (now - self.on_green_candidate_since) < 3.0:
            return

        self.on_green_confirmed = True
        print("[DistanceScreen] on_green 확정 → PuttDistanceScreen 전환")

        if callable(self.on_open_putt):
            ctx = dict(
                parent_window=self.parent_window,
                hole_row=self.hole_row,
                gc_center_lat=self.gc_center_lat,
                gc_center_lng=self.gc_center_lng,
                cur_lat=self.cur_lat,
                cur_lng=self.cur_lng,
            )
            self.stop()
            self.on_open_putt(ctx)

    # ---------- 거리 계산 ---------- #

    def _compute_distances(self) -> Dict:
        center, center_alt, points = self._get_green_center_and_points(self.selected_green)
        front_p, back_p = self._select_front_back(center, points)

        P = (self.E_cur, self.N_cur)
        flat_front = euclidean_dist(P, (front_p[0], front_p[1]))
        flat_center = euclidean_dist(P, center)
        flat_back = euclidean_dist(P, (back_p[0], back_p[1]))

        parent_alt = getattr(self.parent_window, "altitude", 0.0) if self.parent_window else 0.0
        alt_offset = getattr(self.parent_window, "alt_offset", 0.0) if self.parent_window else 0.0
        current_alt = parent_alt - alt_offset
        diff_h = center_alt - current_alt

        # SHOT
        shot_distance = None
        if self.start_E is not None and self.start_N is not None and self.start_alt is not None:
            flat_shot = euclidean_dist(P, (self.start_E, self.start_N))
            delta_alt = self.start_alt - current_alt
            # 기존 구현과 동일 형태(실질적으로 flat_shot)
            inside = flat_shot * flat_shot
            shot_distance = math.sqrt(max(0.0, inside))

        dist_mode = getattr(app_settings, "dist_mode", "직선")
        if dist_mode == "보정":
            eDist = flat_center + diff_h
            eDist_clamped = 80.0 if eDist < 80 else 220.0 if eDist > 220 else eDist
            landing = -0.11 * eDist_clamped + 64.0
            landing_rad = math.radians(landing) if landing != 0 else 1e-6
            extended_dist = diff_h / math.tan(landing_rad)

            distance_front = flat_front + extended_dist
            distance_center = flat_center + extended_dist
            distance_back = flat_back + extended_dist
            mode = "corrected"
        else:
            distance_front = math.sqrt(flat_front * flat_front + diff_h * diff_h)
            distance_center = math.sqrt(flat_center * flat_center + diff_h * diff_h)
            distance_back = math.sqrt(flat_back * flat_back + diff_h * diff_h)
            mode = "direct"

        unit = getattr(app_settings, "unit", "M")
        conv = 1.0
        unit_label = "M"
        if unit == "Yd":
            conv = 1.09361
            unit_label = "Y"

        distance_front = round(distance_front * conv)
        distance_center = round(distance_center * conv)
        distance_back = round(distance_back * conv)

        shot_display = round(shot_distance * conv) if shot_distance is not None else None

        return dict(
            front=distance_front,
            center=distance_center,
            back=distance_back,
            diff_h=diff_h,
            mode=mode,
            unit_label=unit_label,
            shot=shot_display,
        )

    # ---------- 렌더 ---------- #

    def _render_screen(self):
        self.canvas.delete("all")

        # background
        try:
            bg = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
            self.canvas.create_image(0, 0, anchor="nw", image=bg)
        except FileNotFoundError:
            pass

        # BACK hitbox 다시 그림
        self._draw_back_hitbox()

        # top bar
        self._draw_top_bar()

        dist_info = self._compute_distances()
        self._check_ongreen(dist_info)

        self._draw_hole_info()
        self._draw_green_selector()
        self._draw_distances(dist_info)
        self._draw_shot_info(dist_info)

        if dist_info["mode"] == "corrected":
            self._draw_slope_indicator(dist_info["diff_h"])

        self.canvas.create_window(400, 400, window=self.greenview_button)

    def _draw_top_bar(self):
        now_str = time.strftime("%H:%M")
        battery_str = "87%"
        self.canvas.create_text(LCD_WIDTH // 2 - 50, 80, text=now_str, fill="white", font=("Helvetica", 20, "bold"))
        self.canvas.create_text(LCD_WIDTH // 2 + 50, 80, text=battery_str, fill="white", font=("Helvetica", 20, "bold"))

    def _draw_hole_info(self):
        assert self.hole_row is not None
        hole = self.hole_row.get("Hole", "")
        par = self.hole_row.get("PAR", "")
        self.canvas.create_text(LCD_WIDTH // 4 - 10, LCD_HEIGHT // 2 - 25, text=f"H{hole}", fill="white", font=("Helvetica", 30, "bold"))
        self.canvas.create_text(LCD_WIDTH // 4 - 10, LCD_HEIGHT // 2 + 25, text=f"P{par}", fill="white", font=("Helvetica", 30, "bold"))

    def _draw_green_selector(self):
        if not self.has_rg:
            return

        img = self.load_image("left_green.png") if self.selected_green == "L" else self.load_image("right_green.png")
        x = LCD_WIDTH // 2 - 110
        y = LCD_HEIGHT // 2 - 90
        icon_id = self.canvas.create_image(x, y, image=img)
        self.canvas.tag_bind(icon_id, "<Button-1>", self._on_toggle_green)

    def _on_toggle_green(self, event):
        if not self.has_rg:
            return
        self.selected_green = "R" if self.selected_green == "L" else "L"
        if self.parent_window is not None:
            setattr(self.parent_window, "last_green", self.selected_green)
        self._render_screen()

    def _draw_distances(self, info: Dict):
        center_y = LCD_HEIGHT // 2
        unit_label = info["unit_label"]

        self.canvas.create_text(LCD_WIDTH // 2, center_y, text=f"{info['center']}", fill="white", font=("Helvetica", 70, "bold"))
        self.canvas.create_text(LCD_WIDTH // 2, center_y - 80, text=f"{info['back']}", fill="white", font=("Helvetica", 40, "bold"))
        self.canvas.create_text(LCD_WIDTH // 2, center_y + 80, text=f"{info['front']}", fill="white", font=("Helvetica", 40, "bold"))
        self.canvas.create_text(LCD_WIDTH // 2 + 120, center_y + 20, text=unit_label, fill="white", font=("Helvetica", 40, "bold"))

    def _draw_shot_info(self, info: Dict):
        try:
            shot_bg = self.load_image("confirm.png")
        except FileNotFoundError:
            return

        x_center = LCD_WIDTH // 2
        y_center = LCD_HEIGHT - 70

        bg_id = self.canvas.create_image(x_center, y_center, image=shot_bg)
        self.canvas.tag_bind(bg_id, "<Button-1>", self._on_shot)

        shot = info.get("shot")
        if shot is None:
            t = self.canvas.create_text(x_center, y_center, text="SHOT", fill="white", font=("Helvetica", 18, "bold"))
            self.canvas.tag_bind(t, "<Button-1>", self._on_shot)
        else:
            t1 = self.canvas.create_text(x_center, y_center - 20, text="SHOT", fill="white", font=("Helvetica", 12, "bold"))
            t2 = self.canvas.create_text(x_center, y_center + 8, text=f"{shot}", fill="white", font=("Helvetica", 30, "bold"))
            self.canvas.tag_bind(t1, "<Button-1>", self._on_shot)
            self.canvas.tag_bind(t2, "<Button-1>", self._on_shot)

    def _on_shot(self, event):
        self._update_position_from_parent()
        parent_alt = getattr(self.parent_window, "altitude", 0.0) if self.parent_window else 0.0
        alt_offset = getattr(self.parent_window, "alt_offset", 0.0) if self.parent_window else 0.0
        current_alt = parent_alt - alt_offset
        self.start_E = self.E_cur
        self.start_N = self.N_cur
        self.start_alt = current_alt
        self.shot_active = True
        self._render_screen()

    def _draw_slope_indicator(self, diff_h: float):
        if diff_h == 0:
            return
        icon_name = "uphill.png" if diff_h > 0 else "downhill.png"
        img = self.load_image(icon_name)
        self.canvas.create_image(LCD_WIDTH - 115, LCD_HEIGHT // 2 - 30, image=img)
        self.canvas.create_text(LCD_WIDTH - 110, LCD_HEIGHT // 2 - 70, text=f"{int(round(diff_h)):+d}",
                                fill="white", font=("Helvetica", 25, "bold"))

    # ---------- GreenView ---------- #

    def _on_green_view(self):
        if not callable(self.on_open_green_view):
            return
        ctx = dict(
            parent_window=self.parent_window,
            hole_row=self.hole_row,
            gc_center_lat=self.gc_center_lat,
            gc_center_lng=self.gc_center_lng,
            cur_lat=self.cur_lat,
            cur_lng=self.cur_lng,
            selected_green=self.selected_green,
        )
        self.stop()
        self.on_open_green_view(ctx)
