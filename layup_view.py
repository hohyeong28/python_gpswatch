# layup_view.py (Frame/Screen 버전)
#
# - GreenView와 동일한 Frame 전환 구조
# - BK1~BK6, HZ1~HZ4를 활용한 LayupView
# - 후보 선정 조건:
#   1) 현재 위치 기준 그린(center) 방향 전방(dot>0)
#   2) 거리(미터 기준) 100~250  (odd=앞 포인트 기준)
#   3) 조건 만족 후보 중 최소거리 2개 선택
# - 좌/우 판정: 현재->그린 벡터 기준으로 후보가 좌/우인지 결정
# - 표시(요구 반영):
#   - 상단 중앙: 그린 center까지 거리
#   - 좌/우: bunker/water 아이콘 + (위: 현재->even, 아래: 현재->odd)
#   - 하단 중앙: "LAY UP"
#   - STOP 버튼: on_back()
#
# [추가 요구 반영]
# - 거리 표시값은 모두 "현재 위치 기준":
#   상단 = 현재→even, 하단 = 현재→odd
# - 후보 0개면 즉시 자동 복귀(on_back)

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


def dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


# -------------------- Layup Screen -------------------- #

class LayupViewScreen(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        on_back: Optional[Callable[[], None]] = None,
    ):
        super().__init__(master, width=LCD_WIDTH, height=LCD_HEIGHT, bg="black")
        self.pack_propagate(False)

        self.on_back = on_back

        # assets
        self.base_dir = Path(__file__).parent
        self.asset_dir = self.base_dir / "assets_image"
        self.images: Dict[Tuple[str, Optional[Tuple[int, int]]], ImageTk.PhotoImage] = {}

        # canvas
        self.canvas = tk.Canvas(self, width=LCD_WIDTH, height=LCD_HEIGHT, highlightthickness=0, bg="black")
        self.canvas.pack()

        # STOP button
        self.stop_button = tk.Button(self, text="STOP", command=self._on_stop)

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

        # green center/alt (for distance mode identical to measure_distance)
        self.green_center_EN: Tuple[float, float] = (0.0, 0.0)
        self.green_center_alt: float = 0.0

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
            print("[LayupViewScreen] transformer 생성 실패")
            return

        self._update_current_position()
        self.green_center_EN, self.green_center_alt = self._get_green_center_and_alt()

    def start(self):
        self.stop()
        self._render()
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
        self._render()
        self._after_id = self.after(1000, self._auto_update_loop)

    # ---------- images ---------- #

    def load_image(self, filename: str, size: Optional[Tuple[int, int]] = None) -> ImageTk.PhotoImage:
        key = (filename, size)
        if key in self.images:
            return self.images[key]
        path = self.asset_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"[LayupView] 이미지 없음: {path}")
        img = Image.open(path).convert("RGBA")
        if size is not None:
            img = img.resize(size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.images[key] = photo
        return photo

    # ---------- data helpers ---------- #

    def _get_EN_point(self, label: str) -> Optional[Tuple[float, float]]:
        assert self.hole_row is not None
        e = self.hole_row.get(f"{label}_E", np.nan)
        n = self.hole_row.get(f"{label}_N", np.nan)
        if pd.isna(e) or pd.isna(n):
            return None
        return float(e), float(n)

    def _get_green_center_and_alt(self) -> Tuple[Tuple[float, float], float]:
        assert self.hole_row is not None

        if self.selected_green == "L":
            center_label = "LG"
            alt_key = "LG_dAlt"
            pts_prefix = "LG"
        else:
            center_label = "RG"
            alt_key = "RG_dAlt"
            pts_prefix = "RG"

        c = self._get_EN_point(center_label)
        if c is None:
            pts = []
            for k in range(1, 21):
                p = self._get_EN_point(f"{pts_prefix}{k}")
                if p is not None:
                    pts.append(p)
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                c = (sum(xs) / len(xs), sum(ys) / len(ys))
            else:
                c = (self.E_cur, self.N_cur)

        a = self.hole_row.get(alt_key, np.nan)
        if pd.isna(a):
            a = 0.0

        return (float(c[0]), float(c[1])), float(a)

    # ---------- distance model (same as measure_distance) ---------- #

    def _compute_extended_dist(self) -> float:
        parent_alt = getattr(self.parent_window, "altitude", 0.0) if self.parent_window else 0.0
        alt_offset = getattr(self.parent_window, "alt_offset", 0.0) if self.parent_window else 0.0
        current_alt = parent_alt - alt_offset

        diff_h = self.green_center_alt - current_alt
        flat_center = euclidean_dist((self.E_cur, self.N_cur), self.green_center_EN)

        eDist = flat_center + diff_h
        eDist_clamped = 80.0 if eDist < 80 else 220.0 if eDist > 220 else eDist
        landing = -0.11 * eDist_clamped + 64.0
        landing_rad = math.radians(landing) if landing != 0 else 1e-6
        extended = diff_h / math.tan(landing_rad)
        return float(extended)

    def _distance_display_value(self, flat_dist_m: float) -> Tuple[float, int, str]:
        parent_alt = getattr(self.parent_window, "altitude", 0.0) if self.parent_window else 0.0
        alt_offset = getattr(self.parent_window, "alt_offset", 0.0) if self.parent_window else 0.0
        current_alt = parent_alt - alt_offset

        diff_h = self.green_center_alt - current_alt

        dist_mode = getattr(app_settings, "dist_mode", "직선")
        if dist_mode == "보정":
            extended = self._compute_extended_dist()
            d_m = flat_dist_m + extended
        else:
            d_m = math.sqrt(flat_dist_m * flat_dist_m + diff_h * diff_h)

        unit = getattr(app_settings, "unit", "M")
        conv = 1.0
        unit_label = "M"
        if unit == "Yd":
            conv = 1.09361
            unit_label = "Y"

        disp = int(round(d_m * conv))
        return d_m, disp, unit_label

    # ---------- candidate selection ---------- #

    def _iter_layup_pairs(self) -> List[Dict]:
        pairs = [
            ("BK", 1, 2),
            ("BK", 3, 4),
            ("BK", 5, 6),
            ("HZ", 1, 2),
            ("HZ", 3, 4),
        ]

        out: List[Dict] = []
        for typ, odd, even in pairs:
            p_odd = self._get_EN_point(f"{typ}{odd}")
            p_even = self._get_EN_point(f"{typ}{even}")
            if p_odd is None or p_even is None:
                continue
            out.append({
                "type": typ,
                "odd_idx": odd,
                "even_idx": even,
                "p_odd": p_odd,   # front
                "p_even": p_even, # back
            })
        return out

    def _select_best_candidates(self) -> List[Dict]:
        cur = (self.E_cur, self.N_cur)
        green = self.green_center_EN
        v_to_green = (green[0] - cur[0], green[1] - cur[1])

        candidates: List[Dict] = []
        for it in self._iter_layup_pairs():
            p_odd = it["p_odd"]
            p_even = it["p_even"]

            # 방향 조건: 현재->그린 방향 전방(dot>0) (odd 기준)
            v_to_odd = (p_odd[0] - cur[0], p_odd[1] - cur[1])
            if dot(v_to_green, v_to_odd) <= 0:
                continue

            # 거리 조건(미터 기준): odd 거리(보정/직선 반영된 미터값) 100~250
            flat_odd = euclidean_dist(cur, p_odd)
            odd_m, odd_disp, unit_label = self._distance_display_value(flat_odd)
            if odd_m < 100.0 or odd_m > 250.0:
                continue

            # 표시 상단값: 현재 -> even (요구 반영)
            flat_even = euclidean_dist(cur, p_even)
            _even_m, even_disp, _ = self._distance_display_value(flat_even)

            # 좌/우 판정(odd 기준)
            s = cross(v_to_green, v_to_odd)

            candidates.append({
                "type": it["type"],
                "odd": p_odd,
                "even": p_even,
                "odd_m": odd_m,
                "odd_disp": odd_disp,     # 하단 표시
                "even_disp": even_disp,   # 상단 표시
                "unit_label": unit_label,
                "side_sign": s,
            })

        # 최소거리 2개 선택 (odd_m 기준)
        candidates.sort(key=lambda x: x["odd_m"])
        selected = candidates[:2]

        # 좌/우 확정
        if len(selected) == 2:
            selected.sort(key=lambda x: x["side_sign"], reverse=True)
            selected[0]["slot"] = "L"
            selected[1]["slot"] = "R"
        elif len(selected) == 1:
            selected[0]["slot"] = "L" if selected[0]["side_sign"] >= 0 else "R"

        return selected

    # ---------- render ---------- #

    def _render(self):
        self.canvas.delete("all")

        # background
        try:
            bg = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
            self.canvas.create_image(0, 0, anchor="nw", image=bg)
        except FileNotFoundError:
            pass

        # green center distance (top center)
        self.green_center_EN, self.green_center_alt = self._get_green_center_and_alt()
        flat_center = euclidean_dist((self.E_cur, self.N_cur), self.green_center_EN)
        _, green_disp, _unit_label = self._distance_display_value(flat_center)

        cx = LCD_WIDTH / 2
        cy = LCD_HEIGHT / 2
        radius = LCD_WIDTH / 2

        self.canvas.create_text(cx, cy - radius + 90, text=f"{green_disp}", fill="white",
                                font=("Helvetica", 30, "bold"))

        # hole/par (left)
        hole = int(self.hole_row.get("Hole", 0)) if self.hole_row is not None else 0
        par = int(self.hole_row.get("PAR", 0)) if self.hole_row is not None else 0
        self.canvas.create_text(cx - radius + 80, cy - 20, text=f"H{hole}", fill="white",
                                font=("Helvetica", 22, "bold"))
        self.canvas.create_text(cx - radius + 80, cy + 20, text=f"P{par}", fill="white",
                                font=("Helvetica", 22, "bold"))

        # candidates
        selected = self._select_best_candidates()

        # [요구 반영] 후보가 0개면 즉시 자동 복귀
        if not selected:
            # 현재 _render() 호출 스택에서 바로 show()가 일어나면 충돌 가능성이 있어 after(0)로 디커플링
            self.after(0, self._on_stop)
            return

        slot_map = {it.get("slot"): it for it in selected}

        # icon positions (예시 이미지 형태)
        left_icon_center = (cx - 60, cy )
        right_icon_center = (cx + 90, cy )

        self._draw_slot(slot_map.get("L"), left_icon_center)
        self._draw_slot(slot_map.get("R"), right_icon_center)

        # label bottom center
        self.canvas.create_text(cx, cy + radius - 110, text="LAY UP", fill="white",
                                font=("Helvetica", 22, "bold"))

        # STOP button
        self.canvas.create_window(400, 400, window=self.stop_button)

    def _draw_slot(self, item: Optional[Dict], center_xy: Tuple[float, float]):
        """
        아이콘 + 위/아래 숫자 표시 (요구 반영)
        - 위: 현재->even
        - 아래: 현재->odd
        """
        if item is None:
            return

        x, y = center_xy

        icon_name = "bunker.png" if item["type"] == "BK" else "water.png"
        try:
            icon = self.load_image(icon_name)
        except FileNotFoundError:
            icon = None

        if icon is not None:
            self.canvas.create_image(x, y, image=icon)

        # top: current -> even
        self.canvas.create_text(x, y - 55, text=f"{item['even_disp']}", fill="white",
                                font=("Helvetica", 16, "bold"))
        # bottom: current -> odd
        self.canvas.create_text(x, y + 55, text=f"{item['odd_disp']}", fill="white",
                                font=("Helvetica", 16, "bold"))

    # ---------- stop/back ---------- #

    def _on_stop(self):
        self.stop()
        if callable(self.on_back):
            self.on_back()
