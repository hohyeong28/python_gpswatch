# meas_putt_distance.py (Frame/Screen 버전)
#
# - Toplevel 제거 → PuttDistanceScreen(tk.Frame)
# - START/Restart(이미지) 동작 유지
# - out-of-green(3초) → scoring 호출은 콜백(on_open_scoring)으로 전환
# - on_back() 제공(보통 DistanceScreen으로 복귀)
# - 단독 실행 코드 제거

import time
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Callable

import tkinter as tk
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
from pyproj import CRS, Transformer

from config import LCD_WIDTH, LCD_HEIGHT


def make_transformer(lat0: float, lon0: float) -> Optional[Transformer]:
    if np.isnan(lat0) or np.isnan(lon0):
        return None
    wgs84 = CRS.from_epsg(4326)
    aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m"
    )
    return Transformer.from_crs(wgs84, aeqd, always_xy=True)


def euclidean_dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


class PuttDistanceScreen(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        on_back: Optional[Callable[[], None]] = None,
        on_open_scoring: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(master, width=LCD_WIDTH, height=LCD_HEIGHT, bg="black")
        self.pack_propagate(False)

        self.on_back = on_back
        self.on_open_scoring = on_open_scoring

        self.base_dir = Path(__file__).parent
        self.asset_dir = self.base_dir / "assets_image"
        self.images: Dict[str, ImageTk.PhotoImage] = {}

        self.canvas = tk.Canvas(self, width=LCD_WIDTH, height=LCD_HEIGHT, highlightthickness=0, bg="black")
        self.canvas.pack()

        # context
        self.parent_window: Optional[tk.Misc] = None
        self.hole_row: Optional[pd.Series] = None
        self.gc_center_lat: float = float("nan")
        self.gc_center_lng: float = float("nan")
        self.cur_lat: float = float("nan")
        self.cur_lng: float = float("nan")

        self.tf: Optional[Transformer] = None
        self.E_cur: float = 0.0
        self.N_cur: float = 0.0

        self.putt_start_E: Optional[float] = None
        self.putt_start_N: Optional[float] = None

        self.measuring: bool = False
        self.current_putt_dist: float = 0.0

        # green polygon
        self.selected_green: str = "L"
        self.green_polygon: List[Tuple[float, float]] = []

        # out-of-green
        self.out_of_green_since: Optional[float] = None
        self.scoring_called: bool = False

        self._after_id: Optional[str] = None

        # BACK hitbox
        self._draw_back_hitbox()

    # ---------- public ---------- #

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
            print("[PuttDistanceScreen] transformer 생성 실패")
            return

        self._update_position_from_parent()

        self.selected_green = self._detect_selected_green()
        self.green_polygon = self._build_green_polygon()

        self.measuring = False
        self.current_putt_dist = 0.0
        self.putt_start_E = None
        self.putt_start_N = None

        self.out_of_green_since = None
        self.scoring_called = False

    def start(self):
        self.stop()
        self._render_screen()
        self._after_id = self.after(1000, self._auto_update_loop)

    def stop(self):
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    # ---------- back ---------- #

    def _draw_back_hitbox(self):
        self.canvas.create_text(60, 40, text="BACK", fill="white", font=("Helvetica", 14, "bold"))
        back_region = self.canvas.create_rectangle(0, 0, 120, 80, outline="", fill="")
        self.canvas.tag_bind(back_region, "<Button-1>", lambda e: self._on_back())

    def _on_back(self):
        self.stop()
        if callable(self.on_back):
            self.on_back()

    # ---------- position ---------- #

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

        if self.measuring and self.putt_start_E is not None and self.putt_start_N is not None:
            flat = euclidean_dist((self.E_cur, self.N_cur), (self.putt_start_E, self.putt_start_N))
            self.current_putt_dist = round(flat * 10.0) / 10.0

        self._check_out_of_green()
        self._render_screen()
        self._after_id = self.after(1000, self._auto_update_loop)

    # ---------- images ---------- #

    def load_image(self, filename: str, size=None) -> ImageTk.PhotoImage:
        key = filename if size is None else f"{filename}_{size[0]}x{size[1]}"
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

    # ---------- green polygon ---------- #

    def _check_rg_exists(self) -> bool:
        assert self.hole_row is not None
        rg_e = self.hole_row.get("RG_E", np.nan)
        rg_n = self.hole_row.get("RG_N", np.nan)
        return not (pd.isna(rg_e) or pd.isna(rg_n))

    def _detect_selected_green(self) -> str:
        has_rg = self._check_rg_exists()
        last = getattr(self.parent_window, "last_green", "L") if self.parent_window else "L"
        if not has_rg:
            return "L"
        return last if last in ("L", "R") else "L"

    def _build_green_polygon(self) -> List[Tuple[float, float]]:
        assert self.hole_row is not None
        prefix = "LG" if self.selected_green == "L" else "RG"
        poly: List[Tuple[float, float]] = []
        for k in range(1, 21):
            e = self.hole_row.get(f"{prefix}{k}_E", np.nan)
            n = self.hole_row.get(f"{prefix}{k}_N", np.nan)
            if pd.isna(e) or pd.isna(n):
                continue
            poly.append((float(e), float(n)))
        return poly

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

    def _check_out_of_green(self):
        if self.scoring_called:
            return
        if not self.green_polygon:
            return

        inside = self._is_point_in_polygon(self.E_cur, self.N_cur, self.green_polygon)
        if inside:
            self.out_of_green_since = None
            return

        now = time.time()
        if self.out_of_green_since is None:
            self.out_of_green_since = now
            return

        if (now - self.out_of_green_since) < 3.0:
            return

        self.scoring_called = True
        print("[PuttDistanceScreen] 그린 이탈 3초 → scoring 전환")

        if callable(self.on_open_scoring):
            ctx = dict(parent_window=self.parent_window, hole_row=self.hole_row)
            self.stop()
            self.on_open_scoring(ctx)

    # ---------- UI ---------- #

    def _render_screen(self):
        self.canvas.delete("all")

        try:
            bg_img = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
            self.canvas.create_image(0, 0, anchor="nw", image=bg_img)
        except FileNotFoundError:
            pass

        self._draw_back_hitbox()

        center_y = LCD_HEIGHT // 2

        try:
            flag_img = self.load_image("hole&falg.png")
            self.canvas.create_image(LCD_WIDTH // 2, center_y - 140, image=flag_img)
        except FileNotFoundError:
            pass

        if not self.measuring:
            try:
                start_img = self.load_image("start_putt.png")
                start_id = self.canvas.create_image(LCD_WIDTH // 2, center_y, image=start_img)
                self.canvas.tag_bind(start_id, "<Button-1>", self._on_start_putt)
            except FileNotFoundError:
                pass

            self.canvas.create_text(LCD_WIDTH // 2, center_y + 140, text="퍼팅 거리", fill="white",
                                    font=("Helvetica", 20, "bold"))
        else:
            # restart image
            try:
                restart_img = self.load_image("restart_putt.png", size=(70, 40))
                rx = LCD_WIDTH // 2 - 130
                ry = center_y
                rid = self.canvas.create_image(rx, ry, image=restart_img)
                self.canvas.tag_bind(rid, "<Button-1>", self._on_restart_putt)
            except FileNotFoundError:
                pass

            dist_str = f"{self.current_putt_dist:.1f}"
            self.canvas.create_text(LCD_WIDTH // 2, center_y, text=dist_str, fill="white",
                                    font=("Helvetica", 40, "bold"))
            self.canvas.create_text(LCD_WIDTH // 2 + 80, center_y, text="M", fill="white",
                                    font=("Helvetica", 30, "bold"))

            try:
                from_to_img = self.load_image("from_to.png")
                self.canvas.create_image(LCD_WIDTH // 2, center_y - 70, image=from_to_img)
                self.canvas.create_image(LCD_WIDTH // 2, center_y + 60, image=from_to_img)
            except FileNotFoundError:
                pass

            try:
                ball_img = self.load_image("ball.png")
                self.canvas.create_image(LCD_WIDTH // 2, center_y + 110, image=ball_img)
            except FileNotFoundError:
                pass

            self.canvas.create_text(LCD_WIDTH // 2, center_y + 150, text="퍼팅 거리", fill="white",
                                    font=("Helvetica", 20, "bold"))

    # ---------- events ---------- #

    def _reset_putt_start_to_current(self):
        self._update_position_from_parent()
        self.putt_start_E = self.E_cur
        self.putt_start_N = self.N_cur
        self.measuring = True
        self.current_putt_dist = 0.0

    def _on_start_putt(self, event):
        self._reset_putt_start_to_current()
        self._render_screen()

    def _on_restart_putt(self, event):
        self._reset_putt_start_to_current()
        self._render_screen()
