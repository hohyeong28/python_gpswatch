# measure_distance.py
#
# - PlayGolfWindow.find_1st_hole() 에서 홀 fix 후 호출되는 거리 측정 화면
# - 요구사항:
#   1) 상단 바: 현재 시간(PC 시간), 배터리 87%
#   2) L/R 좌우 그린 선택:
#        - LG는 항상 있음
#        - RG가 없으면 L/R 아이콘 자체를 표시하지 않고 LG만 사용
#        - RG가 있으면 left_green.png / right_green.png 로 토글 (기본 L)
#   3) 홀 정보: "H{Hole}", "P{PAR}"
#   4) setting.app_settings 연동:
#        - unit: M → m 그대로, Yd → Y표시 + 1.09361 배 후 반올림
#        - dist_mode: "보정" / "직선"
#   5) 거리:
#        - fix_point: T1~T6, IP1, IP2 중 현재 위치에 가장 가까운 지점
#        - 선택된 그린(LG 또는 RG)에 대해 front/center/back 계산
#        - "직선": sqrt(flat^2 + diff_h^2) (diff_h는 center-alt - fix-alt), uphill/downhill 미표시
#        - "보정": 주어진 공식에 따라 보정 거리 계산 + uphill/downhill + diff_h 표시
#   6) 하단: confirm.png + "SHOT"
#   7) background.png는 entry_menu / playgolf와 동일하게 전체 배경으로 사용
#
#   추가 변경:
#   - DistanceWindow 는 parent(PlayGolfWindow)의 latitude/longitude 를 1초마다 읽어와
#     현재 위치(E_cur, N_cur)를 갱신하고, 거리를 재계산/렌더링한다.
#   - fix_point 는 창이 열린 시점의 위치 기준으로 한 번만 선정한다.
#   - SHOT 버튼:
#       * 클릭 순간의 위치/고도를 start_point 로 기록
#       * 이후 start_point ~ 현재 위치까지의 shot_distance 를 계산/표시
#       * 이동하면서 shot_distance 는 계속 갱신
#       * 이동 중 다시 클릭하면 start_point 를 새로 설정
#
#   - check_ongreen():
#       * distance_front <= 15 일 때 동작
#       * 선택된 그린(L/R)의 LG1~20 또는 RG1~20 좌표로 이루어진 폐곡선 내부에
#         현재 위치(E_cur, N_cur)가 있는지 검사
#       * 3초 동안 연속 내부일 경우 on_green 확정
#       * on_green 확정 시 meas_putt_distance.open_putt_window(...) 실행
#
#   - GreenView 버튼:
#       * 화면 (400, 400) 위치에 "GreenView" 버튼 추가
#       * 버튼 클릭 시 green_view.open_green_view_window(...) 실행
#       * DistanceWindow 의 selected_green(L/R) 정보를 그대로 전달

import math
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, List

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
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m")
    return Transformer.from_crs(wgs84, aeqd, always_xy=True)  # (lon, lat) → (E, N)


def euclidean_dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


# -------------------- Distance Window -------------------- #

class DistanceWindow(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Tk,
        hole_row: pd.Series,
        gc_center_lat: float,
        gc_center_lng: float,
        cur_lat: float,
        cur_lng: float,
    ):
        super().__init__(parent)

        self.title("Measure Distance")
        self.resizable(False, False)

        # 위치: 부모 창 근처
        self.geometry(
            f"{LCD_WIDTH}x{LCD_HEIGHT}+"
            f"{parent.winfo_rootx() + 40}+{parent.winfo_rooty() + 40}"
        )

        # 부모(PlayGolfWindow) 참조
        self.parent_window = parent

        # 상태 값
        self.hole_row = hole_row
        self.gc_center_lat = gc_center_lat
        self.gc_center_lng = gc_center_lng

        # 현재 위치 (초기값: 생성 시점 스냅샷, 이후 parent에서 계속 갱신)
        self.cur_lat = cur_lat
        self.cur_lng = cur_lng

        # SHOT 측정용 start point (ENU + 보정된 alt)
        self.start_E: Optional[float] = None
        self.start_N: Optional[float] = None
        self.start_alt: Optional[float] = None  # current_alt 기준

        # SHOT 버튼 상태 (처음엔 비활성)
        self.shot_active: bool = False

        # on-green 판정용 상태
        self.on_green_candidate_since: Optional[float] = None
        self.on_green_confirmed: bool = False

        self.base_dir = Path(__file__).parent
        self.asset_dir = self.base_dir / "assets_image"

        # 이미지 캐시
        self.images: Dict[Tuple[str, Optional[Tuple[int, int]]], ImageTk.PhotoImage] = {}

        # 캔버스
        self.canvas = tk.Canvas(
            self,
            width=LCD_WIDTH,
            height=LCD_HEIGHT,
            highlightthickness=0,
            bg="black",
        )
        self.canvas.pack()

        # GreenView 버튼 (실제 위젯은 한 번만 만들고, 매 렌더링 시 canvas에 올린다)
        self.greenview_button = tk.Button(
            self,
            text="GreenView",
            command=self._on_green_view,
        )

        # ENU 변환용 transformer
        self.tf = make_transformer(self.gc_center_lat, self.gc_center_lng)
        if self.tf is None:
            print("[DistanceWindow] transformer 생성 실패")
            return

        # 최초 현재 위치를 parent에서 읽어와 ENU 좌표 계산
        self._update_position_from_parent()

        # fix_point 선정 (창이 열린 시점 기준으로 한 번만 고정)
        self.fix_point_name, self.fix_E, self.fix_N, self.fix_alt = \
            self._select_fix_point()

        # RG 존재 여부 판단
        self.has_rg = self._check_rg_exists()

        # 현재 선택 그린 ('L' 또는 'R')
        # RG가 없으면 강제로 L 사용 (LG만 존재)
        if self.has_rg:
            self.selected_green = getattr(self.parent_window, "last_green", "L")
        else:
            self.selected_green = "L"

        # 초기 상태를 부모에도 반영 (최초 홀에서 기본 L 사용 시)
        setattr(self.parent_window, "last_green", self.selected_green)

        # 최초 렌더
        self._render_screen()

        # 주기적으로 현재 위치/거리 갱신 (1초 간격)
        self.after(1000, self._auto_update_loop)

    # ------------- 현재 위치 갱신 (parent 연동) ------------- #

    def _update_position_from_parent(self):
        """
        부모(PlayGolfWindow)의 latitude/longitude 를 읽어와
        self.cur_lat/self.cur_lng 및 ENU(self.E_cur/self.N_cur)를 갱신.
        """
        lat = getattr(self.parent_window, "latitude", self.cur_lat)
        lng = getattr(self.parent_window, "longitude", self.cur_lng)

        self.cur_lat = lat
        self.cur_lng = lng

        if self.tf is not None:
            self.E_cur, self.N_cur = self.tf.transform(self.cur_lng, self.cur_lat)

    def _auto_update_loop(self):
        """
        1초마다 현재 위치를 부모에서 읽어와 갱신하고,
        거리/화면을 다시 계산/렌더링.
        """
        # 이미 on_green 확정된 상태에서도 화면 갱신은 유지 (필요시 여기서 분기 가능)
        self._update_position_from_parent()
        self._render_screen()
        self.after(1000, self._auto_update_loop)

    # ------------- 이미지 로드 ------------- #

    def load_image(self, filename: str, size=None) -> ImageTk.PhotoImage:
        """
        filename + size 조합을 key 로 하여 최초 1회만 로딩하고,
        이후에는 캐시된 PhotoImage 를 그대로 재사용한다.
        """
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

    # ------------- 데이터 헬퍼 ------------- #

    def _get_label_point(self, label: str) -> Optional[Tuple[float, float]]:
        e = self.hole_row.get(f"{label}_E", np.nan)
        n = self.hole_row.get(f"{label}_N", np.nan)
        if pd.isna(e) or pd.isna(n):
            return None
        return float(e), float(n)

    def _get_label_alt(self, label: str) -> Optional[float]:
        a = self.hole_row.get(f"{label}_dAlt", np.nan)
        if pd.isna(a):
            return None
        return float(a)

    # fix_point: T1~T6, IP1, IP2 중 현재 위치와 가장 가까운 포인트
    # (창이 열린 시점의 E_cur/N_cur 기준으로 한 번만 선정)
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
            print("[DistanceWindow] fix_point 후보 없음 → 현재 위치 기준 고도 0 사용")
            return "CUR", self.E_cur, self.N_cur, 0.0

        lab, e, n, alt, _ = min(candidates, key=lambda x: x[4])
        print(f"[DistanceWindow] fix_point: {lab} ({e:.1f},{n:.1f}), alt={alt}")
        return lab, e, n, alt

    # RG 존재 여부
    def _check_rg_exists(self) -> bool:
        rg_e = self.hole_row.get("RG_E", np.nan)
        rg_n = self.hole_row.get("RG_N", np.nan)
        valid = not (pd.isna(rg_e) or pd.isna(rg_n))
        return bool(valid)

    # ------------- Green front/center/back 계산 ------------- #

    def _get_green_center_and_points(
        self, which: str
    ) -> Tuple[Tuple[float, float], float, List[Tuple[float, float, float]]]:
        points = []
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
        else:  # 'R'
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

        center = (float(center_e), float(center_n))
        return center, float(center_alt), points

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

        if best_front is None:
            best_front = points[0]
        if best_back is None:
            best_back = points[-1]

        return best_front, best_back

    # ------------- on-green 관련 유틸 ------------- #

    def _is_point_in_polygon(
        self, x: float, y: float, polygon: List[Tuple[float, float]]
    ) -> bool:
        """
        Ray casting 알고리즘으로 (x, y)가 polygon 내부에 있는지 판정.
        polygon: [(x1, y1), (x2, y2), ...]
        """
        inside = False
        n = len(polygon)
        if n < 3:
            return False

        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]

            # y축 방향으로 ray 교차 여부
            if (y1 > y) != (y2 > y):
                denom = (y2 - y1)
                if abs(denom) < 1e-12:
                    continue
                x_int = (x2 - x1) * (y - y1) / denom + x1
                if x_int > x:
                    inside = not inside

        return inside

    def _check_ongreen(self, dist_info: Dict):
        """
        distance_front <= 15 일 때,
        선택된 그린(L/R)의 폴리곤(LG1~20 / RG1~20)에 대해
        현재 위치(E_cur, N_cur)가 내부인지 검사하고,
        3초간 연속 내부인 경우 on_green 확정 후 퍼팅 모드 진입.
        """
        # 이미 on_green 확정된 상태라면 더 이상 처리하지 않음
        if self.on_green_confirmed:
            return

        # 2) distance_front <= 15 인 경우에만 검사
        distance_front = dist_info.get("front")
        if distance_front is None or distance_front > 15:
            # 조건 벗어나면 후보 타이머 리셋
            self.on_green_candidate_since = None
            return

        # 3) 선택된 그린의 폴리곤(LG1~20 또는 RG1~20) 가져오기
        _, _, points = self._get_green_center_and_points(self.selected_green)
        polygon = [(e, n) for (e, n, _alt) in points]

        if len(polygon) < 3:
            # 폴리곤 구성 불가 → on_green 판정 불가
            self.on_green_candidate_since = None
            return

        # 현재 위치 (E_cur, N_cur)가 폴리곤 내부인지 판정
        inside = self._is_point_in_polygon(self.E_cur, self.N_cur, polygon)

        if not inside:
            # 내부가 아니면 타이머 리셋
            self.on_green_candidate_since = None
            return

        # 4) 내부이고, 3초 동안 계속 내부인지 확인
        now = time.time()
        if self.on_green_candidate_since is None:
            # 처음 내부로 진입한 시점 기록
            self.on_green_candidate_since = now
            return

        elapsed = now - self.on_green_candidate_since
        if elapsed < 3.0:
            # 아직 3초 미만 → 계속 관찰
            return

        # 5) 3초 이상 연속 내부 → on_green 확정
        self.on_green_confirmed = True
        print("[DistanceWindow] on_green 확정, 퍼팅 모드(meas_putt_distance)로 전환")

        # meas_putt_distance.py 와의 연동
        try:
            from meas_putt_distance import open_putt_window

            # 필요에 맞게 시그니처를 맞춰 구현하면 됨
            open_putt_window(
                parent=self.parent_window,
                hole_row=self.hole_row,
                gc_center_lat=self.gc_center_lat,
                gc_center_lng=self.gc_center_lng,
                cur_lat=self.cur_lat,
                cur_lng=self.cur_lng,
            )
        except ImportError:
            print("[DistanceWindow] meas_putt_distance 모듈을 찾을 수 없습니다.")
        except Exception as e:
            print("[DistanceWindow] meas_putt_distance 실행 중 오류:", e)

    # ------------- 거리 계산 ------------- #

    def _compute_distances(self):
        """
        front/center/back + SHOT 거리 계산
        """
        center, center_alt, points = self._get_green_center_and_points(
            self.selected_green
        )
        front_p, back_p = self._select_front_back(center, points)

        P = (self.E_cur, self.N_cur)
        flat_front = euclidean_dist(P, (front_p[0], front_p[1]))
        flat_center = euclidean_dist(P, center)
        flat_back = euclidean_dist(P, (back_p[0], back_p[1]))

        # 보정된 현재 고도
        parent_alt = getattr(self.parent_window, "altitude", 0.0)
        alt_offset = getattr(self.parent_window, "alt_offset", 0.0)
        current_alt = parent_alt - alt_offset

        diff_h = center_alt - current_alt

        # SHOT 거리 계산
        shot_distance = None
        if (
            self.start_E is not None
            and self.start_N is not None
            and self.start_alt is not None
        ):
            flat_shot = euclidean_dist(P, (self.start_E, self.start_N))
            delta_alt = self.start_alt - current_alt
            gps_distance_sq = flat_shot * flat_shot + delta_alt * delta_alt
            inside = gps_distance_sq - delta_alt * delta_alt
            if inside < 0:
                inside = 0.0
            shot_distance = math.sqrt(inside)

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

            distance_center = flat_center + extended_dist
            distance_front = flat_front + extended_dist
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

        distance_front *= conv
        distance_center *= conv
        distance_back *= conv

        distance_front = round(distance_front)
        distance_center = round(distance_center)
        distance_back = round(distance_back)

        shot_display = None
        if shot_distance is not None:
            shot_display = round(shot_distance * conv)

        return dict(
            front=distance_front,
            center=distance_center,
            back=distance_back,
            diff_h=diff_h,
            mode=mode,
            unit_label=unit_label,
            shot=shot_display,
        )

    # ------------- 화면 렌더링 ------------- #

    def _render_screen(self):
        self.canvas.delete("all")

        # background.png 전체 적용
        try:
            bg_img = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
            self.canvas.create_image(0, 0, anchor="nw", image=bg_img)
        except FileNotFoundError:
            pass

        # 상단 바
        self._draw_top_bar()

        # 거리 계산
        dist_info = self._compute_distances()

        # on-green 판정
        self._check_ongreen(dist_info)

        # 홀 정보
        self._draw_hole_info()

        # L/R 선택 아이콘
        self._draw_green_selector()

        # 거리 숫자 표시
        self._draw_distances(dist_info)

        # SHOT 캡 (버튼 + 거리 표시 겸용)
        self._draw_shot_info(dist_info)

        # 오르막/내리막 (보정 모드에서만)
        if dist_info["mode"] == "corrected":
            self._draw_slope_indicator(dist_info["diff_h"])

        # GreenView 버튼 (400, 400 위치에 표시)
        self.canvas.create_window(
            400,
            400,
            window=self.greenview_button,
        )

    def _draw_top_bar(self):
        now_str = time.strftime("%H:%M")
        battery_str = "87%"

        self.canvas.create_text(
            LCD_WIDTH // 2 - 50,
            80,
            text=now_str,
            fill="white",
            font=("Helvetica", 20, "bold"),
        )
        self.canvas.create_text(
            LCD_WIDTH // 2 + 50,
            80,
            text=battery_str,
            fill="white",
            font=("Helvetica", 20, "bold"),
        )

    def _draw_hole_info(self):
        hole = self.hole_row.get("Hole", "")
        par = self.hole_row.get("PAR", "")

        self.canvas.create_text(
            LCD_WIDTH // 4 - 10,
            LCD_HEIGHT // 2 - 25,
            text=f"H{hole}",
            fill="white",
            font=("Helvetica", 30, "bold"),
        )
        self.canvas.create_text(
            LCD_WIDTH // 4 - 10,
            LCD_HEIGHT // 2 + 25,
            text=f"P{par}",
            fill="white",
            font=("Helvetica", 30, "bold"),
        )

    def _draw_green_selector(self):
        if not self.has_rg:
            return

        # 현재 선택 상태에 맞는 이미지 선택
        if self.selected_green == "L":
            img = self.load_image("left_green.png")
        else:
            img = self.load_image("right_green.png")

        x = LCD_WIDTH // 2 - 110
        y = LCD_HEIGHT // 2 - 90  # 하나의 위치만 사용

        # 아이콘 생성 + 토글 버튼 바인딩
        icon_id = self.canvas.create_image(x, y, image=img)
        self.canvas.tag_bind(icon_id, "<Button-1>", self._on_toggle_green)

    def _on_toggle_green(self, event):
        if not self.has_rg:
            return

        # L <-> R 토글
        self.selected_green = "R" if self.selected_green == "L" else "L"

        # 부모에 현재 선택 상태를 저장 (다음 홀에서 초기값으로 사용)
        setattr(self.parent_window, "last_green", self.selected_green)

        # 화면 전체 재렌더 (거리 계산 기준도 자동으로 변경)
        self._render_screen()

    def _draw_distances(self, info: Dict):
        center_y = LCD_HEIGHT // 2
        unit_label = info["unit_label"]

        self.canvas.create_text(
            LCD_WIDTH // 2,
            center_y,
            text=f"{info['center']}",
            fill="white",
            font=("Helvetica", 70, "bold"),
        )

        self.canvas.create_text(
            LCD_WIDTH // 2,
            center_y - 80,
            text=f"{info['back']}",
            fill="white",
            font=("Helvetica", 40, "bold"),
        )

        self.canvas.create_text(
            LCD_WIDTH // 2,
            center_y + 80,
            text=f"{info['front']}",
            fill="white",
            font=("Helvetica", 40, "bold"),
        )

        self.canvas.create_text(
            LCD_WIDTH // 2 + 120,
            center_y + 20,
            text=unit_label,
            fill="white",
            font=("Helvetica", 40, "bold"),
        )

    def _draw_shot_info(self, info: Dict):
        """
        SHOT 거리 표시 + 버튼 역할
        - confirm.png 이미지를 SHOT 캡 배경으로 사용한다.
        - 시작 화면(shot is None):  ⟶  캡 중앙에 'SHOT' (크게)
        - shot_distance 화면(shot is not None): ⟶ 상단에 작은 'SHOT', 그 아래에 큰 거리 숫자
        - 캡 전체가 클릭 영역 (_on_shot)
        """
        shot = info.get("shot")
        unit_label = info.get("unit_label", "M")

        # 배경 이미지 (confirm.png)
        try:
            shot_bg = self.load_image("confirm.png")
        except FileNotFoundError:
            return

        x_center = LCD_WIDTH // 2
        y_center = LCD_HEIGHT - 70  # 캡 위치

        # PNG 배경
        shot_bg_id = self.canvas.create_image(
            x_center,
            y_center,
            image=shot_bg,
        )
        self.canvas.tag_bind(shot_bg_id, "<Button-1>", self._on_shot)

        if shot is None:
            # ───────────── 1) 시작 화면: SHOT만 크게 ─────────────
            shot_text_id = self.canvas.create_text(
                x_center,
                y_center,  # 캡 중앙
                text="SHOT",
                fill="white",
                font=("Helvetica", 18, "bold"),
            )
            self.canvas.tag_bind(shot_text_id, "<Button-1>", self._on_shot)
            return

        # ───────────── 2) shot_distance 화면: 위에 작은 SHOT, 아래에 거리 ─────────────
        # 작은 SHOT (상단)
        shot_label_id = self.canvas.create_text(
            x_center,
            y_center - 20,
            text="SHOT",
            fill="white",
            font=("Helvetica", 12, "bold"),
        )
        self.canvas.tag_bind(shot_label_id, "<Button-1>", self._on_shot)

        # 큰 거리 숫자 (중앙)
        shot_val_id = self.canvas.create_text(
            x_center,
            y_center + 8,
            text=f"{shot}",
            fill="white",
            font=("Helvetica", 30, "bold"),
        )
        self.canvas.tag_bind(shot_val_id, "<Button-1>", self._on_shot)

    def _draw_slope_indicator(self, diff_h: float):
        if diff_h == 0:
            return

        icon_name = "uphill.png" if diff_h > 0 else "downhill.png"
        img = self.load_image(icon_name)
        self.canvas.create_image(
            LCD_WIDTH - 115,
            LCD_HEIGHT // 2 - 30,
            image=img,
        )

        diff_str = f"{int(round(diff_h)):+d}"
        self.canvas.create_text(
            LCD_WIDTH - 110,
            LCD_HEIGHT // 2 - 70,
            text=diff_str,
            fill="white",
            font=("Helvetica", 25, "bold"),
        )

    def _on_shot(self, event):
        """
        SHOT 클릭 시:
          1) 현재 위치(E_cur, N_cur)와 보정된 current_alt 를 start_point 로 저장
          2) 이후부터 shot_distance 가 매 주기마다 갱신/표시됨
          3) 이동 중 다시 클릭하면 start_point 를 새로 설정
        """
        self._update_position_from_parent()

        parent_alt = getattr(self.parent_window, "altitude", 0.0)
        alt_offset = getattr(self.parent_window, "alt_offset", 0.0)
        current_alt = parent_alt - alt_offset

        self.start_E = self.E_cur
        self.start_N = self.N_cur
        self.start_alt = current_alt

        self.shot_active = True

        print(
            f"[DistanceWindow] SHOT start_point set: "
            f"E={self.start_E:.1f}, N={self.start_N:.1f}, alt={self.start_alt:.2f}"
        )

        self._render_screen()

    def _on_green_view(self, event=None):
        """
        GreenView 버튼 클릭 시 green_view.py 실행.
        - green_view.open_green_view_window(...) 를 호출한다.
        - DistanceWindow 의 selected_green(L/R) 정보를 그대로 전달한다.
        - parent 는 DistanceWindow(self)를 넘겨, STOP 시 이 창으로 쉽게 복귀 가능.
        """
        try:
            import green_view
        except ImportError:
            print("[DistanceWindow] green_view 모듈을 찾을 수 없습니다.")
            return

        try:
            if hasattr(green_view, "open_green_view_window"):
                green_view.open_green_view_window(
                    parent=self,  # DistanceWindow 를 부모로 사용
                    hole_row=self.hole_row,
                    gc_center_lat=self.gc_center_lat,
                    gc_center_lng=self.gc_center_lng,
                    cur_lat=self.cur_lat,
                    cur_lng=self.cur_lng,
                    selected_green=self.selected_green,
                )
            elif hasattr(green_view, "open_green_view"):
                # fallback: 다른 이름의 팩토리 함수가 있을 경우
                green_view.open_green_view(
                    parent=self,
                    hole_row=self.hole_row,
                    gc_center_lat=self.gc_center_lat,
                    gc_center_lng=self.gc_center_lng,
                    cur_lat=self.cur_lat,
                    cur_lng=self.cur_lng,
                    selected_green=self.selected_green,
                )
            elif hasattr(green_view, "main"):
                green_view.main()
            else:
                print(
                    "[DistanceWindow] green_view 모듈에 실행용 엔트리 함수 "
                    "(open_green_view_window / open_green_view / main) 가 없습니다."
                )
        except Exception as e:
            print("[DistanceWindow] green_view 실행 중 오류:", e)


def open_distance_window(
    parent: tk.Tk,
    hole_row: pd.Series,
    gc_center_lat: float,
    gc_center_lng: float,
    cur_lat: float,
    cur_lng: float,
) -> DistanceWindow:
    return DistanceWindow(parent, hole_row, gc_center_lat, gc_center_lng, cur_lat, cur_lng)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Distance Test")

    dummy_row = pd.Series(
        {
            "GC_name_code": 1002,
            "course_name": "아웃",
            "Hole": 1,
            "PAR": 4,
            "LG_E": 200, "LG_N": 150, "LG_dAlt": 0,
            "RG_E": np.nan, "RG_N": np.nan, "RG_dAlt": np.nan,
        }
    )

    root.latitude = 37.340218
    root.longitude = 126.940889
    root.altitude = 70.0
    root.alt_offset = 0.0

    win = DistanceWindow(
        parent=root,
        hole_row=dummy_row,
        gc_center_lat=37.34,
        gc_center_lng=126.94,
        cur_lat=37.340218,
        cur_lng=126.940889,
    )
    root.mainloop()
