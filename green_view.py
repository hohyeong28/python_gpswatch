import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import tkinter as tk
from PIL import Image, ImageTk

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

from config import LCD_WIDTH, LCD_HEIGHT  # 예: 466 x 466


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
    """
    - 현재 홀의 그린 컨투어(LG/RG)와 pin_position을 회전/스케일링해서 보여주는 창
    - 입력:
        * hole_row: DB 한 행 (LG/RG, LG1~20/RG1~20, T3/IP1/IP2 ENU 포함)
        * gc_center_lat/lng: ENU 변환을 위한 GC 중심 WGS84
        * cur_lat/lng: 현재 위치 WGS84
        * selected_green: 'L' or 'R'
      (pin_position 은 일단 center(LG 또는 RG)로 설정, 나중에 별도 값을 쓸 수 있음)
    """

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

        # ENU 변환
        self.tf = make_transformer(self.gc_center_lat, self.gc_center_lng)
        if self.tf is None:
            print("[GreenView] transformer 생성 실패")
            return

        # 현재 위치 ENU
        self.E_cur, self.N_cur = self.tf.transform(self.cur_lng, self.cur_lat)

        # 데이터 추출
        self.center_EN = self._get_green_center()
        self.pin_EN = self._get_pin_position()  # 최초에는 center와 동일
        self.green_points = self._get_green_points()
        self.origin_EN = self._get_origin_point()

        # 기하 계산 (world → rotated → screen)
        self._compute_geometry_and_draw()

    # --------- DB 필드 접근 --------- #

    def _get_EN_point(self, label: str) -> Optional[Tuple[float, float]]:
        e = self.hole_row.get(f"{label}_E", np.nan)
        n = self.hole_row.get(f"{label}_N", np.nan)
        if pd.isna(e) or pd.isna(n):
            return None
        return float(e), float(n)

    def _get_green_center(self) -> Tuple[float, float]:
        if self.selected_green == "L":
            label = "LG"
        else:
            label = "RG"
        p = self._get_EN_point(label)
        if p is not None:
            return p
        # center 필드가 없으면 에지 평균으로 대체
        pts = self._get_green_points()
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        # 마지막 fallback: 현재 위치
        return self.E_cur, self.N_cur

    def _get_pin_position(self) -> Tuple[float, float]:
        """
        나중에 핀 위치가 별도 필드/설정으로 주어지면 여기서 사용.
        현재는 center와 동일하게 사용.
        """
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
        """
        원점 ENU:
          - IP2 존재 시: IP2
          - 그 외, IP1 존재 시: IP1
          - 그 외, T3 존재 시: T3
          - 모두 없으면 현재 위치 사용
        """
        for label in ["IP2", "IP1", "T3"]:
            p = self._get_EN_point(label)
            if p is not None:
                return p
        return self.E_cur, self.N_cur

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
        target_angle = -math.pi / 2  # (0, -1) 방향 = 12시
        theta = target_angle - angle_v

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        def rot(p):
            x, y = p
            return (x * cos_t - y * sin_t, x * sin_t + y * cos_t)

        G_rot = [rot(p) for p in Gpts_rel]
        origin_rot = rot(origin_rel)  # 현재는 사용 안 하지만 남겨둠
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

        # 화면 중심
        cx = LCD_WIDTH / 2
        cy = LCD_HEIGHT / 2

        def world_to_screen(p):
            x, y = p
            sx = cx + x * scale
            sy = cy - y * scale  # y축 반전
            return sx, sy

        # 4) entry/front/back(p1,p2) 계산: P0 → pin 방향 ray와 그린 컨투어 교차
        P0 = cur_rot  # 회전 후 현재 위치
        Pin = pin_rot
        dir_vec = (Pin[0] - P0[0], Pin[1] - P0[1])
        norm = math.hypot(dir_vec[0], dir_vec[1])
        if norm == 0:
            dir_vec = (0.0, -1.0)
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
            intersections.sort(key=lambda x: x[1])  # t 기준 정렬
            p_front = intersections[0][0]
            p_back = intersections[-1][0]
        elif len(intersections) == 1:
            p_front = intersections[0][0]
            p_back = intersections[0][0]
        else:
            # 교차 없을 때는 핀과 가장 가까운 점들로 fallback
            distances = [(dist(Pin, p), p) for p in G_rot]
            distances.sort(key=lambda x: x[0])
            p_front = distances[0][1]
            p_back = distances[-1][1]

        entry_distance = dist(P0, p_front)

        # 5) green_pin_length / green_pin_width:
        #    - 길이 축: p_front-p_back 방향(a_dir)과 평행하면서 pin_position을 지나는 직선 ∩ 컨투어
        #    - 폭   축: 위 직선에 수직이면서 pin_position을 지나는 직선 ∩ 컨투어
        a = (p_back[0] - p_front[0], p_back[1] - p_front[1])
        a_norm = math.hypot(a[0], a[1])
        if a_norm == 0:
            # degenerate: 길이/폭 계산 불가
            green_pin_length = 0.0
            width_val = 0.0
            len_p1 = len_p2 = Pin
            p3 = p4 = Pin
        else:
            a_dir = (a[0] / a_norm, a[1] / a_norm)  # 길이축 방향 (front-back 방향)
            perp = (-a_dir[1], a_dir[0])  # 폭축 방향 (수직)

            # 5-1) 길이 축: line(Pin, a_dir) ∩ polygon
            length_inters: List[Tuple[Tuple[float, float], float]] = []
            for i in range(len(G_rot)):
                A = G_rot[i]
                B = G_rot[(i + 1) % len(G_rot)]
                res = segment_line_intersection(A, B, Pin, a_dir)
                if res is not None:
                    length_inters.append(res)

            if len(length_inters) >= 2:
                length_inters.sort(key=lambda x: x[1])  # t 기준 정렬
                len_p1 = length_inters[0][0]
                len_p2 = length_inters[-1][0]
            else:
                # 교차가 부족하면 기존 front/back을 fallback으로 사용
                len_p1 = p_front
                len_p2 = p_back

            green_pin_length = dist(len_p1, len_p2)

            # 5-2) 폭 축: line(Pin, perp) ∩ polygon
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

        # 6) 화면에 그리기

        # 6-1) 그린 컨투어
        contour_screen = []
        for p in G_rot:
            contour_screen.extend(world_to_screen(p))
        self.canvas.create_polygon(
            contour_screen,
            fill="#90EE90",  # 연두색
            outline="white",
            width=3,
        )

        # 6-2) 핀 위치 (빨간 점)
        pin_s = world_to_screen(pin_rot)
        self.canvas.create_oval(
            pin_s[0] - 4,
            pin_s[1] - 4,
            pin_s[0] + 4,
            pin_s[1] + 4,
            fill="red",
            outline="red",
        )

        # 6-3) 핀 중심 기준 십자 점선 (컨투어 밖으로 나가지 않음)
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

        # 6-4) 엔트리 삼각형: front edge 위치에 삼각형 (핀 방향)
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

        # 6-5) entry_distance 텍스트: 삼각형 아래쪽
        text_y = max(tip[1], left[1], right[1]) + 50
        self.canvas.create_text(
            cx,
            text_y,
            text=f"{int(round(entry_distance))}",
            fill="white",
            font=("Helvetica", 32, "bold"),
        )

        # 6-6) 상단/우측 숫자 + 홀/파 정보
        radius = LCD_WIDTH / 2
        self.canvas.create_text(
            cx,
            cy - radius + 80,
            text=f"{int(round(green_pin_length))}",
            fill="white",
            font=("Helvetica", 26, "bold"),
        )
        self.canvas.create_text(
            cx + radius - 80,
            cy,
            text=f"{int(round(width_val))}",
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


# ---------------- 단독 테스트 ---------------- #

if __name__ == "__main__":
    # 간단한 dummy 데이터로 모양/동작만 확인용
    root = tk.Tk()
    root.title("Green View Test")

    # ENU 기준 대략적인 타원형 그린 생성 (LG1~LG20)
    pts = []
    a, b = 25, 15  # m 단위 타원 반지름
    for k in range(20):
        t = 2 * math.pi * k / 20
        x = a * math.cos(t)
        y = b * math.sin(t)
        pts.append((x, y))

    data = {
        "Hole": 10,
        "PAR": 4,
        "LG_E": 0.0,
        "LG_N": 0.0,
    }
    for i, (x, y) in enumerate(pts, start=1):
        data[f"LG{i}_E"] = x
        data[f"LG{i}_N"] = y

    # IP2(원점)과 현재 위치를 그린 아래쪽에 배치
    data["IP2_E"] = 0.0
    data["IP2_N"] = -60.0  # 그린 아래쪽 60m 지점
    hole_row = pd.Series(data)

    # WGS84 는 테스트용 dummy, ENU 변환만 필요하므로 가까운 값 사용
    gc_lat, gc_lng = 37.34, 126.94
    cur_lat, cur_lng = 37.3401, 126.9401  # 현재 위치

    win = GreenViewWindow(
        parent=root,
        hole_row=hole_row,
        gc_center_lat=gc_lat,
        gc_center_lng=gc_lng,
        cur_lat=cur_lat,
        cur_lng=cur_lng,
        selected_green="L",
    )

    root.mainloop()
