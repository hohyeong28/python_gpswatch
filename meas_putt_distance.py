# meas_putt_distance.py
#
# 퍼팅 거리 측정 화면
# - DistanceWindow.check_ongreen() 에서 on_green 확정 후 호출된다.
# - 초기 화면:
#     * background.png 전체 배경
#     * 상단: hole&falg.png
#     * 중앙: start_putt.png (START 버튼)
#     * 하단: "퍼팅 거리" (텍스트)
# - START 클릭 후:
#     * START 버튼은 사라짐
#     * 중앙에는 "거리 + M" 텍스트
#     * 텍스트 위/아래에 from_to.png 2개 (세로 배치)
#     * 그 아래 ball.png
#     * 하단: "퍼팅 거리"
# - putt_start: START 클릭 순간의 lat/lng
# - 이후 putt_start ~ 현재 위치까지의 거리를 1초마다 계산해 중앙에 표시
# - 그린 내부(on_green) 상태에서 그린 외부로 나간 뒤 3초가 지나면 scoring.py 호출

import time
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import tkinter as tk
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
from pyproj import CRS, Transformer

from config import LCD_WIDTH, LCD_HEIGHT


# ---------------- 좌표 변환 / 거리 유틸 ---------------- #

def make_transformer(lat0: float, lon0: float) -> Optional[Transformer]:
    if np.isnan(lat0) or np.isnan(lon0):
        return None
    wgs84 = CRS.from_epsg(4326)
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m")
    return Transformer.from_crs(wgs84, aeqd, always_xy=True)  # (lon, lat) → (E, N)


def euclidean_dist(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


class PuttDistanceWindow(tk.Toplevel):
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

        self.title("Putt Distance")
        self.resizable(False, False)

        # 부모 창 근처에 배치
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

        self.base_dir = Path(__file__).parent
        self.asset_dir = self.base_dir / "assets_image"

        # 이미지 캐시
        self.images: Dict[str, ImageTk.PhotoImage] = {}

        # 캔버스 생성
        self.canvas = tk.Canvas(
            self,
            width=LCD_WIDTH,
            height=LCD_HEIGHT,
            highlightthickness=0,
            bg="black",
        )
        self.canvas.pack()

        # ENU 변환용 transformer
        self.tf = make_transformer(self.gc_center_lat, self.gc_center_lng)
        if self.tf is None:
            print("[PuttDistanceWindow] transformer 생성 실패")
            return

        # 현재 위치 ENU
        self.E_cur: float = 0.0
        self.N_cur: float = 0.0

        # START 클릭 시점의 위치 (putt_start)
        self.putt_start_lat: Optional[float] = None
        self.putt_start_lng: Optional[float] = None
        self.putt_start_E: Optional[float] = None
        self.putt_start_N: Optional[float] = None

        # 측정 모드 여부
        self.measuring: bool = False
        self.current_putt_dist: float = 0.0  # meter 단위, 소수 1자리 표현

        # 현재 사용 중인 그린(L/R) 및 그린 폴리곤
        self.selected_green: str = self._detect_selected_green()
        self.green_polygon: List[Tuple[float, float]] = self._build_green_polygon()

        # 그린 이탈(out-of-green) 상태 관리
        self.out_of_green_since: Optional[float] = None
        self.scoring_called: bool = False

        # 최초 위치 업데이트
        self._update_position_from_parent()

        # 화면 렌더링
        self._render_screen()

        # 주기적 위치/거리 갱신
        self.after(1000, self._auto_update_loop)

    # ---------------- RG 존재 여부 / 사용 그린 결정 ---------------- #

    def _check_rg_exists(self) -> bool:
        rg_e = self.hole_row.get("RG_E", np.nan)
        rg_n = self.hole_row.get("RG_N", np.nan)
        return not (pd.isna(rg_e) or pd.isna(rg_n))

    def _detect_selected_green(self) -> str:
        """
        PlayGolfWindow.last_green 과 RG 존재 여부를 기반으로
        현재 퍼팅에 사용 중인 그린(L/R)을 결정한다.
        RG가 없으면 무조건 L 사용.
        """
        has_rg = self._check_rg_exists()
        last = getattr(self.parent_window, "last_green", "L")
        if not has_rg:
            return "L"
        if last not in ("L", "R"):
            return "L"
        return last

    def _build_green_polygon(self) -> List[Tuple[float, float]]:
        """
        선택된 그린(L/R)의 LG1~20 또는 RG1~20 좌표로 폐곡선 폴리곤 구성.
        ENU 좌표계 (E, N) 그대로 사용.
        """
        prefix = "LG" if self.selected_green == "L" else "RG"
        poly: List[Tuple[float, float]] = []

        for k in range(1, 21):
            e = self.hole_row.get(f"{prefix}{k}_E", np.nan)
            n = self.hole_row.get(f"{prefix}{k}_N", np.nan)
            if pd.isna(e) or pd.isna(n):
                continue
            poly.append((float(e), float(n)))

        if len(poly) < 3:
            print("[PuttDistanceWindow] 그린 폴리곤 포인트 부족:", len(poly))
        return poly

    # ---------------- 위치 갱신 ---------------- #

    def _update_position_from_parent(self):
        lat = getattr(self.parent_window, "latitude", self.cur_lat)
        lng = getattr(self.parent_window, "longitude", self.cur_lng)

        self.cur_lat = lat
        self.cur_lng = lng

        if self.tf is not None:
            self.E_cur, self.N_cur = self.tf.transform(self.cur_lng, self.cur_lat)

    def _auto_update_loop(self):
        self._update_position_from_parent()

        # 측정 모드일 때만 퍼팅 거리 갱신
        if self.measuring and self.putt_start_E is not None and self.putt_start_N is not None:
            flat = euclidean_dist(
                (self.E_cur, self.N_cur),
                (self.putt_start_E, self.putt_start_N),
            )
            # ENU 단위는 m 이므로 그대로 사용, 소수 1자리로 표시
            self.current_putt_dist = round(flat * 10.0) / 10.0

        # 그린 이탈 여부 검사
        self._check_out_of_green()

        self._render_screen()
        self.after(1000, self._auto_update_loop)

    # ---------------- 이미지 로드 ---------------- #

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

    # ---------------- 그린 내부/외부 판정 ---------------- #

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

            if (y1 > y) != (y2 > y):
                denom = (y2 - y1)
                if abs(denom) < 1e-12:
                    continue
                x_int = (x2 - x1) * (y - y1) / denom + x1
                if x_int > x:
                    inside = not inside

        return inside

    def _check_out_of_green(self):
        """
        현재 위치가 선택된 그린 폴리곤 밖으로 나갔는지 검사.
        - 폴리곤 내부 → out_of_green_since 리셋
        - 폴리곤 외부 → 최초 이탈 시각 기록
        - 3초 이상 연속 외부이면 scoring.py 호출 (1회만)
        """
        if self.scoring_called:
            return
        if not self.green_polygon:
            # 폴리곤이 없으면 이탈 판정 불가
            return

        inside = self._is_point_in_polygon(self.E_cur, self.N_cur, self.green_polygon)

        if inside:
            # 다시 그린 내부 → 타이머 리셋
            self.out_of_green_since = None
            return

        # 그린 외부
        now = time.time()
        if self.out_of_green_since is None:
            # 처음 나간 시점 기록
            self.out_of_green_since = now
            return

        elapsed = now - self.out_of_green_since
        if elapsed < 3.0:
            # 아직 3초 미만
            return

        # 3초 이상 외부 상태 유지 → scoring 화면 호출
        self.scoring_called = True
        print("[PuttDistanceWindow] 그린 이탈 3초 경과 → scoring.py 호출")

        try:
            from scoring import open_scoring_window
            open_scoring_window(parent=self.parent_window, hole_row=self.hole_row)
        except ImportError:
            print("[PuttDistanceWindow] scoring 모듈을 찾을 수 없습니다.")
        except Exception as e:
            print("[PuttDistanceWindow] scoring 실행 중 오류:", e)

    # ---------------- 화면 렌더링 ---------------- #

    def _render_screen(self):
        self.canvas.delete("all")

        # background.png 전체 적용
        try:
            bg_img = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
            self.canvas.create_image(0, 0, anchor="nw", image=bg_img)
        except FileNotFoundError:
            pass

        center_y = LCD_HEIGHT // 2

        # 상단: hole&falg.png (두 모드 공통, 상단에 배치)
        try:
            flag_img = self.load_image("hole&falg.png")
            self.canvas.create_image(
                LCD_WIDTH // 2,
                center_y - 140,
                image=flag_img,
            )
        except FileNotFoundError as e:
            print(e)

        if not self.measuring:
            # -------- 초기 화면 (START 버튼) -------- #

            try:
                start_img = self.load_image("start_putt.png")
                start_id = self.canvas.create_image(
                    LCD_WIDTH // 2,
                    center_y,
                    image=start_img,
                )
                # START 버튼 클릭 시 퍼팅 거리 측정 시작
                self.canvas.tag_bind(start_id, "<Button-1>", self._on_start_putt)
            except FileNotFoundError as e:
                print(e)

            # 하단 텍스트: "퍼팅 거리"
            self.canvas.create_text(
                LCD_WIDTH // 2,
                center_y + 140,
                text="퍼팅 거리",
                fill="white",
                font=("Helvetica", 20, "bold"),
            )

        else:
            # -------- 측정 화면 -------- #

            # 중앙 거리 텍스트 (예: "9.5")
            dist_str = f"{self.current_putt_dist:.1f}"
            self.canvas.create_text(
                LCD_WIDTH // 2,
                center_y,
                text=dist_str,
                fill="white",
                font=("Helvetica", 40, "bold"),
            )

            # 단위 "M"
            self.canvas.create_text(
                LCD_WIDTH // 2 + 80,
                center_y,
                text="M",
                fill="white",
                font=("Helvetica", 30, "bold"),
            )

            # from_to.png 위/아래 2개 배치
            try:
                from_to_img = self.load_image("from_to.png")
                # 위쪽 from_to (flag와 거리 사이)
                self.canvas.create_image(
                    LCD_WIDTH // 2,
                    center_y - 70,
                    image=from_to_img,
                )
                # 아래쪽 from_to (거리와 공 사이)
                self.canvas.create_image(
                    LCD_WIDTH // 2,
                    center_y + 60,
                    image=from_to_img,
                )
            except FileNotFoundError as e:
                print(e)

            # ball.png (거리 아래)
            try:
                ball_img = self.load_image("ball.png")
                self.canvas.create_image(
                    LCD_WIDTH // 2,
                    center_y + 110,
                    image=ball_img,
                )
            except FileNotFoundError as e:
                print(e)

            # 하단 텍스트: "퍼팅 거리"
            self.canvas.create_text(
                LCD_WIDTH // 2,
                center_y + 150,
                text="퍼팅 거리",
                fill="white",
                font=("Helvetica", 20, "bold"),
            )

    # ---------------- 이벤트 핸들러 ---------------- #

    def _on_start_putt(self, event):
        """
        START 버튼 클릭:
          1) 현재 위치(lat/lng, ENU)를 putt_start 로 기록
          2) 측정 모드로 전환
          3) 이후 _auto_update_loop 에서 거리 갱신
        """
        # 최신 위치로 갱신
        self._update_position_from_parent()

        self.putt_start_lat = self.cur_lat
        self.putt_start_lng = self.cur_lng
        self.putt_start_E = self.E_cur
        self.putt_start_N = self.N_cur

        self.measuring = True
        self.current_putt_dist = 0.0

        print(
            f"[PuttDistanceWindow] Putt start set: "
            f"lat={self.putt_start_lat:.7f}, lng={self.putt_start_lng:.7f}, "
            f"E={self.putt_start_E:.2f}, N={self.putt_start_N:.2f}"
        )

        self._render_screen()


def open_putt_window(
    parent: tk.Tk,
    hole_row: pd.Series,
    gc_center_lat: float,
    gc_center_lng: float,
    cur_lat: float,
    cur_lng: float,
) -> PuttDistanceWindow:
    """
    DistanceWindow.check_ongreen() 에서 호출할 진입 함수.
    """
    return PuttDistanceWindow(
        parent=parent,
        hole_row=hole_row,
        gc_center_lat=gc_center_lat,
        gc_center_lng=gc_center_lng,
        cur_lat=cur_lat,
        cur_lng=cur_lng,
    )


if __name__ == "__main__":
    # 단독 테스트용
    root = tk.Tk()
    root.title("Putt Distance Test")

    dummy_row = pd.Series(
        {
            "GC_name_code": 1002,
            "course_name": "아웃",
            "Hole": 1,
            "PAR": 4,
        }
    )

    root.latitude = 37.340218
    root.longitude = 126.940889

    win = open_putt_window(
        parent=root,
        hole_row=dummy_row,
        gc_center_lat=37.34,
        gc_center_lng=126.94,
        cur_lat=37.340218,
        cur_lng=126.940889,
    )

    root.mainloop()
