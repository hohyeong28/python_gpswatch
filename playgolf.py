# playgolf.py
#
# - 해상도 : 466 x 466 (config.py 사용 가능 시 거기에 맞춰 수정 가능)
# - background.png 사용
# - 시작 시 5초간 "골프장 찾는 중 ..." + Find_Sat.png 표시
# - 이후 GC 탐색을 주기적으로 수행하여 현재 골프장을 결정하고, 골프장 이름을 화면 중앙에 표시
#   -> 3초간 표시 후 "티박스로 / 이동해 주세요" 안내 문구 3초 출력
#   -> 그 후 현재 홀을 찾는 작업을 주기적으로 수행하여 1st_hole fix 후 measure_distance 실행
#
# DB 파일 ( /DB/ 디렉토리 ):
#   Test_DB_Name_DB.csv  (GC_name_code, GC_name, center_lat, center_lng)
#   Test_DB.csv          (GC_name_code, ..., T1_E, T1_N, ...)

import math
from pathlib import Path
from typing import Optional, Dict

import tkinter as tk
from PIL import Image, ImageTk

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

from config import LCD_WIDTH, LCD_HEIGHT
from simul_gps import GPSSimulator

threshold_dist = 20

sim_gps_file = "남서울CC.xlsx"

# -------------------- Haversine 거리 계산 -------------------- #


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    위도/경도(deg) 두 점 사이의 Haversine 거리 (meters)
    """
    R = 6371000.0  # 지구 반지름 (m)

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def make_transformer(lat0: float, lon0: float) -> Optional[Transformer]:
    """
    WGS84 위도/경도를 중심(lat0, lon0) 기준의 평면(동/북, 미터) 좌표로 바꾸는 transformer 생성.
    (AEQD 투영)
    """
    if np.isnan(lat0) or np.isnan(lon0):
        return None
    wgs84 = CRS.from_epsg(4326)
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m")
    return Transformer.from_crs(wgs84, aeqd, always_xy=True)  # (lon, lat) → (E, N)


# -------------------- 선분 거리 유틸 -------------------- #


def point_segment_distance(px: float, py: float,
                           ax: float, ay: float,
                           bx: float, by: float) -> float:
    """
    점 P(px,py)와 선분 AB(ax,ay)-(bx,by) 사이의 최소 거리 (m)
    """
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    denom = abx * abx + aby * aby
    if denom == 0:
        # A와 B가 같은 점인 경우
        return math.hypot(px - ax, py - ay)

    t = (apx * abx + apy * aby) / denom
    if t < 0:
        cx, cy = ax, ay
    elif t > 1:
        cx, cy = bx, by
    else:
        cx = ax + t * abx
        cy = ay + t * aby

    return math.hypot(px - cx, py - cy)


# -------------------- Play Golf 창 -------------------- #


class PlayGolfWindow(tk.Toplevel):
    def __init__(self, master: tk.Tk):
        super().__init__(master)

        self.title("Play Golf")
        self.resizable(False, False)

        # 부모 창 기준 위치
        self.geometry(
            f"{LCD_WIDTH}x{LCD_HEIGHT}+"
            f"{master.winfo_rootx() + 20}+{master.winfo_rooty() + 20}"
        )

        # 현재 위치 (초기값)
        self.latitude = 37.339553
        self.longitude = 126.939744
        self.altitude = 69

        # alt 보정값 (GPS alt - fix_point DB alt)
        self.alt_offset: float = 0.0

        # 시뮬레이터 핸들 보관용
        self.gps_sim: Optional[GPSSimulator] = None

        # 이미지 캐시 (GC 방지)
        self.images: Dict[str, ImageTk.PhotoImage] = {}

        # 경로 설정
        self.base_dir = Path(__file__).parent
        self.asset_dir = self.base_dir / "assets_image"

        # DB 폴더 내의 파일 참조
        self.db_dir = self.base_dir / "DB"
        self.name_db_path = self.db_dir / "Test_DB_Name_DB.csv"
        self.db_path = self.db_dir / "Test_DB.csv"

        # ★ GPS 시뮬레이터 생성 및 시작
        gps_file = self.base_dir / sim_gps_file
        self.gps_sim = GPSSimulator(
            excel_path=gps_file,   # simul_gps.py 의 인자명 기준
            tk_root=self,          # after() 호출용
            on_update=self.on_gps_update,
        )
        if self.gps_sim.gps_df is not None:
            self.gps_sim.start()

        # 골프장 관련 상태
        self.current_gc_code = None
        self.current_gc_name = None
        self.gc_center_lat = None
        self.gc_center_lng = None
        self.gc_df: Optional[pd.DataFrame] = None

        # 캔버스 생성
        self.canvas = tk.Canvas(
            self,
            width=LCD_WIDTH,
            height=LCD_HEIGHT,
            highlightthickness=0,
            bg="black",
        )
        self.canvas.pack()

        # 배경 그리기
        self.draw_background()

        # Find_Sat 관련 ID
        self.find_sat_text_id = None
        self.find_sat_image_id = None

        # 골프장 이름 텍스트 ID
        self.gc_text_id = None

        # 초기 Find_Sat 화면 표시 후 5초 뒤 GC 탐색 루프 시작
        self.show_find_sat()
        self.after(5000, self.on_find_sat_finished)

    # ---------- GPS 시뮬레이터 콜백 ---------- #

    def on_gps_update(self, t: float, lat: float, lng: float, alt: float):
        """
        GPSSimulator 에서 1초마다 호출되는 콜백.
        여기서 PlayGolfWindow 의 현재 위치를 갱신한다.
        """
        self.latitude = lat
        self.longitude = lng
        self.altitude = alt

        print(
            f"[PlayGolf] GPS(t={t:.0f}s): "
            f"lat={self.latitude:.6f}, lng={self.longitude:.6f}, alt={self.altitude}"
        )

    # ---------- 이미지 로드 ---------- #

    def load_image(self, filename: str, size=None) -> ImageTk.PhotoImage:
        path = self.asset_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")

        img = Image.open(path).convert("RGBA")
        if size is not None:
            img = img.resize(size, Image.LANCZOS)

        photo = ImageTk.PhotoImage(img)
        self.images[filename] = photo
        return photo

    # ---------- 배경 ---------- #

    def draw_background(self):
        bg_img = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
        self.canvas.create_image(0, 0, anchor="nw", image=bg_img)

    # ---------- 위성 탐색 UI ---------- #

    def show_find_sat(self):
        """처음 5초간 '골프장 찾는 중 ...' + Find_Sat 표시"""

        # 텍스트 (이미지 위쪽)
        self.find_sat_text_id = self.canvas.create_text(
            LCD_WIDTH // 2 + 20,
            LCD_HEIGHT // 2 - 100,
            text="골프장 찾는 중 ...",
            fill="#2BBF3F",                # 녹색
            font=("Helvetica", 18, "bold"),
        )

        # 이미지 (중앙)
        find_img = self.load_image("Find_Sat.png")
        self.find_sat_image_id = self.canvas.create_image(
            LCD_WIDTH // 2,
            LCD_HEIGHT // 2,
            image=find_img,
        )

        print(
            f"[PlayGolf] 현재 위치 lat={self.latitude}, "
            f"lon={self.longitude}, alt={self.altitude}"
        )

    def on_find_sat_finished(self):
        """
        Find_Sat UI 제거 후 GC 탐색 루프 시작.
        """

        # Find_Sat 제거
        if self.find_sat_text_id is not None:
            self.canvas.delete(self.find_sat_text_id)
            self.find_sat_text_id = None

        if self.find_sat_image_id is not None:
            self.canvas.delete(self.find_sat_image_id)
            self.find_sat_image_id = None

        # GC 탐색 루프 시작
        self.start_find_gc_loop()

    # ---------- GC 탐색 루프 ---------- #

    def start_find_gc_loop(self):
        """
        GC가 fix 될 때까지 주기적으로 find_GC()를 호출.
        성공하면 이름을 표시하고 다음 단계로 진행.
        실패하면 일정 시간 후 재시도.
        """
        gc_name = self.find_GC()

        if gc_name is not None:
            # 찾은 골프장 이름 화면 중앙 표시 (녹색)
            self.gc_text_id = self.canvas.create_text(
                LCD_WIDTH // 2,
                LCD_HEIGHT // 2,
                text=gc_name,
                fill="#2BBF3F",
                font=("Helvetica", 20, "bold"),
            )
            print(f"[PlayGolf] 현재 골프장: {gc_name}")

            # 3초 후 "티박스로 / 이동해 주세요" 안내 표시
            self.after(3000, self.show_move_to_tee_box)
        else:
            # 실패 시: 로그 출력 후 일정 시간 뒤 재시도
            print("[PlayGolf] 3km 이내 골프장 없음, 5초 후 재탐색")
            self.after(5000, self.start_find_gc_loop)

    def show_move_to_tee_box(self):
        """골프장 이름 3초 표시 후 안내 문구로 전환"""

        # 기존 골프장 이름 텍스트 제거
        if self.gc_text_id is not None:
            self.canvas.delete(self.gc_text_id)
            self.gc_text_id = None

        # "티박스로" / "이동해 주세요" 안내 문구 표시
        self.canvas.create_text(
            LCD_WIDTH // 2,
            LCD_HEIGHT // 2 - 10,
            text="티박스로",
            fill="#2BBF3F",
            font=("Helvetica", 22, "bold"),
        )

        self.canvas.create_text(
            LCD_WIDTH // 2,
            LCD_HEIGHT // 2 + 30,
            text="이동해 주세요",
            fill="#2BBF3F",
            font=("Helvetica", 22, "bold"),
        )

        print("[PlayGolf] 티박스로 이동 안내 표시")

        # 1초 후 현재 홀 인식 루프 시작
        self.after(1000, self.start_find_1st_hole_loop)

    # ---------- 골프장 찾기 (find_GC) ---------- #

    def find_GC(self) -> Optional[str]:
        """
        현재 위/경도(self.latitude, self.longitude)를 기반으로 골프장 결정.

        1) Test_DB_Name_DB.csv 로드
        2) Haversine 거리로 3km 이내 GC 후보(candidate) 추출
        3) 후보 1개면 해당 GC 확정
        4) 후보 2개 이상이면 Test_DB.csv에서 각 후보 GC의 T1_E/T1_N과
           현재 위치(ENU 변환 후) 사이의 거리 최소값을 비교해 가장 가까운 GC 선택
        5) 골프장 fix 시 해당 GC의 모든 데이터를 self.gc_df에 로딩
        6) 최종 선택된 GC_name 반환 (없으면 None)
        """

        # ---------------- Name_DB 로드 ----------------
        if not self.name_db_path.exists():
            print(f"[ERROR] Name_DB 파일 없음: {self.name_db_path}")
            return None

        name_df = pd.read_csv(self.name_db_path)
        required_cols = ["GC_name_code", "GC_name", "center_lat", "center_lng"]
        for c in required_cols:
            if c not in name_df.columns:
                print(f"[ERROR] Name_DB에 필수 컬럼 누락: {c}")
                return None

        # float 변환
        name_df["center_lat"] = pd.to_numeric(name_df["center_lat"], errors="coerce")
        name_df["center_lng"] = pd.to_numeric(name_df["center_lng"], errors="coerce")

        # ---------------- 1차: 3km 이내 후보 선택 ----------------
        candidates = []

        for _, row in name_df.iterrows():
            gc_code = row["GC_name_code"]
            gc_name = row["GC_name"]
            clat = row["center_lat"]
            clng = row["center_lng"]

            if pd.isna(clat) or pd.isna(clng):
                continue

            d = haversine(self.latitude, self.longitude, clat, clng)
            if d <= 3000.0:
                candidates.append({
                    "GC_name_code": gc_code,
                    "GC_name": gc_name,
                    "center_lat": clat,
                    "center_lng": clng,
                    "distance": d,
                })

        if len(candidates) == 0:
            return None

        # DB 전체는 한 번만 로드 (후보가 여러 개일 때 필요)
        db_df = None

        # 후보 1개인 경우: 바로 fix
        if len(candidates) == 1:
            chosen = candidates[0]
            chosen_code = chosen["GC_name_code"]
            chosen_name = chosen["GC_name"]
            chosen_clat = chosen["center_lat"]
            chosen_clng = chosen["center_lng"]
        else:
            # ---------------- 2차: T1 기준 상세 판별 ----------------
            if not self.db_path.exists():
                print(f"[ERROR] DB 파일 없음: {self.db_path}")
                return None

            db_df = pd.read_csv(self.db_path)

            if "GC_name_code" not in db_df.columns:
                print("[ERROR] DB에 GC_name_code 컬럼 없음")
                return None

            if "T1_E" not in db_df.columns or "T1_N" not in db_df.columns:
                print("[ERROR] DB에 T1_E/T1_N 컬럼이 필요합니다.")
                return None

            best_gc_name = None
            best_gc_code = None
            best_center_lat = None
            best_center_lng = None
            best_min_t1_dist = float("inf")

            for cand in candidates:
                gc_code = cand["GC_name_code"]
                gc_name = cand["GC_name"]
                clat = cand["center_lat"]
                clng = cand["center_lng"]

                tf = make_transformer(clat, clng)
                if tf is None:
                    continue

                E_cur, N_cur = tf.transform(self.longitude, self.latitude)

                sub = db_df[db_df["GC_name_code"] == gc_code]
                if sub.empty:
                    continue

                min_t1_dist = float("inf")
                for _, row in sub.iterrows():
                    t1_e = row.get("T1_E", np.nan)
                    t1_n = row.get("T1_N", np.nan)
                    if pd.isna(t1_e) or pd.isna(t1_n):
                        continue

                    d = math.hypot(E_cur - t1_e, N_cur - t1_n)
                    if d < min_t1_dist:
                        min_t1_dist = d

                if min_t1_dist < best_min_t1_dist:
                    best_min_t1_dist = min_t1_dist
                    best_gc_name = gc_name
                    best_gc_code = gc_code
                    best_center_lat = clat
                    best_center_lng = clng

            if best_gc_name is None:
                return None

            chosen_name = best_gc_name
            chosen_code = best_gc_code
            chosen_clat = best_center_lat
            chosen_clng = best_center_lng

        # 여기까지 오면 chosen_* 에 최종 골프장 정보가 있음
        self.current_gc_code = chosen_code
        self.current_gc_name = chosen_name
        self.gc_center_lat = chosen_clat
        self.gc_center_lng = chosen_clng

        # 해당 GC의 모든 데이터 로딩
        if db_df is None:
            if not self.db_path.exists():
                print(f"[ERROR] DB 파일 없음: {self.db_path}")
                return None
            db_df = pd.read_csv(self.db_path)

        if "GC_name_code" not in db_df.columns:
            print("[ERROR] DB에 GC_name_code 컬럼 없음")
            return None

        self.gc_df = db_df[db_df["GC_name_code"] == self.current_gc_code].copy()
        print(
            f"[PlayGolf] GC fix: {self.current_gc_name}, "
            f"rows={len(self.gc_df)}"
        )

        return self.current_gc_name

    # ---------- 현재 홀 찾기 루프 ---------- #

    def start_find_1st_hole_loop(self):
        """
        현재 홀을 찾을 때까지 주기적으로 find_1st_hole() 호출.
        성공하면 loop 종료, 실패 시 일정 시간 후 재시도.
        """
        ok = self.find_1st_hole()
        if ok:
            print("[find_1st_hole] 홀 fix 완료, 반복 종료")
            return
        else:
            self.after(1000, self.start_find_1st_hole_loop)

    # ---------- 현재 홀 찾기 (find_1st_hole) ---------- #

    def log_current_position(self, prefix: str = ""):
        if prefix:
            prefix = f" [{prefix}]"
        print(
            f"[PlayGolf]{prefix} "
            f"lat={self.latitude:.6f}, lng={self.longitude:.6f}, alt={self.altitude}"
        )

    def find_1st_hole(self) -> bool:
        """
        1. self.gc_df (해당 골프장 전체 홀 DB)를 기준으로 각 행(각 홀)을 검사
        2. 각 홀에 대해 다음 포인트를 이용해 직선을 생성:
           - 프론트: T1~T2, T2~T3 (존재하는 경우)
           - 백/접근: T5~T6~IP1~IP2~GC(그린 중앙) 체인
             (포인트가 없으면 스킵하고 다음 유효 포인트와 직접 연결)
        3. 현재 위치를 ENU로 변환 후 각 직선에 대한 최소 거리 계산
        4. 최소 거리 <= threshold_dist 인 홀들 중에서 가장 가까운 홀을 현재 홀로 fix
        5. 성공 시 True, 실패 시 False 반환
        """
        # ★ 현재 위경도 로그 출력 (WGS84)
        self.log_current_position(prefix="find_1st_hole - WGS84")

        if self.gc_df is None or self.gc_df.empty:
            print("[find_1st_hole] GC 데이터 없음 (gc_df 비어 있음)")
            return False

        if self.gc_center_lat is None or self.gc_center_lng is None:
            print("[find_1st_hole] GC 중심 좌표 없음")
            return False

        # 현재 위치 ENU 변환 (해당 GC 기준)
        tf = make_transformer(self.gc_center_lat, self.gc_center_lng)
        if tf is None:
            print("[find_1st_hole] Transformer 생성 실패")
            return False

        E_cur, N_cur = tf.transform(self.longitude, self.latitude)

        def get_point(row, label: str):
            """row에서 label_E, label_N을 읽어 ENU 좌표 반환 (없거나 NaN이면 None)"""
            e = row.get(f"{label}_E", np.nan)
            n = row.get(f"{label}_N", np.nan)
            if pd.isna(e) or pd.isna(n):
                return None
            return float(e), float(n)

        def get_green_center(row):
            """LG_E/N, RG_E/N로부터 그린 중앙 좌표 계산"""
            lg_e = row.get("LG_E", np.nan)
            lg_n = row.get("LG_N", np.nan)
            rg_e = row.get("RG_E", np.nan)
            rg_n = row.get("RG_N", np.nan)

            lg_valid = not (pd.isna(lg_e) or pd.isna(lg_n))
            rg_valid = not (pd.isna(rg_e) or pd.isna(rg_n))

            if lg_valid and rg_valid:
                return (float(lg_e + rg_e) / 2.0, float(lg_n + rg_n) / 2.0)
            elif lg_valid:
                return float(lg_e), float(lg_n)
            elif rg_valid:
                return float(rg_e), float(rg_n)
            else:
                return None

        def build_chain_segments(points_order, row):
            """
            points_order: ["T5","T6","IP1","IP2","GC"] 와 같이 포인트 이름 리스트
            row: 한 홀의 데이터
            존재하는 포인트들끼리만 인접 선분 생성 (중간 포인트 없으면 스킵하고 다음 것으로 연결)
            """
            pts = {}
            for name in points_order:
                if name == "GC":
                    pts[name] = get_green_center(row)
                else:
                    pts[name] = get_point(row, name)

            segs = []
            prev = None
            for name in points_order:
                p = pts[name]
                if p is None:
                    continue
                if prev is not None:
                    segs.append((prev, p))
                prev = p
            return segs

        best_row = None
        best_dist = float("inf")

        for _, row in self.gc_df.iterrows():
            segments = []

            # 프론트(T1,T2,T3) 체인
            segments += build_chain_segments(["T1", "T2", "T3"], row)
            # 백/접근(T5,T6,IP1,IP2,GC) 체인
            segments += build_chain_segments(["T5", "T6", "IP1", "IP2", "GC"], row)

            if not segments:
                continue

            hole_min = float("inf")
            for (a, b) in segments:
                dist = point_segment_distance(E_cur, N_cur, a[0], a[1], b[0], b[1])
                if dist < hole_min:
                    hole_min = dist

            # Threshold 이내 + 전역 최소 거리 갱신
            if hole_min <= threshold_dist and hole_min < best_dist:
                best_dist = hole_min
                best_row = row

        if best_row is None:
#            print(f"[find_1st_hole] {threshold_dist}m 이내에 어느 홀의 동선도 없음 (현재 홀 판단 불가)")
            return False

        # 현재 홀 fix
        gc_code = best_row.get("GC_name_code")
        course_name = best_row.get("course_name", "")
        hole = best_row.get("Hole", "")

        print(
            f"[find_1st_hole] 현재 홀 fix: "
            f"GC_code={gc_code}, course={course_name}, hole={hole}, "
            f"dist≈{best_dist:.1f} m"
        )

        # ---------- alt_offset 계산 ----------
        # 1) fix_point 후보(T1~T6, IP1, IP2) 중 현재 ENU 위치(E_cur,N_cur)에 가장 가까운 포인트 선택
        # 2) 해당 포인트의 DB 고도(dAlt)를 fix_point_db_alt 로 사용
        # 3) alt_offset = GPS_alt_at_fix - fix_point_db_alt

        def get_label_alt(row, label: str) -> Optional[float]:
            a = row.get(f"{label}_dAlt", np.nan)
            if pd.isna(a):
                return None
            return float(a)

        fix_labels = ["T1", "T2", "T3", "T4", "T5", "T6", "IP1", "IP2"]
        best_fix_dist = float("inf")
        best_fix_alt_db: Optional[float] = None
        best_fix_label: Optional[str] = None

        for lab in fix_labels:
            p = get_point(best_row, lab)
            if p is None:
                continue
            d = math.hypot(E_cur - p[0], N_cur - p[1])
            alt_db = get_label_alt(best_row, lab)
            if alt_db is None:
                continue
            if d < best_fix_dist:
                best_fix_dist = d
                best_fix_alt_db = alt_db
                best_fix_label = lab

        if best_fix_alt_db is not None:
            gps_alt_at_fix = self.altitude  # find_1st_hole fix 시점의 GPS alt
            self.alt_offset = gps_alt_at_fix - best_fix_alt_db
            print(
                f"[find_1st_hole] alt_offset 계산: "
                f"fix_label={best_fix_label}, "
                f"GPS_alt={gps_alt_at_fix:.2f}, "
                f"fix_alt_db={best_fix_alt_db:.2f}, "
                f"alt_offset={self.alt_offset:.2f}"
            )
        else:
            # 적절한 fix_point alt 를 찾지 못한 경우, 보정 0으로
            self.alt_offset = 0.0
            print("[find_1st_hole] fix_point alt 없음 → alt_offset = 0.0")

        # -----------------------------------

        try:
            from measure_distance import open_distance_window
            open_distance_window(
                parent=self,
                hole_row=best_row,
                gc_center_lat=self.gc_center_lat,
                gc_center_lng=self.gc_center_lng,
                cur_lat=self.latitude,
                cur_lng=self.longitude,
            )
        except Exception as e:
            print(f"[find_1st_hole] measure_distance 호출 중 오류: {e}")

        return True

    def find_next_hole(self) -> bool:
        """
        다음 홀 탐색 함수.

        1. self.gc_df (해당 골프장 전체 홀 DB)를 기준으로 각 행(각 홀)을 검사
        2. 각 홀에 대해 T1~T6 포인트들만 이용해 직선 체인 생성:
             - ["T1", "T2", "T3", "T4", "T5", "T6"] 순서로 존재하는 포인트를 이어 선분 목록 생성
        3. 현재 위치를 ENU로 변환 후 각 직선에 대한 최소 거리 계산
        4. 최소 거리 <= threshold_dist 인 홀들 중에서 가장 가까운 홀을 "다음 홀"로 fix
        5. alt_offset 재계산 후 measure_distance 화면 호출
        6. 성공 시 True, 실패 시 False 반환
        """

        # ★ 현재 위경도 로그 출력 (WGS84)
        self.log_current_position(prefix="find_next_hole - WGS84")

        if self.gc_df is None or self.gc_df.empty:
            print("[find_next_hole] GC 데이터 없음 (gc_df 비어 있음)")
            return False

        if self.gc_center_lat is None or self.gc_center_lng is None:
            print("[find_next_hole] GC 중심 좌표 없음")
            return False

        # 현재 위치 ENU 변환 (해당 GC 기준)
        tf = make_transformer(self.gc_center_lat, self.gc_center_lng)
        if tf is None:
            print("[find_next_hole] Transformer 생성 실패")
            return False

        # self.longitude, self.latitude 는 WGS84 좌표라고 가정
        E_cur, N_cur = tf.transform(self.longitude, self.latitude)

        def get_point(row, label: str):
            """row에서 label_E, label_N을 읽어 ENU 좌표 반환 (없거나 NaN이면 None)"""
            e = row.get(f"{label}_E", np.nan)
            n = row.get(f"{label}_N", np.nan)
            if pd.isna(e) or pd.isna(n):
                return None
            return float(e), float(n)

        def build_chain_segments(points_order, row):
            """
            points_order: ["T1","T2","T3","T4","T5","T6"] 와 같이 포인트 이름 리스트
            row: 한 홀의 데이터
            존재하는 포인트들끼리만 인접 선분 생성 (중간 포인트 없으면 스킵하고 다음 것으로 연결)
            """
            pts = {}
            for name in points_order:
                pts[name] = get_point(row, name)

            segs = []
            prev = None
            for name in points_order:
                p = pts[name]
                if p is None:
                    continue
                if prev is not None:
                    segs.append((prev, p))
                prev = p
            return segs

        best_row = None
        best_dist = float("inf")

        # threshold_dist 와 point_segment_distance 는 find_1st_hole 과 동일하게 사용
        for _, row in self.gc_df.iterrows():
            segments = []

            # T1~T6 체인만 사용 (IP1, IP2, GC는 사용하지 않음)
            segments += build_chain_segments(["T1", "T2", "T3", "T4", "T5", "T6"], row)

            if not segments:
                continue

            hole_min = float("inf")
            for (a, b) in segments:
                dist = point_segment_distance(E_cur, N_cur, a[0], a[1], b[0], b[1])
                if dist < hole_min:
                    hole_min = dist

            # Threshold 이내 + 전역 최소 거리 갱신
            if hole_min <= threshold_dist and hole_min < best_dist:
                best_dist = hole_min
                best_row = row

        if best_row is None:
            # print(f"[find_next_hole] {threshold_dist}m 이내에 어느 홀의 T1~T6도 없음 (다음 홀 판단 불가)")
            return False

        # 다음 홀 fix
        gc_code = best_row.get("GC_name_code")
        course_name = best_row.get("course_name", "")
        hole = best_row.get("Hole", "")

        print(
            f"[find_next_hole] 다음 홀 fix: "
            f"GC_code={gc_code}, course={course_name}, hole={hole}, "
            f"dist≈{best_dist:.1f} m"
        )

        # ---------- alt_offset 계산 ----------
        # 1) fix_point 후보(T1~T6, IP1, IP2) 중 현재 ENU 위치(E_cur,N_cur)에 가장 가까운 포인트 선택
        # 2) 해당 포인트의 DB 고도(dAlt)를 fix_point_db_alt 로 사용
        # 3) alt_offset = GPS_alt_at_fix - fix_point_db_alt

        def get_label_alt(row, label: str) -> Optional[float]:
            a = row.get(f"{label}_dAlt", np.nan)
            if pd.isna(a):
                return None
            return float(a)

        fix_labels = ["T1", "T2", "T3", "T4", "T5", "T6", "IP1", "IP2"]
        best_fix_dist = float("inf")
        best_fix_alt_db: Optional[float] = None
        best_fix_label: Optional[str] = None

        for lab in fix_labels:
            p = get_point(best_row, lab)
            if p is None:
                continue
            d = math.hypot(E_cur - p[0], N_cur - p[1])
            alt_db = get_label_alt(best_row, lab)
            if alt_db is None:
                continue
            if d < best_fix_dist:
                best_fix_dist = d
                best_fix_alt_db = alt_db
                best_fix_label = lab

        if best_fix_alt_db is not None:
            gps_alt_at_fix = self.altitude  # find_next_hole fix 시점의 GPS alt
            self.alt_offset = gps_alt_at_fix - best_fix_alt_db
            print(
                f"[find_next_hole] alt_offset 계산: "
                f"fix_label={best_fix_label}, "
                f"GPS_alt={gps_alt_at_fix:.2f}, "
                f"fix_alt_db={best_fix_alt_db:.2f}, "
                f"alt_offset={self.alt_offset:.2f}"
            )
        else:
            # 적절한 fix_point alt 를 찾지 못한 경우, 보정 0으로
            self.alt_offset = 0.0
            print("[find_next_hole] fix_point alt 없음 → alt_offset = 0.0")

        # -----------------------------------

        try:
            from measure_distance import open_distance_window
            open_distance_window(
                parent=self,
                hole_row=best_row,
                gc_center_lat=self.gc_center_lat,
                gc_center_lng=self.gc_center_lng,
                cur_lat=self.latitude,
                cur_lng=self.longitude,
            )
        except Exception as e:
            print(f"[find_next_hole] measure_distance 호출 중 오류: {e}")

        return True


# 외부에서 entry_menu.py 등이 호출할 helper 함수
def open_play_golf_window(parent: tk.Tk) -> PlayGolfWindow:
    return PlayGolfWindow(parent)


# 단독 실행 테스트용
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Play Golf Standalone")
    win = PlayGolfWindow(root)
    root.mainloop()
