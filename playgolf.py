# playgolf.py (Frame + on_open_distance 콜백 버전)
#
# - 새 윈도우 생성 없음(Toplevel 없음)
# - GPS 시뮬레이터(simul_gps.py)는 Play Golf 화면 "선택 후(start 호출)"부터 실행
# - BACK/화면 이탈 시 stop()으로 GPS 시뮬레이터 및 after 루프 종료
# - find_1st_hole()에서 DistanceScreen 전환은 ScreenManager 콜백(on_open_distance)으로 위임
# - find_next_hole 로직은 find_next_hole.py(NextHoleFinder) 기반으로 PlayGolfFrame에서 수행
#   * 티 클러스터(T1~T6 최소거리) 방식
#   * 후보 감지 시 5~10초 주기 + 누적 60초 확정
#   * out-of-green 확정 + 그린 폴리곤 최소거리 30m 이상 이탈 후 탐색 활성화
# - 기존 기능은 유지(FindSat/GC 탐색/1st hole fix/alt_offset 계산/화면 전환 흐름)

import math
import time
from pathlib import Path
from typing import Optional, Dict, Callable, List, Tuple

import tkinter as tk
from PIL import Image, ImageTk

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

from config import LCD_WIDTH, LCD_HEIGHT
from simul_gps import GPSSimulator

# [추가] Next hole finder
from find_next_hole import NextHoleFinder, NextHoleConfig

threshold_dist = 20
sim_gps_file = "남서울CC.xlsx"


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def make_transformer(lat0: float, lon0: float) -> Optional[Transformer]:
    if np.isnan(lat0) or np.isnan(lon0):
        return None
    wgs84 = CRS.from_epsg(4326)
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m")
    return Transformer.from_crs(wgs84, aeqd, always_xy=True)


def point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    denom = abx * abx + aby * aby
    if denom == 0:
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


class PlayGolfFrame(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        on_back: Optional[Callable[[], None]] = None,
        on_open_distance: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(master, width=LCD_WIDTH, height=LCD_HEIGHT, bg="black")
        self.pack_propagate(False)

        self.on_back = on_back
        self.on_open_distance = on_open_distance

        # 현재 위치(초기값)
        self.latitude = 37.339553
        self.longitude = 126.939744
        self.altitude = 69

        self.alt_offset: float = 0.0
        self.last_green: str = "L"

        # [추가] 현재 홀 row (1st hole fix 후 저장)
        self.current_hole_row: Optional[pd.Series] = None

        # [추가] out-of-green 확정 플래그 (putt/score 흐름에서 세팅)
        self.out_of_green_confirmed: bool = False

        # [추가] NextHoleFinder 및 루프 제어
        self._next_finder: Optional[NextHoleFinder] = None
        self._after_next_hole: Optional[str] = None
        self._next_interval_ms_normal: int = 60000     # 평상시 60초
        self._next_interval_ms_candidate: int = 5000   # 후보 감지 시 5초(요구 5~10초 중 5초로 고정)
        self._next_interval_ms: int = self._next_interval_ms_normal

        # 이미지 캐시
        self.images: Dict[str, ImageTk.PhotoImage] = {}

        # 경로/DB
        self.base_dir = Path(__file__).parent
        self.asset_dir = self.base_dir / "assets_image"
        self.db_dir = self.base_dir / "DB"
        self.name_db_path = self.db_dir / "Test_DB_Name_DB.csv"
        self.db_path = self.db_dir / "Test_DB.csv"

        # 골프장 상태
        self.current_gc_code = None
        self.current_gc_name = None
        self.gc_center_lat = None
        self.gc_center_lng = None
        self.gc_df: Optional[pd.DataFrame] = None

        # 캔버스
        self.canvas = tk.Canvas(self, width=LCD_WIDTH, height=LCD_HEIGHT, highlightthickness=0, bg="black")
        self.canvas.pack()

        # BACK
        self.draw_background()
        self._draw_back_hitbox()

        # Find_Sat 관련 ID
        self.find_sat_text_id = None
        self.find_sat_image_id = None
        self.gc_text_id = None

        # GPS simulator (생성만 하고 start는 start()에서)
        self.gps_sim: Optional[GPSSimulator] = None
        self._gps_started: bool = False

        # after() 예약 id들(화면 이탈 시 cancel)
        self._after_find_sat: Optional[str] = None
        self._after_find_gc: Optional[str] = None
        self._after_move_to_tee: Optional[str] = None
        self._after_find_hole: Optional[str] = None

        # 실행 상태
        self._running: bool = False

    # ---------------- lifecycle ---------------- #

    def start(self):
        """
        entry_menu(ScreenManager)에서 playgolf 화면으로 전환될 때 호출.
        - GPS 시뮬레이터 start
        - FindSat UI 표시 및 5초 후 GC 탐색 시작
        """
        if self._running:
            return
        self._running = True

        # GPS sim 생성/시작
        if self.gps_sim is None:
            gps_file = self.base_dir / sim_gps_file
            self.gps_sim = GPSSimulator(
                excel_path=gps_file,
                tk_root=self,      # after() 호출용
                on_update=self.on_gps_update,
            )

        if (self.gps_sim is not None) and (self.gps_sim.gps_df is not None) and (not self._gps_started):
            self.gps_sim.start()
            self._gps_started = True

        # FindSat 시작
        self.show_find_sat()

        # 5초 후 GC 탐색으로 전환
        self._cancel_after(self._after_find_sat)
        self._after_find_sat = self.after(5000, self.on_find_sat_finished)

    def stop(self):
        """
        entry_menu(ScreenManager)에서 playgolf 화면을 떠날 때 호출.
        - GPS 시뮬레이터 stop
        - 예약된 after() 모두 취소
        """
        self._running = False

        # after 취소
        self._cancel_after(self._after_find_sat); self._after_find_sat = None
        self._cancel_after(self._after_find_gc); self._after_find_gc = None
        self._cancel_after(self._after_move_to_tee); self._after_move_to_tee = None
        self._cancel_after(self._after_find_hole); self._after_find_hole = None

        # [추가] next hole 모니터링 취소
        self._stop_next_hole_monitor()

        # GPS sim stop
        try:
            if self.gps_sim is not None and self._gps_started:
                self.gps_sim.stop()
        except Exception:
            pass
        self._gps_started = False

    def _cancel_after(self, after_id: Optional[str]):
        if after_id is None:
            return
        try:
            self.after_cancel(after_id)
        except Exception:
            pass

    # ---------------- BACK ---------------- #

    def _draw_back_hitbox(self):
        self.canvas.create_text(60, 40, text="BACK", fill="white", font=("Helvetica", 14, "bold"))
        back_region = self.canvas.create_rectangle(0, 0, 120, 80, outline="", fill="")
        self.canvas.tag_bind(back_region, "<Button-1>", self._on_back)

    def _on_back(self, event=None):
        # playgolf 화면 종료
        self.stop()
        if callable(self.on_back):
            self.on_back()

    # ---------------- GPS 업데이트 ---------------- #

    def on_gps_update(self, t: float, lat: float, lng: float, alt: float):
        self.latitude = lat
        self.longitude = lng
        self.altitude = alt

        print(
            f"[PlayGolf] GPS(t={t:.0f}s): "
            f"lat={self.latitude:.6f}, lng={self.longitude:.6f}, alt={self.altitude}"
        )

    # ---------------- 이미지 ---------------- #

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

    def draw_background(self):
        self.canvas.delete("all")
        try:
            bg_img = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
            self.canvas.create_image(0, 0, anchor="nw", image=bg_img)
        except FileNotFoundError:
            pass

    # ---------------- UI 단계 ---------------- #

    def show_find_sat(self):
        """
        처음 5초간 '골프장 찾는 중 ...' + Find_Sat 표시
        """
        if not self._running:
            return

        self.draw_background()
        self._draw_back_hitbox()

        self.find_sat_text_id = self.canvas.create_text(
            LCD_WIDTH // 2 + 20, LCD_HEIGHT // 2 - 100,
            text="골프장 찾는 중 ...",
            fill="#2BBF3F",
            font=("Helvetica", 18, "bold"),
        )

        try:
            find_img = self.load_image("Find_Sat.png")
            self.find_sat_image_id = self.canvas.create_image(
                LCD_WIDTH // 2, LCD_HEIGHT // 2, image=find_img
            )
        except FileNotFoundError:
            self.find_sat_image_id = None

    def on_find_sat_finished(self):
        """
        Find_Sat UI 제거 후 GC 탐색 루프 시작.
        """
        if not self._running:
            return

        if self.find_sat_text_id is not None:
            self.canvas.delete(self.find_sat_text_id)
            self.find_sat_text_id = None

        if self.find_sat_image_id is not None:
            self.canvas.delete(self.find_sat_image_id)
            self.find_sat_image_id = None

        self.start_find_gc_loop()

    def start_find_gc_loop(self):
        if not self._running:
            return

        gc_name = self.find_GC()
        if gc_name is not None:
            self.draw_background()
            self._draw_back_hitbox()

            self.gc_text_id = self.canvas.create_text(
                LCD_WIDTH // 2, LCD_HEIGHT // 2,
                text=gc_name,
                fill="#2BBF3F",
                font=("Helvetica", 20, "bold"),
            )
            self._cancel_after(self._after_move_to_tee)
            self._after_move_to_tee = self.after(3000, self.show_move_to_tee_box)
        else:
            self._cancel_after(self._after_find_gc)
            self._after_find_gc = self.after(5000, self.start_find_gc_loop)

    def show_move_to_tee_box(self):
        if not self._running:
            return

        if self.gc_text_id is not None:
            self.canvas.delete(self.gc_text_id)
            self.gc_text_id = None

        self.draw_background()
        self._draw_back_hitbox()

        self.canvas.create_text(
            LCD_WIDTH // 2, LCD_HEIGHT // 2 - 10,
            text="티박스로",
            fill="#2BBF3F",
            font=("Helvetica", 22, "bold"),
        )
        self.canvas.create_text(
            LCD_WIDTH // 2, LCD_HEIGHT // 2 + 30,
            text="이동해 주세요",
            fill="#2BBF3F",
            font=("Helvetica", 22, "bold"),
        )

        self._cancel_after(self._after_find_hole)
        self._after_find_hole = self.after(1000, self.start_find_1st_hole_loop)

    # ---------------- GC 찾기 ---------------- #

    def find_GC(self) -> Optional[str]:
        if not self.name_db_path.exists():
            return None

        name_df = pd.read_csv(self.name_db_path)
        required_cols = ["GC_name_code", "GC_name", "center_lat", "center_lng"]
        for c in required_cols:
            if c not in name_df.columns:
                return None

        name_df["center_lat"] = pd.to_numeric(name_df["center_lat"], errors="coerce")
        name_df["center_lng"] = pd.to_numeric(name_df["center_lng"], errors="coerce")

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

        if not candidates:
            return None

        db_df = None
        if len(candidates) == 1:
            chosen = candidates[0]
            chosen_code = chosen["GC_name_code"]
            chosen_name = chosen["GC_name"]
            chosen_clat = chosen["center_lat"]
            chosen_clng = chosen["center_lng"]
        else:
            if not self.db_path.exists():
                return None

            db_df = pd.read_csv(self.db_path)
            best = None
            best_min = float("inf")

            for cand in candidates:
                tf = make_transformer(cand["center_lat"], cand["center_lng"])
                if tf is None:
                    continue

                E_cur, N_cur = tf.transform(self.longitude, self.latitude)

                sub = db_df[db_df["GC_name_code"] == cand["GC_name_code"]]
                if sub.empty:
                    continue

                min_t1 = float("inf")
                for _, r in sub.iterrows():
                    t1_e = r.get("T1_E", np.nan)
                    t1_n = r.get("T1_N", np.nan)
                    if pd.isna(t1_e) or pd.isna(t1_n):
                        continue
                    d = math.hypot(E_cur - t1_e, N_cur - t1_n)
                    min_t1 = min(min_t1, d)

                if min_t1 < best_min:
                    best_min = min_t1
                    best = cand

            if best is None:
                return None

            chosen_code = best["GC_name_code"]
            chosen_name = best["GC_name"]
            chosen_clat = best["center_lat"]
            chosen_clng = best["center_lng"]

        self.current_gc_code = chosen_code
        self.current_gc_name = chosen_name
        self.gc_center_lat = chosen_clat
        self.gc_center_lng = chosen_clng

        if db_df is None:
            db_df = pd.read_csv(self.db_path)

        self.gc_df = db_df[db_df["GC_name_code"] == self.current_gc_code].copy()
        return self.current_gc_name

    # ---------------- 홀 찾기 ---------------- #

    def start_find_1st_hole_loop(self):
        if not self._running:
            return

        ok = self.find_1st_hole()
        if ok:
            print("[find_1st_hole] 홀 fix 완료, 반복 종료")
            return

        self._cancel_after(self._after_find_hole)
        self._after_find_hole = self.after(1000, self.start_find_1st_hole_loop)

    def find_1st_hole(self) -> bool:
        if self.gc_df is None or self.gc_df.empty:
            return False
        if self.gc_center_lat is None or self.gc_center_lng is None:
            return False

        tf = make_transformer(self.gc_center_lat, self.gc_center_lng)
        if tf is None:
            return False

        E_cur, N_cur = tf.transform(self.longitude, self.latitude)

        def get_point(row, label: str):
            e = row.get(f"{label}_E", np.nan)
            n = row.get(f"{label}_N", np.nan)
            if pd.isna(e) or pd.isna(n):
                return None
            return float(e), float(n)

        def get_green_center(row):
            lg_e = row.get("LG_E", np.nan)
            lg_n = row.get("LG_N", np.nan)
            rg_e = row.get("RG_E", np.nan)
            rg_n = row.get("RG_N", np.nan)

            lg_valid = not (pd.isna(lg_e) or pd.isna(lg_n))
            rg_valid = not (pd.isna(rg_e) or pd.isna(rg_n))

            if lg_valid and rg_valid:
                return (float(lg_e + rg_e) / 2.0, float(lg_n + rg_n) / 2.0)
            if lg_valid:
                return float(lg_e), float(lg_n)
            if rg_valid:
                return float(rg_e), float(rg_n)
            return None

        def build_chain_segments(points_order, row):
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
            segments += build_chain_segments(["T1", "T2", "T3"], row)
            segments += build_chain_segments(["T5", "T6", "IP1", "IP2", "GC"], row)
            if not segments:
                continue

            hole_min = float("inf")
            for (a, b) in segments:
                d = point_segment_distance(E_cur, N_cur, a[0], a[1], b[0], b[1])
                hole_min = min(hole_min, d)

            if hole_min <= threshold_dist and hole_min < best_dist:
                best_dist = hole_min
                best_row = row

        if best_row is None:
            return False

        # alt_offset 계산(기존 유지)
        self.alt_offset = self._compute_alt_offset_for_row(best_row, E_cur, N_cur)

        # [추가] 현재 홀 저장 및 next-hole 모니터링 초기화/시작
        self.current_hole_row = best_row
        self.out_of_green_confirmed = False
        self._init_next_hole_finder_if_needed()
        self._start_next_hole_monitor(normal=True)

        # ---- ScreenManager에 위임 ----
        if not callable(self.on_open_distance):
            print("[find_1st_hole] on_open_distance 콜백이 없습니다.")
            return False

        ctx = dict(
            parent_window=self,
            hole_row=best_row,
            gc_center_lat=self.gc_center_lat,
            gc_center_lng=self.gc_center_lng,
            cur_lat=self.latitude,
            cur_lng=self.longitude,
        )
        self.on_open_distance(ctx)
        return True

    # ---------------- next hole (public trigger, compatibility) ---------------- #

    def notify_out_of_green_confirmed(self):
        """
        putt_distance / scoring 흐름에서 호출:
        out-of-green 확정 상태로 전환하여 next hole 탐색이 가능해지도록 한다.
        (실제 탐색 활성화는 finder가 '그린 폴리곤 최소거리 >= 30m' 조건으로 제어)
        """
        self.out_of_green_confirmed = True
        # 후보 감지/확정을 빨리 하기 위해 즉시 한 번 tick 시도
        self._next_hole_tick(force=True)

    def find_next_hole(self) -> bool:
        """
        (기존 코드 호환용)
        entry_menu에서 find_next_hole()을 호출하던 흐름이 있을 수 있으므로 유지한다.

        동작:
        - out_of_green_confirmed를 True로 만들고,
        - next hole 판단 tick을 즉시 1회 수행,
        - 이후 루프는 계속 유지(확정 시 자동으로 Distance로 전환).
        반환:
        - 실행 가능하면 True, 불가능(데이터/상태 없음)하면 False
        """
        if self.gc_df is None or self.gc_df.empty:
            return False
        if self.gc_center_lat is None or self.gc_center_lng is None:
            return False
        if self.current_hole_row is None:
            return False

        self.notify_out_of_green_confirmed()
        return True

    # ---------------- next hole (internal) ---------------- #

    def _init_next_hole_finder_if_needed(self):
        if self._next_finder is not None:
            return
        if self.gc_df is None or self.gc_df.empty:
            return
        if self.gc_center_lat is None or self.gc_center_lng is None:
            return

        # 설정값은 논의된 기본을 그대로 사용(tee 35m, 누적 60s/윈도 90s, green exit 30m)
        cfg = NextHoleConfig(
            tee_detect_threshold_m=35.0,
            confirm_cum_seconds=60.0,
            window_seconds=90.0,
            min_green_exit_distance_m=30.0,
            switch_hysteresis_m=3.0,
        )
        self._next_finder = NextHoleFinder(self.gc_df, self.gc_center_lat, self.gc_center_lng, config=cfg)

    def _start_next_hole_monitor(self, normal: bool):
        """
        평상시(60초) / 후보 감지(5초) 스케줄링 시작.
        """
        if not self._running:
            return
        self._init_next_hole_finder_if_needed()
        if self._next_finder is None:
            return

        self._next_interval_ms = self._next_interval_ms_normal if normal else self._next_interval_ms_candidate

        self._cancel_after(self._after_next_hole)
        self._after_next_hole = self.after(self._next_interval_ms, self._next_hole_tick)

    def _stop_next_hole_monitor(self):
        self._cancel_after(self._after_next_hole)
        self._after_next_hole = None
        self._next_interval_ms = self._next_interval_ms_normal

    def _get_selected_green_polygon_EN(self) -> List[Tuple[float, float]]:
        """
        현재 홀(current_hole_row)과 last_green(L/R)을 기준으로 그린 폴리곤(ENU) 반환.
        - 스무딩 전 원본 포인트(LG1~LG20 / RG1~RG20)를 사용한다.
        """
        if self.current_hole_row is None:
            return []
        row = self.current_hole_row
        prefix = "LG" if (self.last_green != "R") else "RG"

        pts: List[Tuple[float, float]] = []
        for k in range(1, 21):
            e = row.get(f"{prefix}{k}_E", np.nan)
            n = row.get(f"{prefix}{k}_N", np.nan)
            if pd.isna(e) or pd.isna(n):
                continue
            pts.append((float(e), float(n)))
        return pts

    def _next_hole_tick(self, force: bool = False):
        """
        NextHoleFinder.update() 호출 및 확정 처리.
        - force=True면 after 스케줄링과 무관하게 즉시 1회 판단
        """
        if not self._running:
            return

        self._init_next_hole_finder_if_needed()
        if self._next_finder is None:
            # finder 준비 안되면 다음 tick은 normal로 유지
            self._start_next_hole_monitor(normal=True)
            return

        if self.current_hole_row is None:
            self._start_next_hole_monitor(normal=True)
            return

        green_poly = self._get_selected_green_polygon_EN()

        confirmed, next_row, dbg = self._next_finder.update(
            current_hole_row=self.current_hole_row,
            cur_lat=self.latitude,
            cur_lng=self.longitude,
            out_of_green_confirmed=self.out_of_green_confirmed,
            green_polygon_EN=green_poly,
            now_ts=time.time(),
        )

        # 후보 감지 시 고주기 전환 판단
        # - finder가 search_enabled이고 is_in=True이며 candidate_dist가 threshold 이하이면 후보 감지로 판단
        search_enabled = bool(dbg.get("search_enabled", False))
        is_in = bool(dbg.get("is_in", False))
        cand_dist = dbg.get("candidate_dist", None)
        cand_close = (cand_dist is not None) and (float(cand_dist) <= self._next_finder.cfg.tee_detect_threshold_m)

        if confirmed and (next_row is not None):
            # 홀 확정: current hole 갱신 + alt_offset 재계산 + 상태 리셋 + measure_distance 재실행
            self.current_hole_row = next_row

            # EN 좌표 계산(alt_offset 재계산용)
            tf = make_transformer(self.gc_center_lat, self.gc_center_lng) if (self.gc_center_lat is not None and self.gc_center_lng is not None) else None
            if tf is not None:
                E_cur, N_cur = tf.transform(self.longitude, self.latitude)
                self.alt_offset = self._compute_alt_offset_for_row(next_row, E_cur, N_cur)
            else:
                self.alt_offset = 0.0

            # 새 홀 시작이므로 out_of_green 플래그 리셋
            self.out_of_green_confirmed = False

            # finder 상태 리셋(권장)
            try:
                self._next_finder.reset()
            except Exception:
                pass

            # measure_distance 재실행(콜백 위임)
            if callable(self.on_open_distance):
                ctx = dict(
                    parent_window=self,
                    hole_row=next_row,
                    gc_center_lat=self.gc_center_lat,
                    gc_center_lng=self.gc_center_lng,
                    cur_lat=self.latitude,
                    cur_lng=self.longitude,
                )
                self.on_open_distance(ctx)

            # 다음 홀 모니터링은 계속(평상시 60초) 유지
            self._start_next_hole_monitor(normal=True)
            return

        # 확정이 아니면 주기 조정
        if search_enabled and is_in and cand_close:
            # 후보 감지 -> 고주기
            self._start_next_hole_monitor(normal=False)
        else:
            # 평상시
            self._start_next_hole_monitor(normal=True)

        # force 호출이면 after는 이미 위에서 재예약되므로 그대로 종료
        _ = force

    # ---------------- alt_offset util (기존 find_1st_hole 유지 로직) ---------------- #

    def _compute_alt_offset_for_row(self, hole_row: pd.Series, E_cur: float, N_cur: float) -> float:
        """
        기존 find_1st_hole의 alt_offset 계산 로직을 동일하게 재사용:
        - fix_labels(T1~T6, IP1, IP2) 중 현재 위치에서 가장 가까운 점의 dAlt 사용
        - alt_offset = (현재고도 - DB고도)
        """
        def get_point(row, label: str):
            e = row.get(f"{label}_E", np.nan)
            n = row.get(f"{label}_N", np.nan)
            if pd.isna(e) or pd.isna(n):
                return None
            return float(e), float(n)

        def get_label_alt(row, label: str) -> Optional[float]:
            a = row.get(f"{label}_dAlt", np.nan)
            if pd.isna(a):
                return None
            return float(a)

        fix_labels = ["T1", "T2", "T3", "T4", "T5", "T6", "IP1", "IP2"]
        best_fix_dist = float("inf")
        best_fix_alt_db: Optional[float] = None

        for lab in fix_labels:
            p = get_point(hole_row, lab)
            if p is None:
                continue
            alt_db = get_label_alt(hole_row, lab)
            if alt_db is None:
                continue
            d = math.hypot(E_cur - p[0], N_cur - p[1])
            if d < best_fix_dist:
                best_fix_dist = d
                best_fix_alt_db = alt_db

        return (self.altitude - best_fix_alt_db) if best_fix_alt_db is not None else 0.0


def open_play_golf_window(
    parent: tk.Misc,
    on_back: Optional[Callable[[], None]] = None,
    on_open_distance: Optional[Callable[[dict], None]] = None,
) -> PlayGolfFrame:
    return PlayGolfFrame(parent, on_back=on_back, on_open_distance=on_open_distance)
