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
