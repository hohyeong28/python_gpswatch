# find_next_hole.py
#
# [요구사항 반영]
# 1) find_next_hole은 find_1st_hole과 유사한 개념(거리 기반 후보 탐색)이나,
#    다음 홀은 "티 영역" 판별이 가장 안정적이므로 T1~T6만 사용한다.
# 2) 티 클러스터 방식 적용:
#    - 각 홀의 T1~T6 포인트 집합을 티 클러스터로 보고,
#      현재 위치에서의 티 접근 거리는 min(dist(cur, Ti))로 정의한다.
# 3) 후보 감지 시 누적 60초(슬라이딩 윈도우) 조건으로 확정한다.
# 4) On-green 이후 out-of-green 확정 후에도 바로 탐색하지 않고,
#    그린 폴리곤과의 최소 거리가 30m 이상일 때만 탐색/확정을 진행한다.
#
# [사용 방법 요약]
# - PlayGolfFrame 등에서 60초 주기(평상시)로 update(...) 호출.
# - update(...)가 candidate를 감지하면, 호출 주기를 5~10초로 높여 재확인.
# - update(...)가 (confirmed=True, next_row=<pd.Series>)를 반환하면
#   current hole을 변경하고 measure_distance로 전환한다.

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


# --------------------------
# Geometry helpers
# --------------------------

def make_transformer(lat0: float, lon0: float) -> Optional[Transformer]:
    if np.isnan(lat0) or np.isnan(lon0):
        return None
    wgs84 = CRS.from_epsg(4326)
    aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m"
    )
    return Transformer.from_crs(wgs84, aeqd, always_xy=True)  # (lon, lat) -> (E, N)


def euclid(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    # Distance from point P to segment AB
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay

    vv = vx * vx + vy * vy
    if vv <= 1e-12:
        return math.hypot(px - ax, py - ay)

    t = (wx * vx + wy * vy) / vv
    if t < 0.0:
        cx, cy = ax, ay
    elif t > 1.0:
        cx, cy = bx, by
    else:
        cx = ax + t * vx
        cy = ay + t * vy
    return math.hypot(px - cx, py - cy)


def point_polygon_min_distance(px: float, py: float, polygon: List[Tuple[float, float]]) -> float:
    """
    Polygon is a closed loop (list of vertices). We compute min distance to edges.
    If polygon has <2 points, returns inf. If exactly 2, treated as a segment.
    """
    n = len(polygon)
    if n == 0:
        return float("inf")
    if n == 1:
        return math.hypot(px - polygon[0][0], py - polygon[0][1])
    if n == 2:
        return point_segment_distance(px, py, polygon[0][0], polygon[0][1], polygon[1][0], polygon[1][1])

    best = float("inf")
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        d = point_segment_distance(px, py, ax, ay, bx, by)
        if d < best:
            best = d
    return best


# --------------------------
# Data model
# --------------------------

@dataclass
class NextHoleConfig:
    # 티 클러스터 근접 임계값 (m)
    tee_detect_threshold_m: float = 20.0

    # 후보 확정 누적 시간 (s)
    confirm_cum_seconds: float = 2.0   # 60.0(60초) ==> 시뮬레이션을 위해 2초로 적용

    # 누적 계산 슬라이딩 윈도우 (s) : 90초 창에서 60초 이상 근접이면 확정(권장)
    window_seconds: float = 5.0   # 90.0(90초)  ==> 시뮬레이션을 위해 5초로 적용

    # out-of-green 이후 다음 홀 탐색 시작 조건: 그린 폴리곤 최소거리 >= 30m
    min_green_exit_distance_m: float = 30.0

    # 후보가 바뀔 때 너무 민감하게 바뀌지 않도록 약간의 히스테리시스 (m)
    # new_candidate_dist가 (current_candidate_dist - hysteresis)보다 작을 때만 후보 교체
    switch_hysteresis_m: float = 3.0


@dataclass
class CandidateResult:
    hole_row: pd.Series
    tee_min_dist_m: float


# --------------------------
# NextHoleFinder
# --------------------------

class NextHoleFinder:
    """
    상태를 내부에 유지하면서, update() 호출마다 "다음 홀 확정 여부"를 판단한다.

    입력:
    - gc_df: 전체 홀 데이터프레임(각 행이 1홀)
    - gc_center_lat/lng: 동일 로컬 좌표계(ENU) 변환 기준
    - current_hole_row: 현재 플레이 중인 홀(row)
    - cur_lat/lng: 현재 GPS
    - out_of_green_confirmed: out-of-green 확정 여부(putt_distance 등에서 결정)
    - green_polygon_EN: 현재 홀의 그린 폴리곤(ENU) (LG/RG 선택 반영된 폴리곤)
    - now_ts: time.time() 기준 타임스탬프 (None이면 내부에서 time.time())

    출력:
    - (confirmed, next_row_or_none, debug_dict)
    """

    def __init__(self, gc_df: pd.DataFrame, gc_center_lat: float, gc_center_lng: float, config: Optional[NextHoleConfig] = None):
        self.gc_df = gc_df
        self.gc_center_lat = gc_center_lat
        self.gc_center_lng = gc_center_lng
        self.cfg = config or NextHoleConfig()

        self.tf = make_transformer(self.gc_center_lat, self.gc_center_lng)

        # 상태: 후보 홀 + 누적
        self._candidate_id: Optional[int] = None  # df index 또는 고유 id용 (여기서는 df index 사용)
        self._candidate_row: Optional[pd.Series] = None
        self._candidate_best_dist: float = float("inf")

        # sliding window samples: (ts, is_in, dt)
        # dt는 이전 샘플과의 시간 간격(초)로 누적 계산에 사용
        self._samples: Deque[Tuple[float, bool, float]] = deque(maxlen=500)

        self._last_ts: Optional[float] = None

        # 탐색 활성화 조건(그린 30m 이탈 + out-of-green)
        self._search_enabled: bool = False

    # ---------- row helpers ----------

    @staticmethod
    def _get_EN_point(row: pd.Series, label: str) -> Optional[Tuple[float, float]]:
        e = row.get(f"{label}_E", np.nan)
        n = row.get(f"{label}_N", np.nan)
        if pd.isna(e) or pd.isna(n):
            return None
        return float(e), float(n)

    def _tee_points(self, row: pd.Series) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        for k in range(1, 7):
            p = self._get_EN_point(row, f"T{k}")
            if p is not None:
                pts.append(p)
        return pts

    # ---------- candidate selection ----------

    def _find_best_candidate(self, E_cur: float, N_cur: float, current_hole_row: Optional[pd.Series]) -> Optional[CandidateResult]:
        """
        티 클러스터 거리(min dist to T1..T6)가 가장 작은 홀을 찾는다.
        현재 홀은 제외한다.
        """
        if self.gc_df is None or self.gc_df.empty:
            return None

        best: Optional[CandidateResult] = None
        best_dist = float("inf")

        # 현재 홀 식별을 위해 Hole 번호/인덱스를 우선 사용
        cur_hole_num = None
        cur_df_index = None
        if current_hole_row is not None:
            cur_hole_num = current_hole_row.get("Hole", None)
            # current_hole_row가 df의 row view라면 name이 index일 가능성이 큼
            try:
                cur_df_index = int(current_hole_row.name)  # type: ignore
            except Exception:
                cur_df_index = None

        for idx, row in self.gc_df.iterrows():
            # 현재 홀 제외(가능한 경우)
            if cur_df_index is not None and idx == cur_df_index:
                continue
            if cur_hole_num is not None:
                try:
                    if int(row.get("Hole", -999)) == int(cur_hole_num):
                        continue
                except Exception:
                    pass

            tees = self._tee_points(row)
            if not tees:
                continue

            dmin = float("inf")
            for (e, n) in tees:
                d = math.hypot(E_cur - e, N_cur - n)
                if d < dmin:
                    dmin = d

            if dmin < best_dist:
                best_dist = dmin
                best = CandidateResult(hole_row=row, tee_min_dist_m=dmin)

        return best

    # ---------- cumulation ----------

    def _append_sample(self, ts: float, is_in: bool):
        """
        Add sample with dt computed from last_ts.
        """
        if self._last_ts is None:
            dt = 0.0
        else:
            dt = max(0.0, ts - self._last_ts)
        self._last_ts = ts
        self._samples.append((ts, is_in, dt))

        # prune old beyond window_seconds
        self._prune(ts)

    def _prune(self, now_ts: float):
        window = self.cfg.window_seconds
        # remove from left while too old
        while self._samples and (now_ts - self._samples[0][0]) > window:
            self._samples.popleft()

    def _cum_in_seconds(self) -> float:
        """
        Sum dt where is_in==True within current window.
        Note: dt is assigned to the sample at time ts, representing time since prev sample.
        """
        s = 0.0
        for _ts, is_in, dt in self._samples:
            if is_in:
                s += dt
        return s

    # ---------- public API ----------

    def reset(self):
        """
        외부에서 홀 확정 후 호출하여 상태 리셋(권장).
        """
        self._candidate_id = None
        self._candidate_row = None
        self._candidate_best_dist = float("inf")
        self._samples.clear()
        self._last_ts = None
        self._search_enabled = False

    def update(
        self,
        current_hole_row: Optional[pd.Series],
        cur_lat: float,
        cur_lng: float,
        out_of_green_confirmed: bool,
        green_polygon_EN: Optional[List[Tuple[float, float]]],
        now_ts: Optional[float] = None,
    ) -> Tuple[bool, Optional[pd.Series], Dict]:
        """
        Returns:
          confirmed: bool
          next_row: pd.Series | None
          debug: dict
        """
        ts = time.time() if now_ts is None else float(now_ts)
        dbg: Dict = {
            "search_enabled": self._search_enabled,
            "candidate": None,
            "candidate_dist": None,
            "cum_in": None,
            "reason": None,
        }

        # transformer check
        if self.tf is None:
            dbg["reason"] = "transformer_none"
            return False, None, dbg

        # current EN
        E_cur, N_cur = self.tf.transform(cur_lng, cur_lat)

        # 3) out-of-green 확정 이후 + 그린 폴리곤 최소거리 >= 30m 일 때만 탐색 enable
        if out_of_green_confirmed:
            if green_polygon_EN:
                d_green = point_polygon_min_distance(E_cur, N_cur, green_polygon_EN)
            else:
                d_green = float("inf")  # 폴리곤이 없으면 이 조건을 강제하기 어려우므로 탐색 허용
            dbg["green_min_dist"] = d_green

            if d_green >= self.cfg.min_green_exit_distance_m:
                self._search_enabled = True
            else:
                self._search_enabled = False
        else:
            self._search_enabled = False

        dbg["search_enabled"] = self._search_enabled

        # 탐색 비활성 시: 후보/누적 상태를 느슨하게 리셋 (오탐 방지)
        if not self._search_enabled:
            self._candidate_id = None
            self._candidate_row = None
            self._candidate_best_dist = float("inf")
            self._samples.clear()
            self._last_ts = None
            dbg["reason"] = "search_disabled"
            return False, None, dbg

        # 1) 티 클러스터 기반 후보 탐색 (T1~T6만)
        cand = self._find_best_candidate(E_cur, N_cur, current_hole_row)
        if cand is None:
            dbg["reason"] = "no_candidate_in_db"
            # 샘플은 누적할 필요가 없음
            return False, None, dbg

        # 후보 거리
        cand_dist = cand.tee_min_dist_m

        # 후보의 df index 추정(가능하면 name을 사용)
        try:
            cand_id = int(cand.hole_row.name)  # type: ignore
        except Exception:
            cand_id = None

        dbg["candidate"] = cand_id
        dbg["candidate_dist"] = cand_dist

        # 후보 스위칭 정책 (히스테리시스)
        if self._candidate_id is None:
            # 최초 후보 설정
            self._candidate_id = cand_id
            self._candidate_row = cand.hole_row
            self._candidate_best_dist = cand_dist
            self._samples.clear()
            self._last_ts = None
        else:
            # 동일 후보면 best dist 업데이트
            if cand_id == self._candidate_id:
                if cand_dist < self._candidate_best_dist:
                    self._candidate_best_dist = cand_dist
            else:
                # 다른 후보가 더 확실히 가까워졌을 때만 교체
                if cand_dist < (self._candidate_best_dist - self.cfg.switch_hysteresis_m):
                    self._candidate_id = cand_id
                    self._candidate_row = cand.hole_row
                    self._candidate_best_dist = cand_dist
                    self._samples.clear()
                    self._last_ts = None

        # 현재 후보에 대해 "티 클러스터 근접" 여부 판정
        is_in = (cand_id == self._candidate_id) and (cand_dist <= self.cfg.tee_detect_threshold_m)

        # 누적 샘플 추가
        self._append_sample(ts, is_in)

        cum_in = self._cum_in_seconds()
        dbg["cum_in"] = cum_in
        dbg["is_in"] = is_in
        dbg["candidate_best_dist"] = self._candidate_best_dist

        # 2) 누적 60초 만족 시 확정
        if cum_in >= self.cfg.confirm_cum_seconds:
            if self._candidate_row is not None:
                dbg["reason"] = "confirmed"
                return True, self._candidate_row, dbg

        dbg["reason"] = "not_confirmed"
        return False, None, dbg
