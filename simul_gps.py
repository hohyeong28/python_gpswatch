# simul_gps.py

from pathlib import Path
from typing import Optional, Callable

import pandas as pd
import numpy as np
import tkinter as tk


class GPSSimulator:
    """
    gps_data.xlsx(time, lat, lng, alt)을 읽어서
    1초 간격으로 콜백으로 값을 넘겨주는 시뮬레이터.
    """

    def __init__(
        self,
        excel_path: Path,
        tk_root: tk.Misc,
        on_update: Callable[[float, float, float, float], None],
    ):
        """
        :param excel_path: gps_data.xlsx 경로
        :param tk_root: Tk 또는 Toplevel (after 사용용)
        :param on_update: (time, lat, lng, alt) 를 전달 받을 콜백
        """
        self.excel_path = excel_path
        self.tk_root = tk_root
        self.on_update = on_update

        self.gps_df: Optional[pd.DataFrame] = None
        self.index: int = 0

        self._load_excel()

    def _load_excel(self):
        if not self.excel_path.exists():
            print(f"[GPSSim] 파일 없음: {self.excel_path}")
            return

        try:
            df = pd.read_excel(self.excel_path)
        except Exception as e:
            print(f"[GPSSim] 엑셀 읽기 오류: {e}")
            return

        required_cols = ["time", "lat", "lng", "alt"]
        for col in required_cols:
            if col not in df.columns:
                print(f"[GPSSim] 컬럼 누락: {col}")
                return

        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
        df["alt"] = pd.to_numeric(df["alt"], errors="coerce")
        df["time"] = pd.to_numeric(df["time"], errors="coerce")

        df = df.dropna(subset=["time", "lat", "lng"]).reset_index(drop=True)
        if df.empty:
            print("[GPSSim] 유효한 데이터 없음")
            return

        self.gps_df = df
        self.index = 0
        print(f"[GPSSim] gps_data 로드 완료: {len(self.gps_df)} rows")

    def start(self):
        """1초 간격 시뮬레이션 시작."""
        if self.gps_df is None or self.gps_df.empty:
            print("[GPSSim] 시작 불가 (데이터 없음)")
            return
        self._step()

    def _step(self):
        if self.gps_df is None or self.gps_df.empty:
            return

        if self.index >= len(self.gps_df):
            print("[GPSSim] 시뮬레이션 종료 (마지막 샘플)")
            return

        row = self.gps_df.iloc[self.index]
        t = float(row["time"])
        lat = float(row["lat"])
        lng = float(row["lng"])
        alt = float(row["alt"]) if not np.isnan(row["alt"]) else 0.0

        # 위치 업데이트 콜백 호출
        self.on_update(t, lat, lng, alt)

        self.index += 1
        # 1초 후 다시 호출
        self.tk_root.after(2000, self._step)
