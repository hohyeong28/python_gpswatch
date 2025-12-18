# end_golf.py
#
# - END 진입 화면
# - 중앙 좌측 stop.png(종료), 우측 pause.png(일시 멈춤)
# - pause 클릭 시 replay.png(다시 시작)로 토글
# - replay 클릭 시 measure_distance로 복귀(on_replay)
# - stop 클릭 시:
#     * app_settings.scorecard == "사용": 오늘의 결과 표시
#     * else: 프로그램 종료(on_exit)
# - 오늘의 결과 OK 클릭 시:
#     * (scorecard 사용) 라운딩 기록 저장 후 프로그램 종료(on_exit)
#     * (scorecard 미사용) 저장 없이 프로그램 종료(on_exit)
# - 배경은 background.png 사용
# - 라운딩 기록 저장 파일: DB/round_history.json

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple, List

import tkinter as tk
from PIL import Image, ImageTk

from config import LCD_WIDTH, LCD_HEIGHT
from setting import app_settings


class EndGolfScreen(tk.Frame):
    STEPS_FIXED = 3844  # 추후 외부데이터로 교체

    def __init__(
        self,
        master: tk.Misc,
        on_replay: Optional[Callable[[], None]] = None,
        on_exit: Optional[Callable[[], None]] = None,
    ):
        super().__init__(master, width=LCD_WIDTH, height=LCD_HEIGHT, bg="black")
        self.pack_propagate(False)

        self.on_replay = on_replay
        self.on_exit = on_exit

        self.base_dir = Path(__file__).parent
        self.asset_dir = self.base_dir / "assets_image"
        self.db_dir = self.base_dir / "DB"
        self.history_path = self.db_dir / "round_history.json"

        self.images: Dict[Tuple[str, Optional[Tuple[int, int]]], ImageTk.PhotoImage] = {}

        self.canvas = tk.Canvas(self, width=LCD_WIDTH, height=LCD_HEIGHT, highlightthickness=0, bg="black")
        self.canvas.pack()

        self.parent_window = None

        self._mode = "select"   # select | summary
        self._paused = False
        self._after_id = None

    def set_context(self, parent_window=None):
        self.parent_window = parent_window
        self._mode = "select"
        self._paused = False

    def start(self):
        self.stop()
        self._render()

    def stop(self):
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def load_image(self, filename: str, size=None) -> ImageTk.PhotoImage:
        key = (filename, size)
        if key in self.images:
            return self.images[key]

        path = self.asset_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"[EndGolf] 이미지 파일을 찾을 수 없습니다: {path}")

        img = Image.open(path).convert("RGBA")
        if size is not None:
            img = img.resize(size, Image.LANCZOS)

        photo = ImageTk.PhotoImage(img)
        self.images[key] = photo
        return photo

    # ---------------- actions ---------------- #

    def _on_pause_or_replay(self):
        if not self._paused:
            self._paused = True
            self._render()
            return

        # replay
        if callable(self.on_replay):
            self.on_replay()

    def _on_stop(self):
        # scorecard 사용안함이면 결과 화면 없이 종료
        if getattr(app_settings, "scorecard", "사용안함") != "사용":
            if callable(self.on_exit):
                self.on_exit()
            return

        self._mode = "summary"
        self._render()

    def _on_ok(self, event=None):
        # OK = 라운드 종료 확정 + (scorecard 사용 시) 라운딩 기록 저장 + 앱 종료
        if getattr(app_settings, "scorecard", "사용안함") == "사용":
            try:
                self._save_round_history()
            except Exception as e:
                print("[EndGolf] round history save error:", e)

        if callable(self.on_exit):
            self.on_exit()

    # ---------------- round history save ---------------- #

    def _ensure_db_dir(self):
        try:
            self.db_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def _load_history_list(self) -> List[dict]:
        if not self.history_path.exists():
            return []
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _write_history_list(self, data: List[dict]):
        self._ensure_db_dir()
        tmp = self.history_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(self.history_path)

    def _save_round_history(self):
        summary = self._compute_summary()

        gc_name = ""
        if self.parent_window is not None:
            gc_name = str(getattr(self.parent_window, "current_gc_name", "") or "")

        # 날짜: 종료 시점 기준
        date_str = datetime.now().strftime("%Y-%m-%d")

        record = dict(
            date=date_str,
            gc_name=gc_name,
            total_score=int(summary["score_sum"]),
            total_par_score=int(summary["par_score_sum"]),
            total_putt=int(summary["putt_sum"]),
            gir_pct=float(summary["gir_pct"]),
            duration_sec=int(summary["duration_sec"]),
            steps=int(self.STEPS_FIXED),
        )

        hist = self._load_history_list()
        hist.append(record)
        self._write_history_list(hist)

    # ---------------- summary metrics ---------------- #

    def _get_round_scores(self) -> Dict[int, Dict[str, Any]]:
        if self.parent_window is None:
            return {}
        d = getattr(self.parent_window, "round_scores", None)
        return d if isinstance(d, dict) else {}

    def _format_duration(self, seconds: float) -> str:
        if seconds < 0:
            seconds = 0
        minutes = int(seconds // 60)
        h = minutes // 60
        m = minutes % 60
        return f"{h}시간{m}분" if h > 0 else f"{m}분"

    def _compute_summary(self) -> Dict[str, Any]:
        scores = self._get_round_scores()
        rows = []
        for _, v in scores.items():
            try:
                hole = int(v.get("Hole", 0))
                par = int(v.get("PAR", 0))
                score = int(v.get("score", 0))
                putt = int(v.get("putt_score", 0))
                par_score = int(v.get("par_score", 0))
            except Exception:
                continue
            if hole <= 0:
                continue
            rows.append(dict(hole=hole, par=par, score=score, putt=putt, par_score=par_score))
        rows.sort(key=lambda r: r["hole"])

        holes_played = len(rows)
        score_sum = sum(r["score"] for r in rows)
        par_score_sum = sum(r["par_score"] for r in rows)

        putt_sum = sum(r["putt"] for r in rows)
        putt_avg = (putt_sum / holes_played) if holes_played > 0 else 0.0

        # GIR: (score - putt) == (PAR - 2)
        gir_cnt = 0
        for r in rows:
            if (r["score"] - r["putt"]) == (r["par"] - 2):
                gir_cnt += 1
        gir_pct = (gir_cnt / holes_played * 100.0) if holes_played > 0 else 0.0

        # 운동시간: now - round_start_time
        start_ts = getattr(self.parent_window, "round_start_time", None) if self.parent_window else None
        if isinstance(start_ts, (int, float)):
            dur = time.time() - float(start_ts)
        else:
            dur = 0.0

        return dict(
            duration_str=self._format_duration(dur),
            duration_sec=int(dur) if dur > 0 else 0,
            holes_played=holes_played,
            score_sum=score_sum,
            par_score_sum=par_score_sum,
            putt_sum=putt_sum,
            putt_avg=putt_avg,
            gir_pct=gir_pct,
        )

    # ---------------- render ---------------- #

    def _render(self):
        self.canvas.delete("all")

        # background
        try:
            bg = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
            self.canvas.create_image(0, 0, anchor="nw", image=bg)
        except Exception:
            pass

        if self._mode == "select":
            self._render_select()
        else:
            self._render_summary()

    def _render_select(self):
        center_y = LCD_HEIGHT // 2
        left_x = LCD_WIDTH // 2 - 80
        right_x = LCD_WIDTH // 2 + 80

        # stop
        try:
            stop_img = self.load_image("stop.png")
            stop_id = self.canvas.create_image(left_x, center_y, image=stop_img)
            self.canvas.tag_bind(stop_id, "<Button-1>", lambda e: self._on_stop())
        except Exception:
            stop_id = self.canvas.create_text(left_x, center_y, text="STOP", fill="white", font=("Helvetica", 24, "bold"))
            self.canvas.tag_bind(stop_id, "<Button-1>", lambda e: self._on_stop())

        stop_txt = self.canvas.create_text(left_x, center_y + 80, text="종료", fill="white", font=("Helvetica", 18, "bold"))
        self.canvas.tag_bind(stop_txt, "<Button-1>", lambda e: self._on_stop())

        # pause/replay
        if not self._paused:
            img_name = "pause.png"
            label = "일시 멈춤"
        else:
            img_name = "replay.png"
            label = "다시 시작"

        try:
            pr_img = self.load_image(img_name)
            pr_id = self.canvas.create_image(right_x, center_y, image=pr_img)
            self.canvas.tag_bind(pr_id, "<Button-1>", lambda e: self._on_pause_or_replay())
        except Exception:
            pr_id = self.canvas.create_text(
                right_x, center_y,
                text=("PAUSE" if not self._paused else "REPLAY"),
                fill="white",
                font=("Helvetica", 24, "bold"),
            )
            self.canvas.tag_bind(pr_id, "<Button-1>", lambda e: self._on_pause_or_replay())

        pr_txt = self.canvas.create_text(right_x, center_y + 80, text=label, fill="white", font=("Helvetica", 18, "bold"))
        self.canvas.tag_bind(pr_txt, "<Button-1>", lambda e: self._on_pause_or_replay())

    def _render_summary(self):
        s = self._compute_summary()

        cx = LCD_WIDTH // 2
        cy = LCD_HEIGHT // 2
        outer_r = 205
        inner_r = 150

        self.canvas.create_oval(cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r, outline="white", width=8)
        self.canvas.create_oval(cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r, outline="", fill="black")

        self.canvas.create_text(cx, cy - 120, text="오늘의 결과", fill="#2BBF3F", font=("Helvetica", 22, "bold"))

        left_label_x = cx - 25
        right_val_x = cx + 20
        y0 = cy - 70
        gap = 40

        def row(y, label, value):
            self.canvas.create_text(left_label_x, y, text=label, fill="white", font=("Helvetica", 18, "bold"), anchor="e")
            self.canvas.create_text(right_val_x, y, text=value, fill="white", font=("Helvetica", 18, "bold"), anchor="w")

        row(y0 + gap * 0, "운동시간", s["duration_str"])
        row(y0 + gap * 1, "이동걸음수", f"{self.STEPS_FIXED:,}")
        ps = f"{int(s['par_score_sum']):+d}"
        row(y0 + gap * 2, "스코어", f"{int(s['score_sum'])} ({ps})")
        row(y0 + gap * 3, "퍼팅수", f"{int(s['putt_sum'])} ({s['putt_avg']:.1f} / 홀)")
        row(y0 + gap * 4, "GIR", f"{s['gir_pct']:.0f} %")

        # OK 버튼 (confirm.png + OK 텍스트)
        try:
            ok_img = self.load_image("confirm.png")
            ok_img_id = self.canvas.create_image(LCD_WIDTH // 2, LCD_HEIGHT - 70, image=ok_img)
            ok_text_id = self.canvas.create_text(
                LCD_WIDTH // 2,
                LCD_HEIGHT - 70,
                text="OK",
                fill="white",
                font=("Helvetica", 20, "bold"),
            )
            self.canvas.tag_bind(ok_img_id, "<Button-1>", self._on_ok)
            self.canvas.tag_bind(ok_text_id, "<Button-1>", self._on_ok)
        except FileNotFoundError:
            ok_rect = self.canvas.create_oval(
                LCD_WIDTH // 2 - 60,
                LCD_HEIGHT - 80,
                LCD_WIDTH // 2 + 60,
                LCD_HEIGHT - 40,
                fill="green",
                outline="",
            )
            self.canvas.tag_bind(ok_rect, "<Button-1>", self._on_ok)
            self.canvas.create_text(
                LCD_WIDTH // 2,
                LCD_HEIGHT - 60,
                text="OK",
                fill="white",
                font=("Helvetica", 18, "bold"),
            )
