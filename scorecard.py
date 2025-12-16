# scorecard.py
#
# - ScreenManager 전환용 Scorecard 화면(Frame)
# - 배경: background.png (다른 화면과 동일)
# - 상단: GC_name
# - 상단(초록): total_score 합 + (par_score 합)
# - 테이블: 홀 / 파 / 스코어 / 퍼트
# - 표시: 5행 고정, 터치 드래그(상/하)로 스크롤
# - 데이터: scoring.py의 OK 결과를 홀별로 저장한 dict를 읽어 표시
#
# 요구 데이터(권장):
#   parent_window.round_scores: Dict[int, dict]
#     dict 예: {"Hole":1,"PAR":4,"par_score":1,"score":5,"putt_score":2}

import tkinter as tk
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple

from PIL import Image, ImageTk

from config import LCD_WIDTH, LCD_HEIGHT


class ScorecardScreen(tk.Frame):
    ROWS_PER_PAGE = 5
    DRAG_STEP_PX = 50  # 드래그 누적이 이 값을 넘으면 1행 스크롤

    def __init__(
        self,
        master: tk.Misc,
        on_back: Optional[Callable[[], None]] = None,
    ):
        super().__init__(master, width=LCD_WIDTH, height=LCD_HEIGHT, bg="black")
        self.pack_propagate(False)

        self.on_back = on_back

        self.base_dir = Path(__file__).parent
        self.asset_dir = self.base_dir / "assets_image"
        self.images: Dict[Tuple[str, Optional[Tuple[int, int]]], ImageTk.PhotoImage] = {}

        self.canvas = tk.Canvas(self, width=LCD_WIDTH, height=LCD_HEIGHT, highlightthickness=0, bg="black")
        self.canvas.pack()

        # context
        self.parent_window = None  # PlayGolfFrame 등
        self._rows: List[Dict[str, Any]] = []
        self._start_index: int = 0

        # drag state
        self._drag_active = False
        self._drag_last_y: float = 0.0
        self._drag_accum_dy: float = 0.0

        # bindings
        self._draw_back_hitbox()
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        # Windows 마우스 휠
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)

    # ---------------- public ---------------- #

    def set_context(self, parent_window=None):
        """
        parent_window(예: PlayGolfFrame)에서 score 데이터/GC_name을 읽는다.
        """
        self.parent_window = parent_window
        self._refresh_rows_from_parent()
        self._start_index = 0

    def start(self):
        self._refresh_rows_from_parent()
        self._clamp_start_index()
        self._render()

    def stop(self):
        # 별도 after 루프 없음
        return

    # ---------------- data ---------------- #

    def _get_scores_dict(self) -> Dict[int, Dict[str, Any]]:
        """
        parent_window에 저장된 홀별 스코어 dict를 탐색하여 반환.
        기대 형태:
          { hole_no(int): {"Hole":..., "PAR":..., "par_score":..., "score":..., "putt_score":...}, ... }
        """
        if self.parent_window is None:
            return {}

        for attr in ("round_scores", "scorecard_scores", "scores_by_hole"):
            d = getattr(self.parent_window, attr, None)
            if isinstance(d, dict):
                # 키가 int가 아닐 수 있어도 정렬은 가능하도록 변환 시도는 렌더에서 처리
                return d
        return {}

    def _get_gc_name(self) -> str:
        if self.parent_window is None:
            return ""
        return str(getattr(self.parent_window, "current_gc_name", "") or "")

    def _refresh_rows_from_parent(self):
        scores = self._get_scores_dict()

        rows: List[Dict[str, Any]] = []
        for k, v in scores.items():
            try:
                hole_no = int(v.get("Hole", k))
            except Exception:
                continue

            try:
                par = int(v.get("PAR", 0))
            except Exception:
                par = 0

            try:
                par_score = int(v.get("par_score", 0))
            except Exception:
                par_score = 0

            try:
                total_score = int(v.get("score", par + par_score))
            except Exception:
                total_score = par + par_score

            try:
                putt = int(v.get("putt_score", 0))
            except Exception:
                putt = 0

            rows.append({
                "Hole": hole_no,
                "PAR": par,
                "par_score": par_score,
                "score": total_score,
                "putt_score": putt,
            })

        rows.sort(key=lambda r: r["Hole"])
        self._rows = rows

    def _clamp_start_index(self):
        max_start = max(0, len(self._rows) - self.ROWS_PER_PAGE)
        if self._start_index < 0:
            self._start_index = 0
        elif self._start_index > max_start:
            self._start_index = max_start

    # ---------------- input ---------------- #

    def _draw_back_hitbox(self):
        self.canvas.create_text(60, 40, text="BACK", fill="white", font=("Helvetica", 14, "bold"))
        back_region = self.canvas.create_rectangle(0, 0, 120, 80, outline="", fill="")
        self.canvas.tag_bind(back_region, "<Button-1>", lambda e: self._on_back())

    def _on_back(self):
        self.stop()
        if callable(self.on_back):
            self.on_back()

    def _on_press(self, event):
        self._drag_active = True
        self._drag_last_y = float(event.y)
        self._drag_accum_dy = 0.0

    def _on_drag(self, event):
        if not self._drag_active:
            return

        y = float(event.y)
        dy = y - self._drag_last_y
        self._drag_last_y = y
        self._drag_accum_dy += dy

        # 아래로 드래그(+) = 위쪽(이전 행)으로 스크롤
        # 위로 드래그(-) = 아래쪽(다음 행)으로 스크롤
        while self._drag_accum_dy >= self.DRAG_STEP_PX:
            self._drag_accum_dy -= self.DRAG_STEP_PX
            self._scroll_by(-1)

        while self._drag_accum_dy <= -self.DRAG_STEP_PX:
            self._drag_accum_dy += self.DRAG_STEP_PX
            self._scroll_by(+1)

    def _on_release(self, event):
        self._drag_active = False

    def _on_mousewheel(self, event):
        # Windows: event.delta > 0 위로 스크롤(이전), <0 아래로 스크롤(다음)
        if event.delta > 0:
            self._scroll_by(-1)
        elif event.delta < 0:
            self._scroll_by(+1)

    def _scroll_by(self, delta_rows: int):
        if not self._rows:
            return
        self._start_index += int(delta_rows)
        self._clamp_start_index()
        self._render()

    # ---------------- rendering ---------------- #

    def load_image(self, filename: str, size: Optional[Tuple[int, int]] = None) -> ImageTk.PhotoImage:
        key = (filename, size)
        if key in self.images:
            return self.images[key]

        path = self.asset_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"[Scorecard] 이미지 파일을 찾을 수 없습니다: {path}")

        img = Image.open(path).convert("RGBA")
        if size is not None:
            img = img.resize(size, Image.LANCZOS)

        photo = ImageTk.PhotoImage(img)
        self.images[key] = photo
        return photo

    def _render(self):
        self.canvas.delete("all")

        # background
        try:
            bg = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
            self.canvas.create_image(0, 0, anchor="nw", image=bg)
        except FileNotFoundError:
            pass

        self._draw_back_hitbox()

        gc_name = self._get_gc_name()

        total_sum = sum(r["score"] for r in self._rows)
        par_score_sum = sum(r["par_score"] for r in self._rows)

        cx = LCD_WIDTH // 2

        # GC_name
        self.canvas.create_text(
            cx, 85,
            text=gc_name,
            fill="white",
            font=("Helvetica", 26, "bold"),
        )

        # total (+par_score)
        ps = f"{par_score_sum:+d}"
        self.canvas.create_text(
            cx, 125,
            text=f"{total_sum} ({ps})",
            fill="#2BBF3F",
            font=("Helvetica", 22, "bold"),
        )

        # header bar
        bar_w = 300
        bar_h = 40
        bar_x0 = cx - bar_w // 2
        bar_y0 = 160
        bar_x1 = cx + bar_w // 2
        bar_y1 = bar_y0 + bar_h
        self.canvas.create_rectangle(bar_x0, bar_y0, bar_x1, bar_y1, fill="#404040", outline="#404040")

        # header text
        cols = ["홀", "파", "스코어", "퍼트"]
        col_x = [cx - 105, cx - 35, cx + 45, cx + 120]
        for t, x in zip(cols, col_x):
            self.canvas.create_text(x, bar_y0 + bar_h // 2, text=t, fill="white", font=("Helvetica", 16, "bold"))

        # rows (5 lines)
        start = self._start_index
        end = min(len(self._rows), start + self.ROWS_PER_PAGE)
        visible = self._rows[start:end]

        row_y0 = 215
        row_gap = 40

        for i, r in enumerate(visible):
            y = row_y0 + i * row_gap
            self.canvas.create_text(col_x[0], y, text=str(r["Hole"]), fill="white", font=("Helvetica", 18, "bold"))
            self.canvas.create_text(col_x[1], y, text=str(r["PAR"]), fill="white", font=("Helvetica", 18, "bold"))
            self.canvas.create_text(col_x[2], y, text=str(r["score"]), fill="#2BBF3F", font=("Helvetica", 18, "bold"))
            self.canvas.create_text(col_x[3], y, text=str(r["putt_score"]), fill="white", font=("Helvetica", 18, "bold"))

        # scroll indicator (optional, subtle)
        if len(self._rows) > self.ROWS_PER_PAGE:
            # show "start-end / total"
            self.canvas.create_text(
                cx, LCD_HEIGHT - 35,
                text=f"{start+1}-{end} / {len(self._rows)}",
                fill="white",
                font=("Helvetica", 12, "bold"),
            )
