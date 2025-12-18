# round_history.py
#
# - entry_menu에서 "라운딩기록" 선택 시 표시되는 화면
# - 저장 파일: DB/round_history.json
# - 표시: 날짜, 골프장명, 총스코어(파스코어)
# - 최근 5개 표시, 많으면 드래그 스크롤
# - OK 버튼(confirm.png) 클릭 시 entry_menu로 복귀(on_ok)

import json
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple, List

import tkinter as tk
from PIL import Image, ImageTk

from config import LCD_WIDTH, LCD_HEIGHT


class RoundHistoryScreen(tk.Frame):
    ROWS_PER_PAGE = 5
    DRAG_STEP_PX = 50

    def __init__(
        self,
        master: tk.Misc,
        on_ok: Optional[Callable[[], None]] = None,
    ):
        super().__init__(master, width=LCD_WIDTH, height=LCD_HEIGHT, bg="black")
        self.pack_propagate(False)

        self.on_ok = on_ok

        self.base_dir = Path(__file__).parent
        self.asset_dir = self.base_dir / "assets_image"
        self.db_dir = self.base_dir / "DB"
        self.history_path = self.db_dir / "round_history.json"

        self.images: Dict[Tuple[str, Optional[Tuple[int, int]]], ImageTk.PhotoImage] = {}

        self.canvas = tk.Canvas(self, width=LCD_WIDTH, height=LCD_HEIGHT, highlightthickness=0, bg="black")
        self.canvas.pack()

        self._rows: List[Dict[str, Any]] = []
        self._start_index = 0

        # drag state
        self._drag_active = False
        self._drag_last_y = 0.0
        self._drag_accum_dy = 0.0

        # bindings
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)  # Windows

    def start(self):
        self._load_rows()
        self._start_index = 0
        self._clamp_start_index()
        self._render()

    def stop(self):
        return

    # ---------------- io ---------------- #

    def _load_rows(self):
        self._rows = []
        if not self.history_path.exists():
            return
        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return
        except Exception:
            return

        # 최근 기록이 마지막에 append되는 구조이므로, 최신순으로 역순 정렬
        rows: List[Dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            date = str(item.get("date", "") or "")
            gc_name = str(item.get("gc_name", "") or "")
            total_score = item.get("total_score", 0)
            total_par_score = item.get("total_par_score", 0)
            try:
                total_score = int(total_score)
            except Exception:
                total_score = 0
            try:
                total_par_score = int(total_par_score)
            except Exception:
                total_par_score = 0

            rows.append(dict(
                date=date,
                gc_name=gc_name,
                score=total_score,
                par_score=total_par_score,
            ))

        rows = list(reversed(rows))
        self._rows = rows

    # ---------------- ui helpers ---------------- #

    def load_image(self, filename: str, size: Optional[Tuple[int, int]] = None) -> ImageTk.PhotoImage:
        key = (filename, size)
        if key in self.images:
            return self.images[key]

        path = self.asset_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"[RoundHistory] 이미지 파일을 찾을 수 없습니다: {path}")

        img = Image.open(path).convert("RGBA")
        if size is not None:
            img = img.resize(size, Image.LANCZOS)

        photo = ImageTk.PhotoImage(img)
        self.images[key] = photo
        return photo

    def _clamp_start_index(self):
        max_start = max(0, len(self._rows) - self.ROWS_PER_PAGE)
        if self._start_index < 0:
            self._start_index = 0
        elif self._start_index > max_start:
            self._start_index = max_start

    # ---------------- input ---------------- #

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

        while self._drag_accum_dy >= self.DRAG_STEP_PX:
            self._drag_accum_dy -= self.DRAG_STEP_PX
            self._scroll_by(-1)

        while self._drag_accum_dy <= -self.DRAG_STEP_PX:
            self._drag_accum_dy += self.DRAG_STEP_PX
            self._scroll_by(+1)

    def _on_release(self, event):
        self._drag_active = False

    def _on_mousewheel(self, event):
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

    def _on_ok(self, event=None):
        if callable(self.on_ok):
            self.on_ok()

    # ---------------- render ---------------- #

    def _render(self):
        self.canvas.delete("all")

        # background
        try:
            bg = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
            self.canvas.create_image(0, 0, anchor="nw", image=bg)
        except Exception:
            pass

        cx = LCD_WIDTH // 2
        cy = LCD_HEIGHT // 2

        # title (사용자 조정 유지)
        self.canvas.create_text(cx, cy - 120, text="라운드 기록", fill="#2BBF3F", font=("Helvetica", 22, "bold"))

        # header bar (폭/높이 변경 + 위치 조정)
        bar_w = 300
        bar_h = 34
        x0 = cx - bar_w // 2
        y0 = 150
        x1 = cx + bar_w // 2
        y1 = y0 + bar_h
        self.canvas.create_rectangle(x0, y0, x1, y1, fill="#404040", outline="#404040")

        # columns (폭이 줄었으므로 균등 분배)
        col_x_date = x0 + 55
        col_x_gc = x0 + 150
        col_x_score = x1 - 55

        self.canvas.create_text(col_x_date, y0 + bar_h // 2, text="날짜", fill="white", font=("Helvetica", 14, "bold"),
                                anchor="center")
        self.canvas.create_text(col_x_gc, y0 + bar_h // 2, text="골프장", fill="white", font=("Helvetica", 14, "bold"),
                                anchor="center")
        self.canvas.create_text(col_x_score, y0 + bar_h // 2, text="스코어", fill="white", font=("Helvetica", 14, "bold"),
                                anchor="center")

        # rows
        start = self._start_index
        end = min(len(self._rows), start + self.ROWS_PER_PAGE)
        visible = self._rows[start:end]

        row_y0 = y1 + 30
        row_gap = 36

        for i, r in enumerate(visible):
            y = row_y0 + i * row_gap
            date = r.get("date", "")
            gc = r.get("gc_name", "")
            score = int(r.get("score", 0))
            ps = int(r.get("par_score", 0))
            score_str = f"{score} ({ps:+d})"

            self.canvas.create_text(col_x_date, y, text=date, fill="white", font=("Helvetica", 14, "bold"), anchor="center")
            self.canvas.create_text(col_x_gc, y, text=gc, fill="white", font=("Helvetica", 14, "bold"), anchor="center")
            self.canvas.create_text(col_x_score, y, text=score_str, fill="white", font=("Helvetica", 14, "bold"), anchor="center")

        # OK button (confirm.png + OK text)
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
