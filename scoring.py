# scoring.py (Frame/Screen 버전)
#
# - 단일 윈도우(Screen/Frame 전환) 구조 대응
# - Toplevel 제거 → ScoringScreen(tk.Frame)
# - NextHole.py 직접 호출 제거 (전환/정책은 entry_menu(ScreenManager)가 담당)
# - OK 버튼 클릭 시:
#     * Hole, PAR, par_score, score(total), putt_score 를 dict로 만들어
#     * on_done(result) 콜백 호출
# - BACK(좌상단) 제공: on_back() 콜백 호출
# - 단독 실행 코드 제거
#
# [추가/수정]
# - 시뮬레이션용: 10초간 입력이 없으면 자동 OK
# - 자동 OK는 수동 OK와 "완전히 동일"하게 동작:
#   -> 동일한 data dict 생성 + 동일한 on_done 호출 경로(_on_ok) 사용
# - +/-/퍼트수 선택 등 모든 입력은 자동 OK 타이머 리셋
# - OK/BACK 시 자동 OK 타이머 취소(중복 호출 방지)

from pathlib import Path
from typing import Dict, Optional, Callable

import tkinter as tk
from PIL import Image, ImageTk
import pandas as pd

from config import LCD_WIDTH, LCD_HEIGHT


PAR_SCORE_LABELS: Dict[int, str] = {
    -2: "이글",
    -1: "버디",
    0: "파",
    1: "보기",
    2: "D-보기",
    3: "T-보기",
    4: "Q-보기",
}


class ScoringScreen(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        on_back: Optional[Callable[[], None]] = None,
        on_done: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(master, width=LCD_WIDTH, height=LCD_HEIGHT, bg="black")
        self.pack_propagate(False)

        self.on_back = on_back
        self.on_done = on_done

        self.base_dir = Path(__file__).parent
        self.asset_dir = self.base_dir / "assets_image"

        # 이미지 캐시
        self.images: Dict[str, ImageTk.PhotoImage] = {}

        # context
        self.parent_window: Optional[tk.Misc] = None
        self.hole_row: Optional[pd.Series] = None

        # 상태 값
        self.par: int = 4
        self.par_score: int = 0          # -2 ~ +4
        self.putt_score: int = 2         # 기본 2
        self.active_field: str = "par"   # "par" 또는 "putt"

        # [추가] 자동 OK 타이머(시뮬레이션 10초)
        self._auto_ok_after_id: Optional[str] = None
        self._auto_ok_timeout_ms: int = 10_000  # 추후 필드 적용 시 변경 예정

        # 캔버스
        self.canvas = tk.Canvas(
            self,
            width=LCD_WIDTH,
            height=LCD_HEIGHT,
            highlightthickness=0,
            bg="black",
        )
        self.canvas.pack()

        # BACK hitbox (좌상단)
        self._draw_back_hitbox()

    # ---------------- public API ---------------- #

    def set_context(self, parent_window: tk.Misc, hole_row: pd.Series):
        """
        entry_menu(ScreenManager)에서 scoring 화면 진입 시 호출.
        """
        self.parent_window = parent_window
        self.hole_row = hole_row

        self.par = int(hole_row.get("PAR", 4))
        self.par_score = 0
        self.putt_score = 2
        self.active_field = "par"

        # 컨텍스트 갱신 시 타이머는 start()에서 예약(렌더 후)하도록 함

    def start(self):
        """
        scoring 화면 표시 시 호출(1회 렌더).
        - 렌더 후 자동 OK 타이머 시작
        """
        self._render_screen()
        self._schedule_auto_ok()

    def stop(self):
        """
        (선택적으로 entry_menu에서 호출 가능)
        - 자동 OK 타이머 취소
        """
        self._cancel_auto_ok()

    # ---------------- auto OK timer ---------------- #

    def _cancel_auto_ok(self):
        if self._auto_ok_after_id is None:
            return
        try:
            self.after_cancel(self._auto_ok_after_id)
        except Exception:
            pass
        self._auto_ok_after_id = None

    def _schedule_auto_ok(self):
        """
        '무입력 10초'를 위해, 입력이 있을 때마다 이 함수를 호출하여 타이머를 리셋한다.
        """
        self._cancel_auto_ok()
        self._auto_ok_after_id = self.after(self._auto_ok_timeout_ms, self._on_auto_ok_timeout)

    def _on_auto_ok_timeout(self):
        """
        10초 무입력 시 자동 OK.
        수동 OK와 완전히 동일하게 동작하도록 _on_ok(None) 호출.
        """
        self._auto_ok_after_id = None
        # 자동 OK는 수동 OK와 동일 경로
        self._on_ok(event=None)

    # ---------------- image ---------------- #

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

    # ---------------- helpers ---------------- #

    @property
    def par_score_text(self) -> str:
        return PAR_SCORE_LABELS.get(self.par_score, f"{self.par_score:+d}")

    @property
    def total_score(self) -> int:
        return self.par + self.par_score

    def _clamp_values(self):
        self.par_score = max(-2, min(4, self.par_score))
        self.putt_score = max(0, self.putt_score)

    # ---------------- back ---------------- #

    def _draw_back_hitbox(self):
        self.canvas.create_text(
            60, 40,
            text="BACK",
            fill="white",
            font=("Helvetica", 14, "bold"),
        )
        back_region = self.canvas.create_rectangle(0, 0, 120, 80, outline="", fill="")
        self.canvas.tag_bind(back_region, "<Button-1>", lambda e: self._on_back())

    def _on_back(self):
        # BACK은 '입력'이지만, 자동 OK 타이머는 의미 없으므로 취소
        self._cancel_auto_ok()
        if callable(self.on_back):
            self.on_back()

    # ---------------- render ---------------- #

    def _render_screen(self):
        self.canvas.delete("all")
        self._draw_back_hitbox()

        center_y = LCD_HEIGHT // 2

        # background
        try:
            bg_img = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
            self.canvas.create_image(0, 0, anchor="nw", image=bg_img)
        except FileNotFoundError:
            pass

        # Score_Line
        try:
            line_img = self.load_image("Score_Line.png")
            self.canvas.create_image(LCD_WIDTH // 2, center_y, image=line_img)
        except FileNotFoundError:
            pass

        # 상단: 홀/파
        hole_no = self.hole_row.get("Hole", "") if self.hole_row is not None else ""
        self.canvas.create_text(
            LCD_WIDTH // 2 - 40,
            center_y - 140,
            text=f"H{hole_no}",
            fill="white",
            font=("Helvetica", 24, "bold"),
        )
        self.canvas.create_text(
            LCD_WIDTH // 2 + 40,
            center_y - 140,
            text=f"P{self.par}",
            fill="white",
            font=("Helvetica", 24, "bold"),
        )

        # minus
        try:
            minus_img = self.load_image("minus.png")
            minus_id = self.canvas.create_image(LCD_WIDTH // 2 - 90, center_y - 70, image=minus_img)
            self.canvas.tag_bind(minus_id, "<Button-1>", self._on_minus)
        except FileNotFoundError:
            pass

        # plus
        try:
            plus_img = self.load_image("plus.png")
            plus_id = self.canvas.create_image(LCD_WIDTH // 2 + 90, center_y - 70, image=plus_img)
            self.canvas.tag_bind(plus_id, "<Button-1>", self._on_plus)
        except FileNotFoundError:
            pass

        # par_score_text / (par_score)
        self.canvas.create_text(
            LCD_WIDTH // 2,
            center_y - 80,
            text=self.par_score_text,
            fill="white",
            font=("Helvetica", 24, "bold"),
        )
        self.canvas.create_text(
            LCD_WIDTH // 2,
            center_y - 40,
            text=f"({self.par_score:+d})",
            fill="white",
            font=("Helvetica", 24),
        )

        # 하단 좌/우 라벨
        self.canvas.create_text(
            LCD_WIDTH // 4 + 20,
            center_y + 40,
            text="스코어",
            fill="white",
            font=("Helvetica", 18, "bold"),
        )

        putt_color = "lime" if self.active_field == "putt" else "white"
        putt_label_id = self.canvas.create_text(
            LCD_WIDTH * 3 // 4 - 20,
            center_y + 40,
            text="퍼트수",
            fill=putt_color,
            font=("Helvetica", 18, "bold"),
        )
        self.canvas.tag_bind(putt_label_id, "<Button-1>", self._on_select_putt)

        # 좌측: score, 우측: putt_score
        self.canvas.create_text(
            LCD_WIDTH // 4 + 20,
            center_y + 90,
            text=str(self.total_score),
            fill="white",
            font=("Helvetica", 30, "bold"),
        )
        putt_value_color = "lime" if self.active_field == "putt" else "white"
        self.canvas.create_text(
            LCD_WIDTH * 3 // 4 - 20,
            center_y + 90,
            text=str(self.putt_score),
            fill=putt_value_color,
            font=("Helvetica", 30, "bold"),
        )

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

    # ---------------- events ---------------- #

    def _on_select_putt(self, event):
        self._schedule_auto_ok()  # 입력 발생 -> 타이머 리셋
        self.active_field = "par" if self.active_field == "putt" else "putt"
        self._render_screen()

    def _on_plus(self, event):
        self._schedule_auto_ok()  # 입력 발생 -> 타이머 리셋
        if self.active_field == "putt":
            self.putt_score += 1
        else:
            self.par_score += 1
        self._clamp_values()
        self._render_screen()

    def _on_minus(self, event):
        self._schedule_auto_ok()  # 입력 발생 -> 타이머 리셋
        if self.active_field == "putt":
            self.putt_score -= 1
        else:
            self.par_score -= 1
        self._clamp_values()
        self._render_screen()

    def _on_ok(self, event):
        """
        OK 클릭/자동 OK 시:
          - 결과 dict 생성
          - on_done(result) 콜백 호출
          - (다음 화면 전환은 entry_menu가 담당)
        """
        # OK는 종료 이벤트이므로 자동 OK 타이머 취소(중복 방지)
        self._cancel_auto_ok()

        hole_no = int(self.hole_row.get("Hole", 0)) if self.hole_row is not None else 0
        data = {
            "Hole": hole_no,
            "PAR": self.par,
            "par_score": self.par_score,
            "score": self.total_score,
            "putt_score": self.putt_score,
        }

        if callable(self.on_done):
            self.on_done(data)
        else:
            print("[ScoringScreen] scoring result:", data)


# (호환용 별칭)
ScoringFrame = ScoringScreen
