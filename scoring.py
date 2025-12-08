# scoring.py
#
# - 퍼팅 후 스코어 입력 화면
# - 배경: background.png (다른 화면과 동일)
# - 중앙: Score_Line.png (가로 라인)
# - 상단: 홀 정보 / 파 정보 (예: "H10  P4")
# - 중앙 상단: [-] [파 (0)] [+]
# - 하단 좌측: "스코어" / 값(score)
# - 하단 우측: "퍼트수" / 값(putt_score)
# - 퍼트수 텍스트는 선택 시 (활성화) 녹색, 기본은 흰색
# - +/- 기본 대상은 par_score (이글/버디/파/보기/더블보기/트리플보기/쿼드보기)
#   퍼트수를 선택(클릭)하면 이후 +/- 가 putt_score 를 조정
# - OK 버튼 클릭 시:
#   * 홀정보, 파정보, score, putt_score, par_score 를 저장(콜백 호출)
#   * NextHole.py 호출

from pathlib import Path
from typing import Dict, Optional

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


class ScoringWindow(tk.Toplevel):
    def __init__(self, parent: tk.Tk, hole_row: pd.Series):
        super().__init__(parent)

        self.title("Scoring")
        self.resizable(False, False)

        # 부모 창 근처 배치
        self.geometry(
            f"{LCD_WIDTH}x{LCD_HEIGHT}+"
            f"{parent.winfo_rootx() + 40}+{parent.winfo_rooty() + 40}"
        )

        self.parent_window = parent
        self.hole_row = hole_row

        self.base_dir = Path(__file__).parent
        self.asset_dir = self.base_dir / "assets_image"

        # 이미지 캐시
        self.images: Dict[str, ImageTk.PhotoImage] = {}

        # 상태 값
        self.par: int = int(hole_row.get("PAR", 4))  # 기본 PAR 4
        self.par_score: int = 0                      # -2 ~ +4
        self.putt_score: int = 2                     # 기본 2
        self.active_field: str = "par"               # "par" 또는 "putt"

        # 캔버스
        self.canvas = tk.Canvas(
            self,
            width=LCD_WIDTH,
            height=LCD_HEIGHT,
            highlightthickness=0,
            bg="black",
        )
        self.canvas.pack()

        # 초기 렌더링
        self._render_screen()

    # ------------- 이미지 로드 ------------- #

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

    # ------------- 내부 계산 헬퍼 ------------- #

    @property
    def par_score_text(self) -> str:
        return PAR_SCORE_LABELS.get(self.par_score, f"{self.par_score:+d}")

    @property
    def total_score(self) -> int:
        """해당 홀의 실제 타수 (PAR + par_score)."""
        return self.par + self.par_score

    def _clamp_values(self):
        # par_score 범위 제한 (-2 ~ +4)
        if self.par_score < -2:
            self.par_score = -2
        if self.par_score > 4:
            self.par_score = 4

        # putt_score 는 0 이상만 허용
        if self.putt_score < 0:
            self.putt_score = 0

    # ------------- 화면 렌더링 ------------- #

    def _render_screen(self):
        self.canvas.delete("all")

        center_y = LCD_HEIGHT // 2

        # background.png 전체 적용
        try:
            bg_img = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
            self.canvas.create_image(0, 0, anchor="nw", image=bg_img)
        except FileNotFoundError:
            pass

        # 중앙 라인: Score_Line.png
        try:
            line_img = self.load_image("Score_Line.png")
            self.canvas.create_image(
                LCD_WIDTH // 2,
                center_y,
                image=line_img,
            )
        except FileNotFoundError as e:
            print(e)

        # 상단: 홀 정보 / 파 정보 (예: H10  P4)
        hole_no = self.hole_row.get("Hole", "")
        par = self.par
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
            text=f"P{par}",
            fill="white",
            font=("Helvetica", 24, "bold"),
        )

        # 가운데: [-] 파 (0) [+]
        # minus 아이콘
        try:
            minus_img = self.load_image("minus.png")
            minus_id = self.canvas.create_image(
                LCD_WIDTH // 2 - 90,
                center_y - 70,
                image=minus_img,
            )
            self.canvas.tag_bind(minus_id, "<Button-1>", self._on_minus)
        except FileNotFoundError as e:
            print(e)

        # plus 아이콘
        try:
            plus_img = self.load_image("plus.png")
            plus_id = self.canvas.create_image(
                LCD_WIDTH // 2 + 90,
                center_y - 70,
                image=plus_img,
            )
            self.canvas.tag_bind(plus_id, "<Button-1>", self._on_plus)
        except FileNotFoundError as e:
            print(e)

        # par_score_text / par_score (괄호 안)
        par_label = self.par_score_text
        self.canvas.create_text(
            LCD_WIDTH // 2,
            center_y - 80,
            text=par_label,
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

        # 하단 좌/우 영역 텍스트
        # 좌측: "스코어"
        self.canvas.create_text(
            LCD_WIDTH // 4 + 20 ,
            center_y + 40,
            text="스코어",
            fill="white",
            font=("Helvetica", 18, "bold"),
        )
        # 우측: "퍼트수" (선택 상태에 따라 색 변경)
        putt_color = "lime" if self.active_field == "putt" else "white"
        putt_label_id = self.canvas.create_text(
            LCD_WIDTH * 3 // 4 - 20,
            center_y + 40,
            text="퍼트수",
            fill=putt_color,
            font=("Helvetica", 18, "bold"),
        )
        # 퍼트수 텍스트를 선택 영역으로 사용
        self.canvas.tag_bind(putt_label_id, "<Button-1>", self._on_select_putt)

        # 좌측 값: score (PAR + par_score)
        self.canvas.create_text(
            LCD_WIDTH // 4 + 20,
            center_y + 90,
            text=str(self.total_score),
            fill="white",
            font=("Helvetica", 30, "bold"),
        )
        # 우측 값: putt_score
        putt_value_color = "lime" if self.active_field == "putt" else "white"
        self.canvas.create_text(
            LCD_WIDTH * 3 // 4 - 20,
            center_y + 90,
            text=str(self.putt_score),
            fill=putt_value_color,
            font=("Helvetica", 30, "bold"),
        )

        # ---------------- OK 버튼 (Canvas 이미지 + 텍스트) ----------------
        try:
            ok_img = self.load_image("confirm.png")

            # 1) OK 이미지 추가
            ok_img_id = self.canvas.create_image(
                LCD_WIDTH // 2,
                LCD_HEIGHT - 70,
                image=ok_img,
            )

            # 2) OK 텍스트 추가 (이미지 위에 겹치는 레이어)
            ok_text_id = self.canvas.create_text(
                LCD_WIDTH // 2,
                LCD_HEIGHT - 70,
                text="OK",
                fill="white",
                font=("Helvetica", 20, "bold"),
            )

            # 3) 이미지와 텍스트 모두 클릭 가능하도록 이벤트 바인딩
            self.canvas.tag_bind(ok_img_id, "<Button-1>", self._on_ok)
            self.canvas.tag_bind(ok_text_id, "<Button-1>", self._on_ok)

        except FileNotFoundError:
            # 이미지가 없어도 텍스트 버튼으로라도 표시 (fallback)
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

    # ------------- 이벤트 핸들러 ------------- #

    def _on_select_putt(self, event):
        """
        퍼트수 텍스트를 클릭하면
          - par 모드 → putt 모드로 전환 (텍스트 초록색, +/-는 putt_score 조절)
          - putt 모드 → par 모드로 전환 (텍스트 흰색, +/-는 par_score 조절)
        """
        if self.active_field == "putt":
            # 이미 퍼트수 모드면 다시 par 모드로
            self.active_field = "par"
        else:
            # 기본(par) 모드에서 클릭 시 퍼트수 모드로
            self.active_field = "putt"

        self._render_screen()

    def _on_plus(self, event):
        """+ 버튼: 기본은 par_score, 퍼트수 활성 시 putt_score."""
        if self.active_field == "putt":
            self.putt_score += 1
        else:
            self.par_score += 1

        self._clamp_values()
        self._render_screen()

    def _on_minus(self, event):
        """- 버튼: 기본은 par_score, 퍼트수 활성 시 putt_score."""
        if self.active_field == "putt":
            self.putt_score -= 1
        else:
            self.par_score -= 1

        self._clamp_values()
        self._render_screen()

    def _on_ok(self, event):
        """
        OK 클릭 시:
          1) 홀번호, PAR, par_score, total_score, putt_score 저장 (콜백)
          2) NextHole.py 호출
        """
        data = {
            "Hole": int(self.hole_row.get("Hole", 0)),
            "PAR": self.par,
            "score": self.total_score,
            "putt_score": self.putt_score,
        }

        # 부모에 저장 콜백이 있으면 호출
        if hasattr(self.parent_window, "on_scoring_done"):
            try:
                self.parent_window.on_scoring_done(data)
            except Exception as e:
                print("[ScoringWindow] on_scoring_done 콜백 오류:", e)
        else:
            print("[ScoringWindow] Scoring result:", data)

        # NextHole.py 호출
        try:
            from NextHole import open_next_hole_window
            open_next_hole_window(parent=self.parent_window, prev_hole_data=data)
        except ImportError:
            print("[ScoringWindow] NextHole 모듈을 찾을 수 없습니다.")
        except Exception as e:
            print("[ScoringWindow] NextHole 실행 중 오류:", e)

        # 현재 스코어 입력 창 닫기
        self.destroy()


def open_scoring_window(parent: tk.Tk, hole_row: pd.Series) -> ScoringWindow:
    """
    meas_putt_distance.py 에서 호출할 진입 함수.
    """
    return ScoringWindow(parent=parent, hole_row=hole_row)


if __name__ == "__main__":
    # 단독 테스트용
    root = tk.Tk()
    root.title("Scoring Test")

    dummy_row = pd.Series(
        {
            "GC_name_code": 1002,
            "course_name": "아웃",
            "Hole": 10,
            "PAR": 4,
        }
    )

    win = open_scoring_window(root, dummy_row)
    root.mainloop()
