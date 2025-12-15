# setting.py
#
# - 단일 윈도우(Screen/Frame 전환) 구조 대응 버전
# - Toplevel 생성 제거
# - 기존 UI/로직 유지(OK 이미지 버튼, 토글 스타일/정렬/배경 유지)
# - OK 클릭 시:
#     * 설정 적용
#     * on_back 콜백 호출(메뉴로 복귀 등)
#
# 기존 요구사항 유지:
# 1. OK 버튼을 Canvas 이미지로 처리하여 PNG 투명 합성
# 2. 토글 버튼 Enable/Disable 동일 테두리 유지
# 3. 라벨 정렬
# 4. background.png 유지
# 5. config.py에서 LCD 크기 참조

import tkinter as tk
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Callable

from PIL import Image, ImageTk

from config import LCD_WIDTH, LCD_HEIGHT


# -------------------- 설정 데이터 모델 -------------------- #

@dataclass
class SettingsModel:
    unit: str = "M"
    dist_mode: str = "보정"
    scorecard: str = "사용"


app_settings = SettingsModel()


# -------------------- 설정 화면(Frame) -------------------- #

class SettingsFrame(tk.Frame):
    """
    단일 윈도우에서 임베드 가능한 설정 화면(Frame).
    - on_back: OK 이후 호출(예: manager.show("menu"))
    """
    def __init__(
        self,
        master: tk.Misc,
        settings: Optional[SettingsModel] = None,
        on_back: Optional[Callable[[], None]] = None,
    ):
        super().__init__(master, width=LCD_WIDTH, height=LCD_HEIGHT, bg="black")

        self.settings = settings if settings is not None else app_settings
        self.on_back = on_back

        # 이미지 캐시
        self.images: Dict[str, ImageTk.PhotoImage] = {}
        self.asset_dir = Path(__file__).parent / "assets_image"

        # 편집용 변수
        self.unit_var = tk.StringVar(value=self.settings.unit)
        self.dist_var = tk.StringVar(value=self.settings.dist_mode)
        self.score_var = tk.StringVar(value=self.settings.scorecard)

        # 캔버스
        self.canvas = tk.Canvas(
            self,
            width=LCD_WIDTH,
            height=LCD_HEIGHT,
            highlightthickness=0,
            bg="black",
        )
        self.canvas.pack()

        # UI 구성
        self._build_ui()

    # --------------------------------------------------------------
    # 이미지 로드
    # --------------------------------------------------------------

    def load_image(self, filename: str, size=None):
        key = filename if size is None else f"{filename}_{size[0]}x{size[1]}"
        if key in self.images:
            return self.images[key]

        path = self.asset_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")

        img = Image.open(path).convert("RGBA")
        if size:
            img = img.resize(size, Image.LANCZOS)

        p = ImageTk.PhotoImage(img)
        self.images[key] = p
        return p

    # --------------------------------------------------------------
    # UI 구성
    # --------------------------------------------------------------

    def _build_ui(self):
        self.canvas.delete("all")

        # 배경
        try:
            bg_img = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
            self.canvas.create_image(0, 0, anchor="nw", image=bg_img)
        except FileNotFoundError:
            pass

        # 제목
        self.canvas.create_text(
            LCD_WIDTH // 2,
            90,
            text="설 정",
            fill="white",
            font=("Helvetica", 26, "bold"),
        )

        # 기준 X 좌표
        LABEL_X = 100

        # 항목 라벨
        self.canvas.create_text(
            LABEL_X, 170,
            text="단위  설정",
            fill="white",
            anchor="w",
            font=("Helvetica", 18),
        )
        self.canvas.create_text(
            LABEL_X, 230,
            text="거리  안내",
            fill="white",
            anchor="w",
            font=("Helvetica", 18),
        )
        self.canvas.create_text(
            LABEL_X, 290,
            text="스코어카드",
            fill="white",
            anchor="w",
            font=("Helvetica", 18),
        )

        # 토글 생성
        self._create_toggle_buttons()

        # OK 버튼 생성
        self._create_ok_button()

    # --------------------------------------------------------------
    # 토글 버튼 생성
    # --------------------------------------------------------------

    def _create_toggle_buttons(self):
        btn_width = 6
        btn_font = ("Helvetica", 14, "bold")

        # ---- 단위 ----
        self.btn_unit_m = tk.Button(
            self.canvas,
            text="M",
            width=btn_width,
            font=btn_font,
            command=lambda: self._set_unit("M"),
            borderwidth=2,
            highlightthickness=0,
        )
        self.btn_unit_yd = tk.Button(
            self.canvas,
            text="Yd",
            width=btn_width,
            font=btn_font,
            command=lambda: self._set_unit("Yd"),
            borderwidth=2,
            highlightthickness=0,
        )
        self.canvas.create_window(280, 170, window=self.btn_unit_m)
        self.canvas.create_window(360, 170, window=self.btn_unit_yd)

        # ---- 거리 ----
        self.btn_dist_corr = tk.Button(
            self.canvas,
            text="보정",
            width=btn_width,
            font=btn_font,
            command=lambda: self._set_dist("보정"),
            borderwidth=2,
            highlightthickness=0,
        )
        self.btn_dist_straight = tk.Button(
            self.canvas,
            text="직선",
            width=btn_width,
            font=btn_font,
            command=lambda: self._set_dist("직선"),
            borderwidth=2,
            highlightthickness=0,
        )
        self.canvas.create_window(280, 230, window=self.btn_dist_corr)
        self.canvas.create_window(360, 230, window=self.btn_dist_straight)

        # ---- 스코어카드 ----
        self.btn_score_on = tk.Button(
            self.canvas,
            text="사용",
            width=btn_width,
            font=btn_font,
            command=lambda: self._set_score("사용"),
            borderwidth=2,
            highlightthickness=0,
        )
        self.btn_score_off = tk.Button(
            self.canvas,
            text="안함",
            width=btn_width,
            font=btn_font,
            command=lambda: self._set_score("안함"),
            borderwidth=2,
            highlightthickness=0,
        )
        self.canvas.create_window(280, 290, window=self.btn_score_on)
        self.canvas.create_window(360, 290, window=self.btn_score_off)

        self._refresh_toggle_styles()

    # --------------------------------------------------------------
    # OK 버튼 (Canvas 이미지 방식)
    # --------------------------------------------------------------

    def _create_ok_button(self):
        try:
            ok_img = self.load_image("confirm.png")
        except FileNotFoundError:
            ok_img = None

        if ok_img is not None:
            self.ok_img_id = self.canvas.create_image(
                LCD_WIDTH // 2,
                LCD_HEIGHT - 70,
                image=ok_img,
            )
            self.canvas.tag_bind(self.ok_img_id, "<Button-1>", self._on_ok_event)

        self.ok_text_id = self.canvas.create_text(
            LCD_WIDTH // 2,
            LCD_HEIGHT - 70,
            text="OK",
            fill="white",
            font=("Helvetica", 20, "bold"),
        )
        self.canvas.tag_bind(self.ok_text_id, "<Button-1>", self._on_ok_event)

    def _on_ok_event(self, event):
        self._on_ok()

    # --------------------------------------------------------------
    # 토글 스타일 처리
    # --------------------------------------------------------------

    def _refresh_toggle_styles(self):
        selected_bg = "#2BBF3F"
        normal_bg = "#303030"
        fg = "white"

        def apply(btn: tk.Button, selected: bool):
            btn.config(
                bg=(selected_bg if selected else normal_bg),
                fg=fg,
                activebackground=(selected_bg if selected else normal_bg),
                activeforeground=fg,
                relief="flat",
            )

        apply(self.btn_unit_m, self.unit_var.get() == "M")
        apply(self.btn_unit_yd, self.unit_var.get() == "Yd")

        apply(self.btn_dist_corr, self.dist_var.get() == "보정")
        apply(self.btn_dist_straight, self.dist_var.get() == "직선")

        apply(self.btn_score_on, self.score_var.get() == "사용")
        apply(self.btn_score_off, self.score_var.get() == "안함")

    # --------------------------------------------------------------
    # 토글 동작
    # --------------------------------------------------------------

    def _set_unit(self, v: str):
        self.unit_var.set(v)
        self._refresh_toggle_styles()

    def _set_dist(self, v: str):
        self.dist_var.set(v)
        self._refresh_toggle_styles()

    def _set_score(self, v: str):
        self.score_var.set(v)
        self._refresh_toggle_styles()

    # --------------------------------------------------------------
    # OK 처리
    # --------------------------------------------------------------

    def _on_ok(self):
        self.settings.unit = self.unit_var.get()
        self.settings.dist_mode = self.dist_var.get()
        self.settings.scorecard = self.score_var.get()

        print("=== 설정 적용 ===")
        print("단위:", self.settings.unit)
        print("거리:", self.settings.dist_mode)
        print("스코어카드:", self.settings.scorecard)

        if callable(self.on_back):
            self.on_back()


# --------------------------------------------------------------
# 외부 호출 함수(호환 유지): 더 이상 Toplevel이 아닌 Frame을 반환
# --------------------------------------------------------------

def open_settings_window(
    parent: tk.Misc,
    settings: Optional[SettingsModel] = None,
    on_back: Optional[Callable[[], None]] = None,
) -> SettingsFrame:
    return SettingsFrame(parent, settings=settings, on_back=on_back)


# 단독 실행 테스트(새 윈도우 생성 아님: root 안에 Frame만)
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry(f"{LCD_WIDTH}x{LCD_HEIGHT}")
    frm = SettingsFrame(root, settings=app_settings, on_back=lambda: root.destroy())
    frm.pack()
    root.mainloop()
