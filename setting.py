# setting.py
#
# 개선 사항 반영 최종본:
# 1. OK 버튼을 Canvas 이미지로 처리하여 PNG의 투명 부분이 배경과 자연스럽게 합성되도록 변경
# 2. 토글 버튼 Enable/Disable 동일한 테두리(border) 유지
# 3. 단위/거리/스코어카드 라벨을 공통 기준 X좌표로 정렬
# 4. background.png 유지
# 5. config.py에서 LCD 크기 참조

import tkinter as tk
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict
from PIL import Image, ImageTk

from config import LCD_WIDTH, LCD_HEIGHT


# -------------------- 설정 데이터 모델 -------------------- #

@dataclass
class SettingsModel:
    unit: str = "M"
    dist_mode: str = "보정"
    scorecard: str = "사용"


app_settings = SettingsModel()


# -------------------- 설정 창 -------------------- #

class SettingWindow(tk.Toplevel):

    def __init__(self, master: tk.Tk, settings: Optional[SettingsModel] = None):
        super().__init__(master)

        self.title("설정")
        self.resizable(False, False)

        # 위치 조정
        self.geometry(
            f"{LCD_WIDTH}x{LCD_HEIGHT}+"
            f"{master.winfo_rootx() + 20}+{master.winfo_rooty() + 20}"
        )

        self.settings = settings if settings is not None else app_settings

        # 이미지 캐시
        self.images: Dict[str, ImageTk.PhotoImage] = {}
        self.asset_dir = Path(__file__).parent / "assets_image"

        # 편집용 변수
        self.unit_var = tk.StringVar(value=self.settings.unit)
        self.dist_var = tk.StringVar(value=self.settings.dist_mode)
        self.score_var = tk.StringVar(value=self.settings.scorecard)

        # 캔버스 생성
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

        # 모달
        self.transient(master)
        self.grab_set()

    # --------------------------------------------------------------
    # 이미지 로드
    # --------------------------------------------------------------

    def load_image(self, filename: str, size=None):
        path = self.asset_dir / filename
        img = Image.open(path).convert("RGBA")
        if size:
            img = img.resize(size, Image.LANCZOS)
        p = ImageTk.PhotoImage(img)
        self.images[filename] = p
        return p

    # --------------------------------------------------------------
    # UI 구성
    # --------------------------------------------------------------

    def _build_ui(self):

        # 배경
        bg_img = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
        self.canvas.create_image(0, 0, anchor="nw", image=bg_img)

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
            highlightthickness=0
        )
        self.btn_unit_yd = tk.Button(
            self.canvas,
            text="Yd",
            width=btn_width,
            font=btn_font,
            command=lambda: self._set_unit("Yd"),
            borderwidth=2,
            highlightthickness=0
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
            highlightthickness=0
        )
        self.btn_dist_straight = tk.Button(
            self.canvas,
            text="직선",
            width=btn_width,
            font=btn_font,
            command=lambda: self._set_dist("직선"),
            borderwidth=2,
            highlightthickness=0
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
            highlightthickness=0
        )
        self.btn_score_off = tk.Button(
            self.canvas,
            text="안함",
            width=btn_width,
            font=btn_font,
            command=lambda: self._set_score("안함"),
            borderwidth=2,
            highlightthickness=0
        )
        self.canvas.create_window(280, 290, window=self.btn_score_on)
        self.canvas.create_window(360, 290, window=self.btn_score_off)

        self._refresh_toggle_styles()

    # --------------------------------------------------------------
    # OK 버튼 (Canvas 이미지 방식)
    # --------------------------------------------------------------

    def _create_ok_button(self):
        ok_img = self.load_image("confirm.png")

        # 1) OK 이미지 추가
        self.ok_img_id = self.canvas.create_image(
            LCD_WIDTH // 2,
            LCD_HEIGHT - 70,
            image=ok_img
        )

        # 2) OK 텍스트 추가 (이미지 위에 겹치는 레이어)
        self.ok_text_id = self.canvas.create_text(
            LCD_WIDTH // 2,
            LCD_HEIGHT - 70,
            text="OK",
            fill="white",
            font=("Helvetica", 20, "bold")
        )

        # 3) 이미지 클릭 가능하도록 이벤트 바인딩
        self.canvas.tag_bind(self.ok_img_id, "<Button-1>", self._on_ok_event)

        # 4) 텍스트도 클릭되도록 이벤트 바인딩
        self.canvas.tag_bind(self.ok_text_id, "<Button-1>", self._on_ok_event)

    # Canvas 이벤트 래퍼
    def _on_ok_event(self, event):
        self._on_ok()

    # --------------------------------------------------------------
    # 토글 스타일 처리
    # --------------------------------------------------------------

    def _refresh_toggle_styles(self):

        selected_bg = "#2BBF3F"
        normal_bg = "#303030"
        fg = "white"

        def apply(btn, selected):
            btn.config(
                bg=(selected_bg if selected else normal_bg),
                fg=fg,
                activebackground=(selected_bg if selected else normal_bg),
                activeforeground=fg,
                relief="flat"      # 전부 동일한 테두리 유지
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

    def _set_unit(self, v):
        self.unit_var.set(v)
        self._refresh_toggle_styles()

    def _set_dist(self, v):
        self.dist_var.set(v)
        self._refresh_toggle_styles()

    def _set_score(self, v):
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

        self.destroy()


# --------------------------------------------------------------
# 외부 호출 함수
# --------------------------------------------------------------

def open_settings_window(parent: tk.Tk, settings: Optional[SettingsModel] = None):
    return SettingWindow(parent, settings)
