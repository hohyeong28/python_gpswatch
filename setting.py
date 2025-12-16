import tkinter as tk
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Callable

from PIL import Image, ImageTk

from config import LCD_WIDTH, LCD_HEIGHT


@dataclass
class SettingsModel:
    unit: str = "M"
    dist_mode: str = "보정"
    scorecard: str = "사용"


app_settings = SettingsModel()


class SettingsFrame(tk.Frame):
    """
    단일 윈도우에서 임베드 가능한 설정 화면(Frame).
    - on_back: OK 이후 호출(예: manager.show("menu") 또는 distance로 복귀)
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

        self.images: Dict[str, ImageTk.PhotoImage] = {}
        self.asset_dir = Path(__file__).parent / "assets_image"

        self.unit_var = tk.StringVar(value=self.settings.unit)
        self.dist_var = tk.StringVar(value=self.settings.dist_mode)
        self.score_var = tk.StringVar(value=self.settings.scorecard)

        self.canvas = tk.Canvas(
            self,
            width=LCD_WIDTH,
            height=LCD_HEIGHT,
            highlightthickness=0,
            bg="black",
        )
        self.canvas.pack()

        self._build_ui()

    # [추가] ScreenManager 진입 시 현재 설정값으로 UI 동기화
    def start(self):
        self.unit_var.set(self.settings.unit)
        self.dist_var.set(self.settings.dist_mode)
        self.score_var.set(self.settings.scorecard)
        self._build_ui()

    # [추가] stop은 현재 타이머 없음(no-op)
    def stop(self):
        return

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

    def _build_ui(self):
        self.canvas.delete("all")

        try:
            bg_img = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
            self.canvas.create_image(0, 0, anchor="nw", image=bg_img)
        except FileNotFoundError:
            pass

        self.canvas.create_text(
            LCD_WIDTH // 2,
            90,
            text="설 정",
            fill="white",
            font=("Helvetica", 26, "bold"),
        )

        LABEL_X = 100
        self.canvas.create_text(LABEL_X, 170, text="단위  설정", fill="white", anchor="w", font=("Helvetica", 18))
        self.canvas.create_text(LABEL_X, 230, text="거리  안내", fill="white", anchor="w", font=("Helvetica", 18))
        self.canvas.create_text(LABEL_X, 290, text="스코어카드", fill="white", anchor="w", font=("Helvetica", 18))

        self._create_toggle_buttons()
        self._create_ok_button()

    def _create_toggle_buttons(self):
        btn_width = 6
        btn_font = ("Helvetica", 14, "bold")

        self.btn_unit_m = tk.Button(self.canvas, text="M", width=btn_width, font=btn_font,
                                    command=lambda: self._set_unit("M"), borderwidth=2, highlightthickness=0)
        self.btn_unit_yd = tk.Button(self.canvas, text="Yd", width=btn_width, font=btn_font,
                                     command=lambda: self._set_unit("Yd"), borderwidth=2, highlightthickness=0)
        self.canvas.create_window(280, 170, window=self.btn_unit_m)
        self.canvas.create_window(360, 170, window=self.btn_unit_yd)

        self.btn_dist_corr = tk.Button(self.canvas, text="보정", width=btn_width, font=btn_font,
                                       command=lambda: self._set_dist("보정"), borderwidth=2, highlightthickness=0)
        self.btn_dist_straight = tk.Button(self.canvas, text="직선", width=btn_width, font=btn_font,
                                           command=lambda: self._set_dist("직선"), borderwidth=2, highlightthickness=0)
        self.canvas.create_window(280, 230, window=self.btn_dist_corr)
        self.canvas.create_window(360, 230, window=self.btn_dist_straight)

        self.btn_score_on = tk.Button(self.canvas, text="사용", width=btn_width, font=btn_font,
                                      command=lambda: self._set_score("사용"), borderwidth=2, highlightthickness=0)
        self.btn_score_off = tk.Button(self.canvas, text="안함", width=btn_width, font=btn_font,
                                       command=lambda: self._set_score("안함"), borderwidth=2, highlightthickness=0)
        self.canvas.create_window(280, 290, window=self.btn_score_on)
        self.canvas.create_window(360, 290, window=self.btn_score_off)

        self._refresh_toggle_styles()

    def _create_ok_button(self):
        try:
            ok_img = self.load_image("confirm.png")
        except FileNotFoundError:
            ok_img = None

        if ok_img is not None:
            self.ok_img_id = self.canvas.create_image(LCD_WIDTH // 2, LCD_HEIGHT - 70, image=ok_img)
            self.canvas.tag_bind(self.ok_img_id, "<Button-1>", self._on_ok_event)

        self.ok_text_id = self.canvas.create_text(
            LCD_WIDTH // 2, LCD_HEIGHT - 70,
            text="OK", fill="white", font=("Helvetica", 20, "bold"),
        )
        self.canvas.tag_bind(self.ok_text_id, "<Button-1>", self._on_ok_event)

    def _on_ok_event(self, event):
        self._on_ok()

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

    def _set_unit(self, v: str):
        self.unit_var.set(v)
        self._refresh_toggle_styles()

    def _set_dist(self, v: str):
        self.dist_var.set(v)
        self._refresh_toggle_styles()

    def _set_score(self, v: str):
        self.score_var.set(v)
        self._refresh_toggle_styles()

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


def open_settings_window(
    parent: tk.Misc,
    settings: Optional[SettingsModel] = None,
    on_back: Optional[Callable[[], None]] = None,
) -> SettingsFrame:
    return SettingsFrame(parent, settings=settings, on_back=on_back)
