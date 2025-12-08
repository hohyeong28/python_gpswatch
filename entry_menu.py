import os
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk

from setting import open_settings_window, app_settings
from playgolf import open_play_golf_window

# LCD 해상도
LCD_WIDTH = 466
LCD_HEIGHT = 466


class GolfWatchApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Gps Golf Watch")

        # 캔버스 생성 (466 x 466)
        self.canvas = tk.Canvas(
            self.root,
            width=LCD_WIDTH,
            height=LCD_HEIGHT,
            highlightthickness=0,
            bg="black",
        )
        self.canvas.pack()

        # 이미지 저장 객체(PhotoImage는 GC 때문에 참조 유지 필요)
        self.images = {}

        # 에셋 폴더 경로
        self.asset_dir = Path(__file__).parent / "assets_image"

        # 이미지 로드 및 배치
        self.draw_background()
        self.draw_entry_line()
        self.draw_play_golf()
        self.draw_setting()
        self.draw_round_history()

        # 터치(클릭) 영역 설정
        self.bind_touch_areas()

    # ---------- 이미지 로드/배치 ---------- #

    def load_image(self, filename: str, size=None):
        """
        /assets_image/filename 이미지를 로드하여 PhotoImage로 리턴.
        size가 지정되면 해당 크기로 리사이즈.
        """
        path = self.asset_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")

        img = Image.open(path).convert("RGBA")
        if size is not None:
            img = img.resize(size, Image.LANCZOS)

        photo = ImageTk.PhotoImage(img)
        # GC 방지를 위해 딕셔너리에 보관
        self.images[filename] = photo
        return photo

    def draw_background(self):
        # background.png를 전체 화면에 맞추어 사용
        bg_img = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
        self.canvas.create_image(0, 0, anchor="nw", image=bg_img)

    def draw_entry_line(self):
        # 경계 분할선 (원본 크기에 맞춰 중앙에 배치)
        entry_img = self.load_image("Entry_Line.png")
        w = entry_img.width()
        h = entry_img.height()
        self.canvas.create_image(
            LCD_WIDTH // 2,
            LCD_HEIGHT // 2,
            anchor="center",
            image=entry_img,
        )

    def draw_play_golf(self):
        # 상단 중앙 Play Golf 이미지
        pg_img = self.load_image("playgolf.png")
        self.canvas.create_image(
            LCD_WIDTH // 2,
            LCD_HEIGHT // 3.5,  # 화면 상단 1/4 지점
            anchor="center",
            image=pg_img,
        )

    def draw_setting(self):
        # 좌하단 "설정" 아이콘 이미지
        st_img = self.load_image("setting.png")
        self.canvas.create_image(
            LCD_WIDTH // 4 + 30,        # 좌측 1/4 지점 + 30 px
            LCD_HEIGHT * 3 // 4 - 30,   # 하단 3/4 지점 - 30 px
            anchor="center",
            image=st_img,
        )

    def draw_round_history(self):
        # 우하단 "라운딩 기록" 아이콘 이미지
        rh_img = self.load_image("round_history.png")
        self.canvas.create_image(
            LCD_WIDTH * 3 // 4 - 30,    # 우측 3/4 지점 - 30 px
            LCD_HEIGHT * 3 // 4 - 30,   # 하단 3/4 지점 - 30 px
            anchor="center",
            image=rh_img,
        )

    # ---------- 터치 영역 및 콜백 ---------- #

    def bind_touch_areas(self):
        """
        화면을 3개의 논리 영역으로 나누어 클릭 이벤트를 바인딩한다.
        1) 상단 반원 영역(여기서는 단순히 상단 사각형) : Play Golf
        2) 좌하단 사각형 : 설정
        3) 우하단 사각형 : 라운딩 기록
        실제 LCD에서는 좌표 보정이 가능하므로 필요 시 좌표만 조정하면 된다.
        """

        # 1. Play Golf 영역 (상단 전체)
        play_region = self.canvas.create_rectangle(
            0,
            0,
            LCD_WIDTH,
            LCD_HEIGHT // 2,
            outline="",  # 보이지 않게
            fill="",
        )
        self.canvas.tag_bind(play_region, "<Button-1>", self.on_play_golf)

        # 2. 설정 영역 (좌측 하단)
        setting_region = self.canvas.create_rectangle(
            0,
            LCD_HEIGHT // 2,
            LCD_WIDTH // 2,
            LCD_HEIGHT,
            outline="",
            fill="",
        )
        self.canvas.tag_bind(setting_region, "<Button-1>", self.on_setting)

        # 3. 라운딩 기록 영역 (우측 하단)
        history_region = self.canvas.create_rectangle(
            LCD_WIDTH // 2,
            LCD_HEIGHT // 2,
            LCD_WIDTH,
            LCD_HEIGHT,
            outline="",
            fill="",
        )
        self.canvas.tag_bind(history_region, "<Button-1>", self.on_round_history)

    # ---------- 콜백 함수 (실제 기능 부분에 연결) ---------- #

    def on_play_golf(self, event):
        print("Play Golf 화면으로 이동합니다.")
        # playgolf.py의 Toplevel 실행
        open_play_golf_window(self.root)

    def on_setting(self, event):
        print("설정 화면으로 이동합니다.")
        # setting.py 의 모달 설정창 실행
        open_settings_window(self.root, app_settings)

    def on_round_history(self, event):
        print("라운딩 기록 화면으로 이동합니다.")
        # TODO: 실제 라운딩 기록 화면 전환 코드로 교체


if __name__ == "__main__":
    root = tk.Tk()
    app = GolfWatchApp(root)
    root.mainloop()
