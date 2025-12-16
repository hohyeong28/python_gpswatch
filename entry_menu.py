import tkinter as tk
from pathlib import Path
from typing import Dict, Optional, Callable

from PIL import Image, ImageTk

from config import LCD_WIDTH, LCD_HEIGHT

from setting import app_settings
import setting
import playgolf

from measure_distance import DistanceScreen
from green_view import GreenViewScreen
from layup_view import LayupViewScreen
from meas_putt_distance import PuttDistanceScreen
from scoring import ScoringScreen


# ---------------- Screen Manager ---------------- #

class ScreenManager(tk.Frame):
    """
    단일 Root 안에서 여러 화면(Frame)을 교체하는 관리자.
    - show(name): 해당 화면을 최상단으로 올림
    """
    def __init__(self, root: tk.Tk):
        super().__init__(root, width=LCD_WIDTH, height=LCD_HEIGHT, bg="black")
        self.root = root
        self.pack_propagate(False)
        self.pack()

        self.screens: Dict[str, tk.Frame] = {}

    def add(self, name: str, screen: tk.Frame):
        self.screens[name] = screen
        screen.place(x=0, y=0, width=LCD_WIDTH, height=LCD_HEIGHT)

    def show(self, name: str):
        if name not in self.screens:
            raise KeyError(f"Unknown screen: {name}")
        self.screens[name].tkraise()


# ---------------- Base Screen ---------------- #

class BaseScreen(tk.Frame):
    def __init__(self, manager: ScreenManager, asset_dir: Path):
        super().__init__(manager, width=LCD_WIDTH, height=LCD_HEIGHT, bg="black")
        self.manager = manager
        self.asset_dir = asset_dir
        self.images: Dict[str, ImageTk.PhotoImage] = {}

    def load_image(self, filename: str, size=None) -> ImageTk.PhotoImage:
        path = self.asset_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")

        img = Image.open(path).convert("RGBA")
        if size is not None:
            img = img.resize(size, Image.LANCZOS)

        photo = ImageTk.PhotoImage(img)
        self.images[f"{filename}_{size}"] = photo
        return photo


# ---------------- Menu Screen (Entry) ---------------- #

class MenuScreen(BaseScreen):
    def __init__(
        self,
        manager: ScreenManager,
        asset_dir: Path,
        on_play: Callable[[], None],
    ):
        super().__init__(manager, asset_dir)

        self.on_play = on_play

        self.canvas = tk.Canvas(
            self, width=LCD_WIDTH, height=LCD_HEIGHT,
            highlightthickness=0, bg="black",
        )
        self.canvas.pack(fill="both", expand=True)

        self._draw_menu()
        self._bind_touch_areas()

    def _draw_menu(self):
        self.canvas.delete("all")

        try:
            bg = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
            self.canvas.create_image(0, 0, anchor="nw", image=bg)
        except FileNotFoundError:
            pass

        try:
            entry_img = self.load_image("Entry_Line.png")
            self.canvas.create_image(LCD_WIDTH // 2, LCD_HEIGHT // 2, anchor="center", image=entry_img)
        except FileNotFoundError:
            pass

        try:
            pg_img = self.load_image("playgolf.png")
            self.canvas.create_image(LCD_WIDTH // 2, int(LCD_HEIGHT / 3.5), anchor="center", image=pg_img)
        except FileNotFoundError:
            pass

        try:
            st_img = self.load_image("setting.png")
            self.canvas.create_image(LCD_WIDTH // 4 + 30, LCD_HEIGHT * 3 // 4 - 30, anchor="center", image=st_img)
        except FileNotFoundError:
            pass

        try:
            rh_img = self.load_image("round_history.png")
            self.canvas.create_image(LCD_WIDTH * 3 // 4 - 30, LCD_HEIGHT * 3 // 4 - 30, anchor="center", image=rh_img)
        except FileNotFoundError:
            pass

    def _bind_touch_areas(self):
        # Play Golf: on_play() 호출하여 start + show 수행
        play_region = self.canvas.create_rectangle(0, 0, LCD_WIDTH, LCD_HEIGHT // 2, outline="", fill="")
        self.canvas.tag_bind(play_region, "<Button-1>", lambda e: self.on_play())

        # Setting
        setting_region = self.canvas.create_rectangle(0, LCD_HEIGHT // 2, LCD_WIDTH // 2, LCD_HEIGHT, outline="", fill="")
        self.canvas.tag_bind(setting_region, "<Button-1>", lambda e: self.manager.show("settings"))

        # History
        history_region = self.canvas.create_rectangle(LCD_WIDTH // 2, LCD_HEIGHT // 2, LCD_WIDTH, LCD_HEIGHT, outline="", fill="")
        self.canvas.tag_bind(history_region, "<Button-1>", lambda e: self.manager.show("history"))


# ---------------- History Screen (placeholder) ---------------- #

class HistoryScreen(BaseScreen):
    def __init__(self, manager: ScreenManager, asset_dir: Path):
        super().__init__(manager, asset_dir)

        self.canvas = tk.Canvas(self, width=LCD_WIDTH, height=LCD_HEIGHT, highlightthickness=0, bg="black")
        self.canvas.pack(fill="both", expand=True)

        self.btn_back = tk.Button(self, text="BACK", command=lambda: self.manager.show("menu"))
        self._render()

    def _render(self):
        self.canvas.delete("all")
        try:
            bg = self.load_image("background.png", size=(LCD_WIDTH, LCD_HEIGHT))
            self.canvas.create_image(0, 0, anchor="nw", image=bg)
        except FileNotFoundError:
            pass

        self.canvas.create_text(LCD_WIDTH // 2, 80, text="ROUND HISTORY", fill="white", font=("Helvetica", 24, "bold"))
        self.canvas.create_text(LCD_WIDTH // 2, LCD_HEIGHT // 2, text="(TODO)", fill="white", font=("Helvetica", 14, "bold"))
        self.canvas.create_window(LCD_WIDTH - 60, 30, window=self.btn_back)


# ---------------- App ---------------- #

class GolfWatchApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Gps Golf Watch")
        self.root.resizable(False, False)

        self.asset_dir = Path(__file__).parent / "assets_image"
        self.manager = ScreenManager(root)

        # ----------- 화면 인스턴스 생성 ----------- #

        self.playgolf_screen = playgolf.PlayGolfFrame(
            self.manager,
            on_back=self._back_from_playgolf,
            on_open_distance=self._open_distance_from_playgolf,
        )

        self.menu_screen = MenuScreen(
            self.manager,
            self.asset_dir,
            on_play=self._go_playgolf,
        )

        self.settings_screen = setting.SettingsFrame(
            self.manager,
            settings=app_settings,
            on_back=lambda: self.manager.show("menu"),
        )

        self.distance_screen = DistanceScreen(
            self.manager,
            on_back=self._back_to_playgolf_from_distance,
            on_open_green_view=self._open_green_view_from_distance,
            on_open_layup_view=self._open_layup_view_from_distance,
            on_open_putt=self._open_putt_from_distance,
        )

        self.green_view_screen = GreenViewScreen(
            self.manager,
            on_back=self._back_to_distance_from_green,
        )

        self.layup_view_screen = LayupViewScreen(
            self.manager,
            on_back=self._back_to_distance_from_layup,
        )

        self.putt_screen = PuttDistanceScreen(
            self.manager,
            on_back=self._back_to_distance_from_putt,
            on_open_scoring=self._open_scoring_from_putt,
        )

        # [보완] scoring BACK도 stop 처리 포함
        self.scoring_screen = ScoringScreen(
            self.manager,
            on_back=self._back_from_scoring,
            on_done=self._on_scoring_done,
        )

        self.history_screen = HistoryScreen(self.manager, self.asset_dir)

        # ----------- ScreenManager 등록 ----------- #
        self.manager.add("menu", self.menu_screen)
        self.manager.add("settings", self.settings_screen)
        self.manager.add("playgolf", self.playgolf_screen)
        self.manager.add("distance", self.distance_screen)
        self.manager.add("green", self.green_view_screen)
        self.manager.add("layup", self.layup_view_screen)
        self.manager.add("putt", self.putt_screen)
        self.manager.add("scoring", self.scoring_screen)
        self.manager.add("history", self.history_screen)

        self.manager.show("menu")

    # ----------------- 공통 전환 유틸 ----------------- #

    def _go_playgolf(self):
        try:
            self.playgolf_screen.start()
        except Exception as e:
            print("[ENTRY] playgolf start error:", e)
        self.manager.show("playgolf")

    def _back_from_playgolf(self):
        try:
            self.playgolf_screen.stop()
        except Exception as e:
            print("[ENTRY] playgolf stop error:", e)
        self.manager.show("menu")

    # ----------------- scoring back 보완 ----------------- #

    def _back_from_scoring(self):
        # [보완] scoring 자동 OK 타이머(after) 취소 보장
        try:
            self.scoring_screen.stop()
        except Exception:
            pass
        self.manager.show("menu")

    # ----------------- 전환 핸들러들 ----------------- #

    def _open_distance_from_playgolf(self, ctx: dict):
        # [보완] distance_screen after 루프 중복 방지
        try:
            self.distance_screen.stop()
        except Exception:
            pass

        self.distance_screen.set_context(
            parent_window=ctx["parent_window"],
            hole_row=ctx["hole_row"],
            gc_center_lat=ctx["gc_center_lat"],
            gc_center_lng=ctx["gc_center_lng"],
            cur_lat=ctx["cur_lat"],
            cur_lng=ctx["cur_lng"],
        )
        self.manager.show("distance")
        self.distance_screen.start()

    def _back_to_playgolf_from_distance(self):
        self.distance_screen.stop()
        self.manager.show("playgolf")

    def _open_green_view_from_distance(self, ctx: dict):
        self.distance_screen.stop()
        self.green_view_screen.set_context(
            parent_window=ctx["parent_window"],
            hole_row=ctx["hole_row"],
            gc_center_lat=ctx["gc_center_lat"],
            gc_center_lng=ctx["gc_center_lng"],
            cur_lat=ctx["cur_lat"],
            cur_lng=ctx["cur_lng"],
            selected_green=ctx["selected_green"],
        )
        self.manager.show("green")
        self.green_view_screen.start()

    def _back_to_distance_from_green(self):
        self.green_view_screen.stop()
        self.manager.show("distance")
        self.distance_screen.start()

    def _open_layup_view_from_distance(self, ctx: dict):
        self.distance_screen.stop()
        self.layup_view_screen.set_context(
            parent_window=ctx["parent_window"],
            hole_row=ctx["hole_row"],
            gc_center_lat=ctx["gc_center_lat"],
            gc_center_lng=ctx["gc_center_lng"],
            cur_lat=ctx["cur_lat"],
            cur_lng=ctx["cur_lng"],
            selected_green=ctx["selected_green"],
        )
        self.manager.show("layup")
        self.layup_view_screen.start()

    def _back_to_distance_from_layup(self):
        self.layup_view_screen.stop()
        self.manager.show("distance")
        self.distance_screen.start()

    def _open_putt_from_distance(self, ctx: dict):
        self.distance_screen.stop()
        self.putt_screen.set_context(
            parent_window=ctx["parent_window"],
            hole_row=ctx["hole_row"],
            gc_center_lat=ctx["gc_center_lat"],
            gc_center_lng=ctx["gc_center_lng"],
            cur_lat=ctx["cur_lat"],
            cur_lng=ctx["cur_lng"],
        )
        self.manager.show("putt")
        self.putt_screen.start()

    def _back_to_distance_from_putt(self):
        self.putt_screen.stop()
        self.manager.show("distance")
        self.distance_screen.start()

    def _open_scoring_from_putt(self, ctx: dict):
        self.putt_screen.stop()

        if app_settings.scorecard == "사용":
            # [보완] scoring 진입 전 타이머가 남아있을 수 있으므로 정리
            try:
                self.scoring_screen.stop()
            except Exception:
                pass

            self.scoring_screen.set_context(
                parent_window=ctx["parent_window"],
                hole_row=ctx["hole_row"],
            )
            self.manager.show("scoring")
            self.scoring_screen.start()
            return

        # scorecard == "안함"
        parent = ctx["parent_window"]
        try:
            if hasattr(parent, "notify_out_of_green_confirmed"):
                parent.notify_out_of_green_confirmed()
            else:
                parent.find_next_hole()
        except Exception as e:
            print("[ENTRY] notify_out_of_green_confirmed error:", e)

        self.manager.show("distance")
        self.distance_screen.start()

    def _on_scoring_done(self, result: dict):
        print("[ENTRY] scoring done:", result)

        # [보완] scoring 자동 OK 타이머(after) 취소 보장
        try:
            self.scoring_screen.stop()
        except Exception:
            pass

        # scorecard 사용 시 OK(또는 자동 OK)가 홀 종료 확정 트리거
        try:
            self.playgolf_screen.notify_out_of_green_confirmed()
        except Exception as e:
            print("[ENTRY] notify_out_of_green_confirmed error:", e)

        self.manager.show("distance")
        self.distance_screen.start()
