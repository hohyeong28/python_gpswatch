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
from scorecard import ScorecardScreen  # [추가]


class ScreenManager(tk.Frame):
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


class MenuScreen(BaseScreen):
    def __init__(
        self,
        manager: ScreenManager,
        asset_dir: Path,
        on_play: Callable[[], None],
        on_open_settings: Callable[[], None],
    ):
        super().__init__(manager, asset_dir)

        self.on_play = on_play
        self.on_open_settings = on_open_settings

        self.canvas = tk.Canvas(self, width=LCD_WIDTH, height=LCD_HEIGHT, highlightthickness=0, bg="black")
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
        play_region = self.canvas.create_rectangle(0, 0, LCD_WIDTH, LCD_HEIGHT // 2, outline="", fill="")
        self.canvas.tag_bind(play_region, "<Button-1>", lambda e: self.on_play())

        # Setting
        setting_region = self.canvas.create_rectangle(0, LCD_HEIGHT // 2, LCD_WIDTH // 2, LCD_HEIGHT, outline="", fill="")
        self.canvas.tag_bind(setting_region, "<Button-1>", lambda e: self.on_open_settings())

        # History
        history_region = self.canvas.create_rectangle(LCD_WIDTH // 2, LCD_HEIGHT // 2, LCD_WIDTH, LCD_HEIGHT, outline="", fill="")
        self.canvas.tag_bind(history_region, "<Button-1>", lambda e: self.manager.show("history"))


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


class GolfWatchApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Gps Golf Watch")
        self.root.resizable(False, False)

        self.asset_dir = Path(__file__).parent / "assets_image"
        self.manager = ScreenManager(root)

        self._settings_return_to: str = "menu"

        self.playgolf_screen = playgolf.PlayGolfFrame(
            self.manager,
            on_back=self._back_from_playgolf,
            on_open_distance=self._open_distance_from_playgolf,
        )

        self.menu_screen = MenuScreen(
            self.manager,
            self.asset_dir,
            on_play=self._go_playgolf,
            on_open_settings=self._open_settings_from_menu,
        )

        self.settings_screen = setting.SettingsFrame(
            self.manager,
            settings=app_settings,
            on_back=self._back_from_settings,
        )

        self.distance_screen = DistanceScreen(
            self.manager,
            on_back=self._back_to_playgolf_from_distance,
            on_open_green_view=self._open_green_view_from_distance,
            on_open_layup_view=self._open_layup_view_from_distance,
            on_open_putt=self._open_putt_from_distance,
            on_open_settings=self._open_settings_from_distance,
            on_open_scorecard=self._open_scorecard_from_distance,  # [추가]
        )

        self.green_view_screen = GreenViewScreen(self.manager, on_back=self._back_to_distance_from_green)
        self.layup_view_screen = LayupViewScreen(self.manager, on_back=self._back_to_distance_from_layup)

        self.putt_screen = PuttDistanceScreen(
            self.manager,
            on_back=self._back_to_distance_from_putt,
            on_open_scoring=self._open_scoring_from_putt,
        )

        self.scoring_screen = ScoringScreen(
            self.manager,
            on_back=self._back_from_scoring,
            on_done=self._on_scoring_done,
        )

        # scorecard
        self.scorecard_screen = ScorecardScreen(
            self.manager,
            on_back=self._back_to_distance_from_scorecard,
        )

        self.history_screen = HistoryScreen(self.manager, self.asset_dir)

        self.manager.add("menu", self.menu_screen)
        self.manager.add("settings", self.settings_screen)
        self.manager.add("playgolf", self.playgolf_screen)
        self.manager.add("distance", self.distance_screen)
        self.manager.add("green", self.green_view_screen)
        self.manager.add("layup", self.layup_view_screen)
        self.manager.add("putt", self.putt_screen)
        self.manager.add("scoring", self.scoring_screen)
        self.manager.add("scorecard", self.scorecard_screen)
        self.manager.add("history", self.history_screen)

        self.manager.show("menu")

    # ---- Settings ----

    def _open_settings_from_menu(self):
        self._settings_return_to = "menu"
        try:
            self.settings_screen.start()
        except Exception:
            pass
        self.manager.show("settings")

    def _open_settings_from_distance(self):
        self._settings_return_to = "distance"
        try:
            self.distance_screen.stop()
        except Exception:
            pass
        try:
            self.settings_screen.start()
        except Exception:
            pass
        self.manager.show("settings")

    def _back_from_settings(self):
        if self._settings_return_to == "distance":
            self.manager.show("distance")
            self.distance_screen.start()
        else:
            self.manager.show("menu")

    # ---- Scorecard ----

    def _open_scorecard_from_distance(self):
        try:
            self.distance_screen.stop()
        except Exception:
            pass

        try:
            self.scorecard_screen.set_context(parent_window=self.playgolf_screen)
        except Exception:
            pass

        self.manager.show("scorecard")
        self.scorecard_screen.start()

    def _back_to_distance_from_scorecard(self):
        try:
            self.scorecard_screen.stop()
        except Exception:
            pass
        self.manager.show("distance")
        self.distance_screen.start()

    # ---- PlayGolf ----

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

    # ---- Transitions ----

    def _open_distance_from_playgolf(self, ctx: dict):
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
            try:
                self.scoring_screen.stop()
            except Exception:
                pass
            self.scoring_screen.set_context(parent_window=ctx["parent_window"], hole_row=ctx["hole_row"])
            self.manager.show("scoring")
            self.scoring_screen.start()
            return

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

    def _back_from_scoring(self):
        try:
            self.scoring_screen.stop()
        except Exception:
            pass
        self.manager.show("menu")

    def _on_scoring_done(self, result: dict):
        print("[ENTRY] scoring done:", result)

        # [추가] 홀별 스코어 저장 (Scorecard에서 사용)
        try:
            hole_no = int(result.get("Hole", 0))
            if hole_no > 0:
                if not hasattr(self.playgolf_screen, "round_scores") or not isinstance(
                        self.playgolf_screen.round_scores, dict):
                    self.playgolf_screen.round_scores = {}
                # 같은 홀 재입력 시 덮어쓰기(최신값 유지)
                self.playgolf_screen.round_scores[hole_no] = result
        except Exception as e:
            print("[ENTRY] round_scores save error:", e)

        # 기존 동작 유지
        try:
            self.scoring_screen.stop()
        except Exception:
            pass
        try:
            self.playgolf_screen.notify_out_of_green_confirmed()
        except Exception as e:
            print("[ENTRY] notify_out_of_green_confirmed error:", e)

        self.manager.show("distance")
        self.distance_screen.start()

