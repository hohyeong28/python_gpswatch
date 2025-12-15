# run_app.py (개발/테스트용 런처)
import tkinter as tk
from entry_menu import GolfWatchApp, LCD_WIDTH, LCD_HEIGHT

def main():
    root = tk.Tk()
    root.geometry(f"{LCD_WIDTH}x{LCD_HEIGHT}")
    root.title("GPS Golf Watch")
    app = GolfWatchApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
