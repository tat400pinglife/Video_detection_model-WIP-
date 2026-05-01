import tkinter as tk
import pyautogui
import pyperclip
import time
import webbrowser

# ==========================================
# 📍 ADJUSTABLE MOUSE COORDINATES
# ==========================================
# Update these X and Y values to match your monitor's resolution and browser layout.
# Tip: You can find your mouse coordinates by running `pyautogui.displayMousePosition()` in your terminal.

YT_TAB_X, YT_TAB_Y = 250, 20          # Coordinates to click the YouTube browser tab
DL_TAB_X, DL_TAB_Y = 450, 20          # Coordinates to click the Download Site browser tab
ADDRESS_BAR_X, ADDRESS_BAR_Y = 500, 60 # Coordinates to click the browser's URL address bar

PASTE_BOX_X, PASTE_BOX_Y = 500, 400   # Coordinates to click the URL input box on the download site
DOWNLOAD_BTN_X, DOWNLOAD_BTN_Y = 600, 450 # Coordinates to click the "Download" button on the site
# ==========================================

class DownloaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YT Auto-Clicker")
        self.root.geometry("250x180")
        
        # Keeps the control panel on top of your browser
        self.root.attributes("-topmost", True) 

        self.sites_opened = False

        # UI Buttons
        self.btn_setup = tk.Button(root, text="1. Open Sites", command=self.open_sites, bg="#e0e0e0")
        self.btn_setup.pack(pady=10, fill="x", padx=20)

        self.btn_download = tk.Button(root, text="2. Download Video", command=self.download_action, bg="#a8e6cf")
        self.btn_download.pack(pady=10, fill="x", padx=20)

        self.btn_next = tk.Button(root, text="3. Next", command=self.next_action, bg="#ffdfba")
        self.btn_next.pack(pady=10, fill="x", padx=20)

    def open_sites(self):
        """Opens YouTube and the download site in adjacent tabs."""
        if not self.sites_opened:
            webbrowser.open("https://www.youtube.com")
            time.sleep(1.5) # Give the browser a moment to open
            webbrowser.open_new_tab("https://app.ytdown.to/en27/")
            self.sites_opened = True
            print("Sites opened. Please ensure YouTube is the first tab and the download site is the second.")

    def download_action(self):
        """Copies the YT URL, switches tabs, pastes, and clicks download."""
        
        # 1. Ensure we are on the YouTube tab
        pyautogui.click(YT_TAB_X, YT_TAB_Y)
        time.sleep(0.5)

        # 2. Click address bar, copy URL
        pyautogui.click(ADDRESS_BAR_X, ADDRESS_BAR_Y)
        time.sleep(0.2)
        pyautogui.hotkey('ctrl', 'a') # Select all
        time.sleep(0.1)
        pyautogui.hotkey('ctrl', 'c') # Copy to clipboard
        time.sleep(0.3)

        # 3. Switch to the Download site tab
        pyautogui.click(DL_TAB_X, DL_TAB_Y)
        time.sleep(0.5)

        # 4. Click the input field and paste the URL
        pyautogui.click(PASTE_BOX_X, PASTE_BOX_Y)
        time.sleep(0.2)
        pyautogui.hotkey('ctrl', 'a') # Select existing text if any
        time.sleep(0.1)
        pyautogui.hotkey('ctrl', 'v') # Paste
        time.sleep(0.3)

        # 5. Click the Download button
        pyautogui.click(DOWNLOAD_BTN_X, DOWNLOAD_BTN_Y)
        print("Download initiated!")

    def next_action(self):
        """Switches back to YouTube so you can find the next video."""
        pyautogui.click(YT_TAB_X, YT_TAB_Y)
        print("Switched back to YouTube. Ready for the next one.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DownloaderApp(root)
    root.mainloop()