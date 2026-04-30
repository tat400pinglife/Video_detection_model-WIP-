import tkinter as tk
from tkinter import messagebox
import csv
import os
import threading
from pathlib import Path


# currently on video number 38 on desktop


# Selenium Imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

class TikTokCyborgApp:
    def __init__(self, root, csv_path):
        self.root = root
        self.root.title("TikTok Cyborg Control")
        
        # Keep GUI floating on top
        self.root.attributes('-topmost', True)
        self.root.geometry("420x180")
        self.root.configure(padx=20, pady=20)

        self.links = []
        self.current_idx = 0

        # Handle window closing gracefully
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # --- UI Elements ---
        self.counter_label = tk.Label(root, text="Starting Chrome...", font=("Arial", 12, "bold"))
        self.counter_label.pack(pady=(0, 10))

        self.link_display = tk.Entry(root, width=50, justify='center')
        self.link_display.pack(pady=(0, 15))

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        self.btn_copy = tk.Button(btn_frame, text="📋 Copy Link", 
                                  font=("Arial", 10), command=self.copy_current)
        self.btn_copy.grid(row=0, column=0, padx=10)

        self.btn_next = tk.Button(btn_frame, text="Next Video ➡️", bg="#008CBA", fg="white", 
                                  font=("Arial", 10, "bold"), command=self.next_link)
        self.btn_next.grid(row=0, column=1, padx=10)

        # 1. Load Data
        self.load_csv(csv_path)
        
        # 2. Launch the Puppet Browser
        self.setup_browser()

        # 3. Start the process
        self.update_ui()
        if self.links:
            self.open_current()

    def setup_browser(self):
        """Spins up a dedicated Chrome window that remembers your login."""
        chrome_options = Options()
        
        # Create a persistent profile folder next to the script
        profile_path = os.path.abspath("tiktok_cyborg_profile")
        chrome_options.add_argument(f"user-data-dir={profile_path}")
        
        # Remove the "Chrome is being controlled by automated software" banner
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)

    def load_csv(self, csv_path):
        if not Path(csv_path).exists():
            messagebox.showerror("Error", f"Could not find {csv_path}!")
            self.root.destroy()
            return

        with open(csv_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader, start=1):
                if not row or len(row) < 2:
                    continue

                link = row[0].strip()
                is_ai_flag = row[1].strip().lower().replace('"', '').replace("'", "")

                if row_idx == 1 and link.lower() in ["link", "url", "video"]: continue
                if not link.startswith("http"): link = "https://" + link

                if "y" in is_ai_flag:
                    self.links.append(link)

        if not self.links:
            messagebox.showwarning("Empty", "No AI links found in the CSV!")
            self.root.destroy()

    def update_ui(self):
        if self.current_idx >= len(self.links):
            self.counter_label.config(text="🎉 All Done!")
            self.link_display.delete(0, tk.END)
            self.btn_copy.config(state="disabled")
            self.btn_next.config(state="disabled")
            return

        current_url = self.links[self.current_idx]
        self.counter_label.config(text=f"Video {self.current_idx + 1} of {len(self.links)}")
        
        self.link_display.delete(0, tk.END)
        self.link_display.insert(0, current_url)

    def open_current(self):
        """Commands Selenium to load the URL in the exact same tab."""
        url = self.links[self.current_idx]
        
        # We use a background thread so the GUI doesn't freeze while the page loads
        def load_page():
            try:
                self.driver.get(url)
            except:
                pass # Ignore errors if user closed the browser manually
                
        threading.Thread(target=load_page, daemon=True).start()

    def copy_current(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.links[self.current_idx])
        self.btn_copy.config(text="✅ Copied!")
        self.root.after(1500, lambda: self.btn_copy.config(text="📋 Copy Link"))

    def next_link(self):
        self.current_idx += 1
        self.update_ui()
        if self.current_idx < len(self.links):
            self.open_current()

    def on_close(self):
        """Kills the Chrome browser when you close the GUI."""
        self.counter_label.config(text="Shutting down...")
        self.root.update()
        try:
            self.driver.quit()
        except:
            pass
        self.root.destroy()

if __name__ == "__main__":
    YOUR_CSV_FILE = "./scripts/youtube/csvs/youtube.csv"
    
    root = tk.Tk()
    app = TikTokCyborgApp(root, YOUR_CSV_FILE)
    root.mainloop()