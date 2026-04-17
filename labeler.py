import tkinter as tk
from tkinter import filedialog, messagebox
import csv
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager # Optional but helpful

class ShortsLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Shorts Labeler")
        self.root.geometry("300x150")
        self.root.attributes('-topmost', True) 

        self.data = []
        self.current_idx = 0
        self.file_path = ""
        self.driver = None # This will hold our browser window

        # --- UI Elements ---
        self.lbl_status = tk.Label(root, text="Please load a CSV file.")
        self.lbl_status.pack(pady=10)

        self.btn_load = tk.Button(root, text="Load CSV", command=self.load_csv)
        self.btn_load.pack(pady=5)

        self.frame_btns = tk.Frame(root)
        self.btn_y = tk.Button(self.frame_btns, text="YES (y)", width=10, bg="lightgreen", 
                               command=lambda: self.record_answer('y'), state=tk.DISABLED)
        self.btn_y.pack(side=tk.LEFT, padx=10)

        self.btn_n = tk.Button(self.frame_btns, text="NO (n)", width=10, bg="lightcoral", 
                               command=lambda: self.record_answer('n'), state=tk.DISABLED)
        self.btn_n.pack(side=tk.RIGHT, padx=10)
        
        self.frame_btns.pack(pady=10)

        # --- Keyboard Shortcuts ---
        self.root.bind('<y>', lambda event: self.record_answer('y') if self.btn_y['state'] == tk.NORMAL else None)
        self.root.bind('<n>', lambda event: self.record_answer('n') if self.btn_n['state'] == tk.NORMAL else None)
        
        # Ensure the browser closes if you close the Python window
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_csv(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not self.file_path: 
            return

        with open(self.file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            self.data = list(reader)

        # Launch the single Chrome window
        try:
            if not self.driver:
                self.driver = webdriver.Chrome()
        except Exception as e:
            messagebox.showerror("Error", f"Could not launch Chrome. Make sure it is installed.\n\nDetails: {e}")
            return

        self.current_idx = 0
        self.btn_load.config(state=tk.DISABLED)
        self.btn_y.config(state=tk.NORMAL)
        self.btn_n.config(state=tk.NORMAL)

        self.show_current_video()

    def show_current_video(self):
        # Skip rows that already have a 'y' or 'n'
        while self.current_idx < len(self.data):
            row = self.data[self.current_idx]
            if len(row) > 1 and row[1].strip().lower() in ['y', 'n']:
                self.current_idx += 1
            else:
                break

        if self.current_idx < len(self.data):
            url = self.data[self.current_idx][0]
            self.lbl_status.config(text=f"Reviewing Video {self.current_idx + 1} of {len(self.data)}")
            
            try:
                # Navigates the EXISTING tab to the new URL
                self.driver.get(url)
            except WebDriverException:
                # Catches the error if you manually closed the Chrome window
                messagebox.showerror("Browser Closed", "The browser window was closed. Please restart the script.")
                self.root.destroy()
        else:
            self.lbl_status.config(text="All done!")
            self.btn_y.config(state=tk.DISABLED)
            self.btn_n.config(state=tk.DISABLED)
            if self.driver:
                self.driver.quit() # Automatically close browser when finished
            messagebox.showinfo("Finished", "All videos labeled. Your CSV is fully updated.")

    def record_answer(self, answer):
        while len(self.data[self.current_idx]) < 2:
            self.data[self.current_idx].append("")

        self.data[self.current_idx][1] = answer
        self.save_csv()

        self.current_idx += 1
        self.show_current_video()

    def save_csv(self):
        with open(self.file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(self.data)

    def on_closing(self):
        if self.driver:
            self.driver.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ShortsLabeler(root)
    root.mainloop()