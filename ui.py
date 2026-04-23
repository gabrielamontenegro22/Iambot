import tkinter as tk
from tkinter import messagebox
import subprocess
import threading
import config
from main import run_bot  # vamos a crear esta función

def get_windows():
    output = subprocess.check_output(["wmctrl", "-l"]).decode()
    windows = []

    for line in output.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4:
            windows.append(parts[3])

    return windows


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Selector de ventana")

        self.selected_window = tk.StringVar()

        tk.Label(root, text="Selecciona una ventana:").pack(pady=10)

        self.listbox = tk.Listbox(root, width=50, height=10)
        self.listbox.pack()

        self.load_windows()

        tk.Button(root, text="Actualizar", command=self.load_windows).pack(pady=5)
        tk.Button(root, text="CAPTURAR", command=self.start_capture).pack(pady=10)

    def load_windows(self):
        self.listbox.delete(0, tk.END)
        for w in get_windows():
            self.listbox.insert(tk.END, w)

    def start_capture(self):
        selection = self.listbox.curselection()

        if not selection:
            messagebox.showerror("Error", "Selecciona una ventana")
            return

        window_name = self.listbox.get(selection[0])

        config.WINDOW_NAME = window_name

        # Ejecutar el bot en otro hilo
        thread = threading.Thread(target=run_bot)
        thread.daemon = True
        thread.start()


def run_ui():
    root = tk.Tk()
    app = App(root)
    root.mainloop()