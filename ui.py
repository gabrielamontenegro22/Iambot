import tkinter as tk
from tkinter import messagebox
import subprocess
import config

from window_capture import get_window_geometry, select_game_zone


def get_windows():
    try:
        output = subprocess.check_output(["wmctrl", "-l"]).decode()
    except Exception:
        return []
    windows = []
    for line in output.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4:
            windows.append(parts[3])
    return windows


class App:

    def __init__(self, root):
        self.root      = root
        self._should_start = False   # señal para app.py

        self.root.title("Vision Bot")
        self.root.resizable(False, False)

        # ── 1. Ventana ────────────────────────────────────────────────
        tk.Label(root, text="1. Selecciona la ventana del juego",
                 font=("Arial", 10, "bold")).pack(pady=(12, 2))

        self.listbox = tk.Listbox(root, width=62, height=10)
        self.listbox.pack(padx=10)

        tk.Button(root, text="Actualizar lista",
                  command=self.load_windows).pack(pady=4)

        self.load_windows()

        # ── 2. Área de juego ──────────────────────────────────────────
        tk.Label(root, text="2. Define el área de juego",
                 font=("Arial", 10, "bold")).pack(pady=(10, 2))

        self.zone_label = tk.Label(
            root,
            text="Sin área definida — se usará la ventana completa",
            fg="gray"
        )
        self.zone_label.pack()

        tk.Button(
            root,
            text="Seleccionar área de juego  (dibuja el rectángulo)",
            bg="#1976D2", fg="white",
            command=self.select_zone
        ).pack(pady=6, ipadx=6, ipady=4)

        # ── 3. Iniciar ────────────────────────────────────────────────
        tk.Label(root, text="3. Inicia el bot",
                 font=("Arial", 10, "bold")).pack(pady=(10, 2))

        tk.Button(
            root,
            text="▶  INICIAR",
            bg="#388E3C", fg="white",
            font=("Arial", 11, "bold"),
            command=self.start
        ).pack(pady=(4, 14), ipadx=20, ipady=6)

    # ─────────────────────────────────────────────────────────────────
    def load_windows(self):
        self.listbox.delete(0, tk.END)
        for w in get_windows():
            self.listbox.insert(tk.END, w)

    # ─────────────────────────────────────────────────────────────────
    def select_zone(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "Primero selecciona la ventana en la lista")
            return

        config.WINDOW_NAME = self.listbox.get(selection[0])

        region = get_window_geometry(config.WINDOW_NAME)
        if not region:
            messagebox.showerror(
                "Error",
                f"No se encontró la ventana: {config.WINDOW_NAME}\n"
                "Asegúrate de que esté abierta y visible."
            )
            return

        # Minimizar para no tapar el juego
        self.root.iconify()
        self.root.update()

        zone = select_game_zone(region)

        self.root.deiconify()
        self.root.lift()

        if zone is None:
            self.zone_label.config(
                text="Selección cancelada — se usará la ventana completa",
                fg="orange"
            )
            config.GAME_ZONE = None
            return

        config.GAME_ZONE = zone
        off_x, off_y, gw, gh = zone
        self.zone_label.config(
            text=f"Área: offset ({off_x}, {off_y})  tamaño {gw}×{gh} px  ✓",
            fg="#2E7D32"
        )

    # ─────────────────────────────────────────────────────────────────
    def start(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "Selecciona una ventana")
            return

        config.WINDOW_NAME = self.listbox.get(selection[0])

        if not hasattr(config, "GAME_ZONE"):
            config.GAME_ZONE = None

        # Señalar a run_ui() que debe iniciar el bot y cerrar Tkinter
        self._should_start = True
        self.root.destroy()   # libera el main thread — cv2 lo necesita


def run_ui():
    """
    Abre la UI de configuración y devuelve True si el usuario
    hizo clic en INICIAR, False si cerró la ventana sin iniciar.

    Después de devolver True, app.py llama run_bot() en el main thread,
    donde cv2.imshow y cv2.waitKey funcionan correctamente.
    """
    root = tk.Tk()
    app  = App(root)
    root.mainloop()          # bloquea hasta root.destroy()
    return app._should_start