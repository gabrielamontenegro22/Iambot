from ui import run_ui
from main import run_bot

if __name__ == "__main__":
    # run_ui() bloquea hasta que el usuario cierra la ventana.
    # Devuelve True solo si hizo clic en INICIAR.
    # run_bot() corre en el main thread para que cv2.imshow funcione.
    if run_ui():
        run_bot()