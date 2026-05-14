"""
Mouse controller con SendInput nativo de Windows (Win32 API).
Latencia de click: ~1ms (vs ~30ms de pynput).

Para sistemas no-Windows, fallback a pynput automatico.
"""
import sys
import ctypes
from ctypes import wintypes


# ════════════════════════════════════════════════════════════════════
# IMPLEMENTACION NATIVA WIN32 SendInput
# ════════════════════════════════════════════════════════════════════

_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    # ── Constantes de Win32 ────────────────────────────────────────
    INPUT_MOUSE          = 0
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP   = 0x0004

    # ── Estructuras para SendInput ─────────────────────────────────
    PUL = ctypes.POINTER(ctypes.c_ulong)

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx",          ctypes.c_long),
            ("dy",          ctypes.c_long),
            ("mouseData",   ctypes.c_ulong),
            ("dwFlags",     ctypes.c_ulong),
            ("time",        ctypes.c_ulong),
            ("dwExtraInfo", PUL),
        ]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk",         ctypes.c_ushort),
            ("wScan",       ctypes.c_ushort),
            ("dwFlags",     ctypes.c_ulong),
            ("time",        ctypes.c_ulong),
            ("dwExtraInfo", PUL),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg",    ctypes.c_ulong),
            ("wParamL", ctypes.c_short),
            ("wParamH", ctypes.c_ushort),
        ]

    class INPUT_UNION(ctypes.Union):
        _fields_ = [
            ("mi", MOUSEINPUT),
            ("ki", KEYBDINPUT),
            ("hi", HARDWAREINPUT),
        ]

    class INPUT(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_ulong),
            ("u",    INPUT_UNION),
        ]

    _user32 = ctypes.windll.user32
    _SendInput      = _user32.SendInput
    _SetCursorPos   = _user32.SetCursorPos

    # Reusable objects — evita allocar memoria cada click
    _extra = ctypes.c_ulong(0)

    def _build_mouse_input(flags):
        """Construye una estructura INPUT para mouse."""
        ii = INPUT_UNION()
        ii.mi = MOUSEINPUT(0, 0, 0, flags, 0, ctypes.pointer(_extra))
        return INPUT(INPUT_MOUSE, ii)

    _CLICK_DOWN = _build_mouse_input(MOUSEEVENTF_LEFTDOWN)
    _CLICK_UP   = _build_mouse_input(MOUSEEVENTF_LEFTUP)
    _click_array = (INPUT * 2)(_CLICK_DOWN, _CLICK_UP)


    def _click_native(screen_x, screen_y):
        """
        Mueve cursor y dispara click usando SendInput.
        Latencia tipica: <2ms total (mover + down + up).
        """
        _SetCursorPos(int(screen_x), int(screen_y))
        # Envia los 2 eventos (down + up) en una sola syscall
        _SendInput(2, ctypes.byref(_click_array), ctypes.sizeof(INPUT))


# ════════════════════════════════════════════════════════════════════
# FALLBACK pynput (no-Windows)
# ════════════════════════════════════════════════════════════════════

if not _IS_WINDOWS:
    from pynput.mouse import Controller, Button
    _mouse = Controller()

    def _click_native(screen_x, screen_y):
        _mouse.position = (int(screen_x), int(screen_y))
        _mouse.click(Button.left)


# ════════════════════════════════════════════════════════════════════
# API publica
# ════════════════════════════════════════════════════════════════════

def move_and_click(x, y, region):
    """
    Click rapido en (x, y) relativos a la region del juego.
    Coordenadas absolutas calculadas a partir de region.left/top.

    Para minimizar la latencia, hace UN solo click (no spray).
    Con SendInput el click llega al sistema en <2ms en lugar de 30ms.
    Eso reduce el lag total del pipeline en ~28ms, dando ~5-6 px menos
    de error por hit a velocidades tipicas de pato (150-200 px/s).
    """
    # Clamp a los limites del area del juego
    x = max(0, min(int(x), region["width"]  - 1))
    y = max(0, min(int(y), region["height"] - 1))

    screen_x = region["left"] + x
    screen_y = region["top"]  + y

    _click_native(screen_x, screen_y)
