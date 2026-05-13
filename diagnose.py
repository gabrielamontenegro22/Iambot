"""
Script de diagnóstico de rendimiento.

Mide y reporta:
  - Servidor de display detectado (Wayland vs X11)  ← CRÍTICO en Pi 5
  - Tiempo de UNA captura mss
  - Tiempo de UNA inferencia YOLO
  - Throughput sostenido en bucle (capture+YOLO simulado)
  - Memoria libre, temperatura CPU, throttling

Ejecutar ANTES de reportar problemas de rendimiento:
    python3 diagnose.py
"""

import os
import sys
import time
import subprocess


def section(title):
    print()
    print("═" * 60)
    print(f"  {title}")
    print("═" * 60)


def check_display_server():
    section("1. SERVIDOR DE DISPLAY")
    session = os.environ.get("XDG_SESSION_TYPE", "?")
    wayland_display = os.environ.get("WAYLAND_DISPLAY", "")
    display = os.environ.get("DISPLAY", "")

    print(f"  XDG_SESSION_TYPE = {session}")
    print(f"  WAYLAND_DISPLAY  = '{wayland_display}'")
    print(f"  DISPLAY          = '{display}'")

    if session == "wayland" or wayland_display:
        print()
        print("  ⚠️  ESTÁS EN WAYLAND.")
        print("  En Raspberry Pi 5 esto causa latencia ALTA de captura de pantalla")
        print("  (mss debe pasar por protocolos de seguridad de Wayland).")
        print()
        print("  SOLUCIÓN RECOMENDADA: cambia a X11.")
        print("    1. sudo raspi-config")
        print("    2. Advanced Options → Wayland → seleccionar X11")
        print("    3. Reiniciar")
        print()
        print("  Esto típicamente reduce el lag de captura de >500ms a ~30ms en Pi 5.")
        return "wayland"
    elif display:
        print("  ✓ X11 detectado — bien para captura rápida con mss.")
        return "x11"
    else:
        print("  ⚠️  Ningún servidor de display visible.")
        return "none"


def check_throttling():
    section("2. THROTTLING / TEMPERATURA")
    try:
        out = subprocess.check_output(
            ["vcgencmd", "get_throttled"], timeout=2
        ).decode().strip()
        print(f"  {out}")
        # Parsear: throttled=0x... — 0 = OK, !=0 = ha habido throttling
        if "throttled=0x0" in out:
            print("  ✓ Sin throttling.")
        else:
            print("  ⚠️  HUBO THROTTLING. Posibles causas:")
            print("     - Fuente de alimentación insuficiente (<5V/5A en Pi 5)")
            print("     - Sobrecalentamiento (instala disipador/ventilador)")
    except Exception as e:
        print(f"  No se pudo leer vcgencmd: {e}")

    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_temp"], timeout=2
        ).decode().strip()
        print(f"  {out}")
    except Exception:
        pass


def check_memory():
    section("3. MEMORIA")
    try:
        out = subprocess.check_output(["free", "-h"], timeout=2).decode()
        for line in out.splitlines():
            if line.startswith("Mem:") or line.startswith("Swap:"):
                print(f"  {line}")
    except Exception as e:
        print(f"  Error: {e}")


def benchmark_mss_capture():
    section("4. LATENCIA DE CAPTURA (mss)")
    try:
        import mss
        import numpy as np
        import cv2
    except ImportError as e:
        print(f"  Falta dependencia: {e}")
        return

    # Captura una región pequeña del centro (320x320) como hace el bot
    print("  Capturando 50 frames de 320x320...")
    times = []
    with mss.mss() as sct:
        # Tomar un monitor cualquiera
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        # Recortar a 320x320 del centro
        cx = monitor["left"] + monitor["width"] // 2
        cy = monitor["top"] + monitor["height"] // 2
        region = {"left": cx - 160, "top": cy - 160, "width": 320, "height": 320}

        # Warmup
        for _ in range(3):
            sct.grab(region)

        for _ in range(50):
            t0 = time.time()
            shot = sct.grab(region)
            img = np.asarray(shot, dtype=np.uint8)
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            times.append(time.time() - t0)

    avg = sum(times) / len(times) * 1000
    mn  = min(times) * 1000
    mx  = max(times) * 1000
    p95 = sorted(times)[int(len(times) * 0.95)] * 1000

    print(f"  Promedio: {avg:.1f} ms")
    print(f"  Mínimo:   {mn:.1f} ms")
    print(f"  P95:      {p95:.1f} ms")
    print(f"  Máximo:   {mx:.1f} ms")

    if avg > 100:
        print()
        print("  ⚠️  Captura LENTA. Causas probables:")
        print("     - Estás en Wayland (ver sección 1)")
        print("     - El compositor está bajo carga (cierra apps innecesarias)")
        print("     - El navegador con el juego está usando la GPU intensivamente")
    elif avg > 30:
        print("  ⚠️  Captura aceptable pero mejorable. Cierra apps en segundo plano.")
    else:
        print("  ✓ Captura rápida.")


def benchmark_yolo():
    section("5. LATENCIA DE INFERENCIA YOLO")
    try:
        from ultralytics import YOLO
        import numpy as np
    except ImportError as e:
        print(f"  Falta dependencia: {e}")
        return

    # Buscar modelo
    candidates = [
        "models/duck_ncnn_model",
        "models/duck.pt",
        "yolov8n.pt",
    ]
    model_path = None
    for c in candidates:
        if os.path.exists(c):
            model_path = c
            break

    if model_path is None:
        print("  No se encontró ningún modelo. Saltando benchmark YOLO.")
        return

    print(f"  Modelo: {model_path}")
    is_ncnn = os.path.isdir(model_path)
    print(f"  Formato: {'NCNN ✓' if is_ncnn else 'PyTorch (⚠️ exporta a NCNN para mejor rendimiento)'}")

    if is_ncnn:
        model = YOLO(model_path, task='detect')
    else:
        model = YOLO(model_path)

    dummy = np.zeros((320, 320, 3), dtype=np.uint8)

    # Warmup
    print("  Warmup...")
    for _ in range(3):
        model(dummy, imgsz=320, verbose=False)

    print("  Midiendo 30 inferencias...")
    times = []
    for _ in range(30):
        t0 = time.time()
        model(dummy, imgsz=320, verbose=False)
        times.append(time.time() - t0)

    avg = sum(times) / len(times) * 1000
    mn  = min(times) * 1000
    print(f"  Promedio: {avg:.1f} ms ({1000/avg:.1f} FPS sostenidos)")
    print(f"  Mínimo:   {mn:.1f} ms")

    if avg > 150:
        print()
        print("  ⚠️  YOLO LENTO. Soluciones:")
        if not is_ncnn:
            print("     - Exporta a NCNN:  yolo export model=models/duck.pt format=ncnn")
            print("       (típicamente 2-4x más rápido en ARM)")
        print("     - Verifica que no hay throttling (sección 2)")
    else:
        print("  ✓ YOLO rápido.")


def check_processes():
    section("6. PROCESOS QUE PUEDEN COMPETIR POR CPU/GPU")
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,pcpu,pmem,comm", "--sort=-pcpu"],
            timeout=2
        ).decode()
        lines = out.splitlines()
        # Header + top 5
        print("  " + lines[0])
        for line in lines[1:8]:
            print("  " + line)

        # Buscar navegadores
        for line in lines[1:]:
            if any(b in line.lower() for b in ["chromium", "firefox", "chrome"]):
                print()
                print(f"  ℹ️  Navegador activo (el juego):")
                print(f"     {line}")
                print("     Está OK — esto es esperado.")
                break
    except Exception as e:
        print(f"  Error: {e}")


def main():
    print()
    print("┌" + "─" * 58 + "┐")
    print("│  DIAGNÓSTICO DE RENDIMIENTO DEL BOT — RASPBERRY PI       │")
    print("└" + "─" * 58 + "┘")

    server = check_display_server()
    check_throttling()
    check_memory()
    benchmark_mss_capture()
    benchmark_yolo()
    check_processes()

    section("RESUMEN")
    if server == "wayland":
        print("  ⚠️  PRIORIDAD #1: cambia a X11. Es muy probable que sea la")
        print("                   causa principal de tu lag.")
    print()
    print("  Compara los tiempos con los esperados en Pi 5 + X11 + NCNN:")
    print("     - Captura: ~10-30 ms")
    print("     - YOLO:    ~50-80 ms")
    print()
    print("  Si tus valores son MUCHO mayores, hay un cuello de botella")
    print("  específico que el diagnóstico de arriba debe haber revelado.")
    print()


if __name__ == "__main__":
    main()