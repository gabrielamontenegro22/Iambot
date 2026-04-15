import pyautogui


def move_and_click(x, y, region):
    screen_x = region["left"] + x
    screen_y = region["top"] + y

    pyautogui.moveTo(screen_x, screen_y, duration=0.1)
    pyautogui.click()