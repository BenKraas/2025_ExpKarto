import os
# --- Fix locale issues for pygame keyboard layout ---
os.environ["SDL_KEYBOARD_ALLOW_SYSTEM_LAYOUT"] = "1"

from PyQt5 import QtWidgets, QtGui, QtCore, QtSvg
import sys
import csv
import random
import math
import time
import pygame
import io
from PIL import Image

from scipy.optimize import linear_sum_assignment
import logging
import threading
import ctypes
import sys
import pygame
import numpy as np

# --- Logging setup ---
def setup_logging(log_path, file_level=logging.DEBUG, console_level=logging.INFO):
    # Clear the log file on every run
    open(log_path, "w").close()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # File handler (all logs)
    fh = logging.FileHandler(log_path)
    fh.setLevel(file_level)
    fh_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(fh_formatter)
    # Console handler (above info)
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(ch_formatter)
    # Remove existing handlers
    logger.handlers = []
    logger.addHandler(fh)
    logger.addHandler(ch)
    logging.debug("Logging initialized: file=%s, file_level=%s, console_level=%s", log_path, file_level, console_level)

def setup_experiment_logging():
    """
    Set up logging for the experiment session before any experiments run.
    """
    log_path = os.path.join(os.path.dirname(__file__), "experiment.log")
    setup_logging(log_path, file_level=logging.INFO, console_level=logging.INFO)
    logging.info("Logging initialized for experiment session.")

def scatter_icons_around_center(center_x, center_y, distances, n_items=5):
    """
    Returns a list of (x, y, angle, distance) for each icon.
    """
    coords = []
    for i in range(n_items):
        dist = distances[i % len(distances)]
        angle = random.uniform(0, 2 * math.pi)
        x = center_x + dist * math.cos(angle)
        y = center_y + dist * math.sin(angle)
        coords.append((int(x), int(y), angle, dist))
    return coords

def load_icon_surfaces(icon_size_px):
    icons_dir = os.path.join(os.path.dirname(__file__), "res", "icons")
    icon_files = [f for f in os.listdir(icons_dir) if f.lower().endswith('.png')]
    assert icon_files, "No PNG icons found in /res/icons (use PNG for pygame)"
    surfaces = []
    for fname in icon_files:
        img = pygame.image.load(os.path.join(icons_dir, fname)).convert_alpha()
        img = pygame.transform.smoothscale(img, (icon_size_px, icon_size_px))
        surfaces.append(img)
    return surfaces

def display_image_fullscreen(image_path, image_height_px):
    img = pygame.image.load(image_path).convert_alpha()
    img = pygame.transform.smoothscale(img, (image_height_px, image_height_px))
    return img

def ensure_svg_icons_converted_to_png(rebuild_svg=False, png_res=128):
    """
    Convert all SVGs in the icons folder to PNGs (same basename).
    If rebuild_svg is True, always rebuild PNGs. Otherwise, skip if PNG exists.
    png_res: output PNG size (width and height in px).
    """
    try:
        import cairosvg
    except Exception as e:
        cairosvg = None
        logging.warning("cairosvg not available, cannot convert SVG to PNG. Install cairosvg to enable this feature.")
        return

    icons_dir = os.path.join(os.path.dirname(__file__), "res", "icons")
    for fname in os.listdir(icons_dir):
        if fname.lower().endswith('.svg'):
            svg_path = os.path.join(icons_dir, fname)
            png_name = os.path.splitext(fname)[0] + ".png"
            png_path = os.path.join(icons_dir, png_name)
            if rebuild_svg or not os.path.exists(png_path):
                if cairosvg is None:
                    logging.error("cairosvg is required for SVG to PNG conversion. Please install it.")
                    print("cairosvg is required for SVG to PNG conversion. Please install it.")
                    sys.exit(1)
                logging.info(f"Converting {fname} to {png_name}...")
                png_bytes = cairosvg.svg2png(url=svg_path, output_width=png_res, output_height=png_res)
                pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
                pil_img.save(png_path)
    return

def optimal_icon_guess_assignment(icon_points, guesses):
    """
    Assigns each guess to an icon (one-to-one, minimal total distance) using Hungarian algorithm.
    Returns a list of guess indices for each icon (None if not enough guesses).
    """
    if not guesses or not icon_points:
        return [None] * len(icon_points)
    if linear_sum_assignment is None:
        # Fallback: greedy assignment (not optimal)
        icon_to_guess = [None] * len(icon_points)
        guess_used = set()
        for i, (ix, iy, _) in enumerate(icon_points):
            min_dist = float('inf')
            min_j = None
            for j, (gx, gy) in enumerate(guesses):
                if j in guess_used:
                    continue
                dist = math.hypot(ix - gx, iy - gy)
                if dist < min_dist:
                    min_dist = dist
                    min_j = j
            if min_j is not None:
                icon_to_guess[i] = min_j
                guess_used.add(min_j)
        return icon_to_guess
    # Build cost matrix
    cost_matrix = []
    for (ix, iy, _) in icon_points:
        row = []
        for (gx, gy) in guesses:
            row.append(math.hypot(ix - gx, iy - gy))
        cost_matrix.append(row)
    # Pad cost matrix if needed
    n = max(len(icon_points), len(guesses))
    for row in cost_matrix:
        row.extend([1e6] * (n - len(row)))
    while len(cost_matrix) < n:
        cost_matrix.append([1e6] * n)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    icon_to_guess = [None] * len(icon_points)
    for i, j in zip(row_ind, col_ind):
        if i < len(icon_points) and j < len(guesses) and cost_matrix[i][j] < 1e5:
            icon_to_guess[i] = j
    return icon_to_guess

def draw_debug_overlay(exp_result, screen, icon_size_px, font, center_x, center_y):
    """
    Draw debug overlay: icons, guesses, lines, and distances, including cardinals.
    """
    icons = []
    guesses = []
    lines = []
    # Draw cardinal lines
    pygame.draw.line(screen, (100, 255, 100), (center_x, 0), (center_x, screen.get_height()), 2)
    pygame.draw.line(screen, (100, 255, 100), (0, center_y), (screen.get_width(), center_y), 2)
    for k, v in exp_result.items():
        icon_x, icon_y, _ = v["icon_pos"]
        icons.append((icon_x, icon_y))
        if v["icon_guess"]:
            guess_x, guess_y = v["icon_guess"]
            guesses.append((guess_x, guess_y))
            lines.append(((icon_x, icon_y), (guess_x, guess_y), v["euclidian_dist"]))
        else:
            guesses.append(None)
            lines.append(((icon_x, icon_y), None, None))

        # Draw icon to cardinals
        x_c_dist = v["metadata"].get("x_c_dist")
        y_c_dist = v["metadata"].get("y_c_dist")
        if x_c_dist is not None:
            # Vertical distance (horizontal line)
            pygame.draw.line(screen, (0, 200, 255), (icon_x, icon_y), (center_x, icon_y), 1)
            label = f"x_c:{x_c_dist:+}"
            text_surf = font.render(label, True, (0, 200, 255))
            mid_x = (icon_x + center_x) // 2
            screen.blit(text_surf, (mid_x, icon_y - 20))
        if y_c_dist is not None:
            # Horizontal distance (vertical line)
            pygame.draw.line(screen, (255, 200, 0), (icon_x, icon_y), (icon_x, center_y), 1)
            label = f"y_c:{y_c_dist:+}"
            text_surf = font.render(label, True, (255, 200, 0))
            mid_y = (icon_y + center_y) // 2
            screen.blit(text_surf, (icon_x + 10, mid_y))

        # Draw guess to cardinals if guess exists
        if v["icon_guess"]:
            guess_x, guess_y = v["icon_guess"]
            guess_x_c_dist = v.get("guess_x_c_dist")
            guess_y_c_dist = v.get("guess_y_c_dist")
            if guess_x_c_dist is not None:
                pygame.draw.line(screen, (0, 255, 180), (guess_x, guess_y), (center_x, guess_y), 1)
                label = f"x_c:{guess_x_c_dist:+}"
                text_surf = font.render(label, True, (0, 255, 180))
                mid_x = (guess_x + center_x) // 2
                screen.blit(text_surf, (mid_x, guess_y + 10))
            if guess_y_c_dist is not None:
                pygame.draw.line(screen, (255, 100, 100), (guess_x, guess_y), (guess_x, center_y), 1)
                label = f"y_c:{guess_y_c_dist:+}"
                text_surf = font.render(label, True, (255, 100, 100))
                mid_y = (guess_y + center_y) // 2
                screen.blit(text_surf, (guess_x + 10, mid_y))

    # Draw icons (blue)
    for (ix, iy) in icons:
        pygame.draw.circle(screen, (0, 128, 255), (ix, iy), icon_size_px // 2)
    # Draw guesses (red)
    for g in guesses:
        if g:
            pygame.draw.circle(screen, (255, 0, 0), g, 8)
    # Draw lines and distances
    for (icon_pos, guess_pos, dist) in lines:
        if guess_pos:
            pygame.draw.line(screen, (128, 0, 128), icon_pos, guess_pos, 2)
            mid_x = (icon_pos[0] + guess_pos[0]) // 2
            mid_y = (icon_pos[1] + guess_pos[1]) // 2
            label = f"{dist:.1f}" if dist is not None else "?"
            text_surf = font.render(label, True, (180, 0, 180))
            screen.blit(text_surf, (mid_x, mid_y))

def pick_random_base_image(test_type):
    """
    Picks a random PNG from /res/base/{test_type}/.
    """
    base_dir = os.path.join(os.path.dirname(__file__), "res", "base", test_type)
    pngs = [f for f in os.listdir(base_dir) if f.lower().endswith('.png')]
    if not pngs:
        raise FileNotFoundError(f"No PNG images found in {base_dir}")
    return os.path.join(base_dir, random.choice(pngs))

def get_easy_spatial_question():
    """
    Returns a random (question, answer) tuple about spatial reasoning and map logic.
    Questions focus on visualization, direction, and spatial relationships.
    All answers are simple and based on logical thinking, not memorized facts.
    """
    qa = [
        # Grundlegende Richtung und Orientierung
        ("Du blickst nach Norden.\nWenn du 90 Grad nach rechts drehst\nin welche Richtung blickst du jetzt?", ["Osten"]),
        ("Du blickst nach Süden.\nWenn du dich komplett umdrehst\nin welche Richtung blickst du jetzt?", ["Norden", "Süden"]),
        ("Du blickst nach Osten.\nWenn du 90 Grad nach links drehst\nin welche Richtung blickst du jetzt?", ["Norden"]),
        ("Du blickst nach Westen.\nWenn du dich 180 Grad drehst\nin welche Richtung blickst du jetzt?", ["Osten"]),
        
        # Kartenlesen und Navigation
        ("Auf einer Standardkarte, auf der Norden oben ist\nin welche Richtung ist deine linke Seite?", ["Westen"]),
        ("Auf einer Standardkarte, auf der Norden oben ist\nin welche Richtung ist deine rechte Seite?", ["Osten"]),
        ("Auf einer Standardkarte, auf der Norden oben ist\nin welche Richtung ist unten?", ["Süden"]),
        ("Wenn du von der unteren linken Ecke zur oberen rechten Ecke einer Karte gehst,\nin welche Richtung reist du?", ["Nordosten"]),
        ("Wenn du von der oberen linken Ecke zur unteren rechten Ecke einer Karte gehst,\nin welche Richtung reist du?", ["Südosten"]),
        
        # Pfad- und Routenlogik
        ("Du gehst 10 Schritte nach Osten, dann 10 Schritte nach Norden.\nIn welche Richtung solltest du zuerst gehen?", ["Süden", "Westen", "Südwesten"]),
        ("Du gehst nach Süden, dann nach Westen, dann nach Norden.\nWelche Richtung macht deinen Weg zurück zum Start?", ["Osten"]),
        ("Du gehst 5 Minuten nach Norden, dann 5 Minuten nach Osten.\nWenn du zurück zu deinem Start gehst, in welche Richtung gehst du?", ["Southwest"]),
        
        # Form- und Geometrielogik
        ("Du stehst im Zentrum eines quadratischen Raumes.\nWenn du geradewegs in eine Ecke gehst\nwelche Art von Linie ist dein Weg?", ["diagonal", "diagonale"]),
        ("Du bist an einem Rand eines Kreises.\nWenn du durch das Zentrum zum gegenüberliegenden Rand gehst\nwie nennt man diese Strecke?", ["Durchmesser"]),
        ("Du faltest ein quadratisches Blatt Papier genau in der Mitte.\nWelche Form hast du jetzt?", ["Rechteck"]),
        
        # Koordinaten- und Positionslogik
        ("Du bist am Äquator.\nWenn du nach Norden gehst, welches Hemisphären betrittst du?", ["Nordhalbkugel"]),
        ("Du bist am Äquator.\nWenn du nach Süden gehst, welches Hemisphären betrittst du?", ["Südhalbkugel"]),
        ("Du stehst am Nordpol.\nWenn du in irgendeine Richtung blickst, wo ist Süden?", ["jede", "alle", "unten"]),
        ("Du stehst am Südpol.\nWenn du in irgendeine Richtung blickst, wo ist Norden?", ["jede", "alle", "unten"]),
        
        # Uhrzeit- und Winkelbeziehungen
        ("Stell dir vor, du bist im Zentrum eines Zifferblatts.\nWenn du auf 3 Uhr zeigst, welche Himmelsrichtung ist das?", ["Osten"]),
        ("Stell dir vor, du bist im Zentrum eines Zifferblatts.\nWenn du auf 12 Uhr zeigst, welche Himmelsrichtung ist das?", ["Norden"]),
        ("Stell dir vor, du bist im Zentrum eines Zifferblatts.\nWenn du auf 6 Uhr zeigst, welche Himmelsrichtung ist das?", ["Süden"]),
        ("Stell dir vor, du bist im Zentrum eines Zifferblatts.\nWenn du auf 9 Uhr zeigst, welche Himmelsrichtung ist das?", ["Westen"]),
        
        # Drehung und Transformation
        ("Du hast eine Karte, bei der Norden nach oben zeigt.\nWenn du sie um 180 Grad drehst,\nin welche Richtung zeigt Norden jetzt?", ["unten", "Süden"]),
        ("Du hast einen Kompass, der alle vier Hauptrichtungen anzeigt.\nWie viele Hauptrichtungen zeigt er?", ["vier"]),
        
        # Mehrstufige räumliche Logik
        ("Du läufst ein Dreieck ab und kehrst zu deinem Ausgangspunkt zurück.\nWie viele Wendungen machst du?", ["drei", "zwei"]),
        ("Du läufst ein Viereck ab und kehrst zu deinem Ausgangspunkt zurück.\nWie viele Wendungen machst du?", ["vier", "drei"]),
        ("Du bist in einem würfelförmigen Raum.\nWenn du von einer Ecke zur weitesten Ecke gehst,\nwelche Art von Pfad ist das?", ["diagonal", "diagonale"]),
        
        # Distanz- und Messlogik
        ("Du bist im Zentrum eines Kreises.\nAlle Punkte am Rand haben die gleiche was von dir?", ["Entfernung", "Distanz"]),
        ("Du gehst vollständig um den äußeren Rand eines Kreises.\nWie nennt man die zurückgelegte Strecke?", ["Umfang"]),
        
        # Fortgeschrittene räumliche Überlegungen
        ("Du schaust auf ein Quadrat.\nWenn du eine Linie ziehst, die zwei gegenüberliegende Ecken verbindet,\nwie nennt man diese Linie?", ["diagonal", "diagonale"]),
        ("Du startest nach Norden und machst vier 90-Grad-Rechtsdrehungen.\nIn welche Richtung blickst du jetzt?", ["Norden"]),
        ("Du startest bei Punkt A und gehst zu Punkt B, dann zu Punkt C, dann zurück zu Punkt A.\nWelche Form hast du nachgezeichnet?", ["Dreieck"]),
    ]
    
    q, a = random.choice(qa)
    # Ensure answer is always a list for consistency
    if isinstance(a, str):
        a = [a]
    return q, a

def run_experiment(
    participant_name,
    test_type,
    n_test,
    rebuild_svg=False,
    debug=False,
    log_file_level=logging.DEBUG,
    log_console_level=logging.INFO,
    image_size_cm=30,
    png_res=128,
    icon_size_px=36,
    n_icons=5,
    distances=None,
    icon_display_time_sec_tuple=(25, 15, 25),  # (icons, question, guess)
    results_path=None,
    screen=None,
    screen_width=None,
    screen_height=None,
    show_info_text=True,  # <--- NEW PARAMETER
):
    """
    Run a single experiment session.
    Returns a dictionary with experiment data for saving.
    """
    if distances is None:
        distances = [110, 220, 330, 440, 550]
    if results_path is None:
        results_path = os.path.join(os.path.dirname(__file__), "results.json")

    # --- Logging ---
    logging.info("| Experiment started")
    ensure_svg_icons_converted_to_png(rebuild_svg=rebuild_svg, png_res=png_res)
    # Remove pygame.init() and display setup here
    if screen is None or screen_width is None or screen_height is None:
        pygame.init()
        info = pygame.display.Info()
        screen_width, screen_height = info.current_w, info.current_h
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
        pygame.display.set_caption("Experiment")
    logging.info(f"| Screen size: {screen_width}x{screen_height} px")
    logging.info(f"| Screen physical size (if known): {screen_width / 96 * 2.54:.1f}cm x {screen_height / 96 * 2.54:.1f}cm (assuming 96 DPI)")
    logging.info(f"| Configured image size: {image_size_cm} cm, PNG icon resolution: {png_res}px")

    # --- Pick base image for test_type ---
    image_path = pick_random_base_image(test_type)
    logging.info(f"| Picked base image: {image_path}")

    # --- Get easy spatial question ---
    question, correct_answers = get_easy_spatial_question()
    logging.info(f"| Selected spatial question: {question} (answers: {correct_answers})")

    # --- Calculate image height in pixels for image_size_cm ---
    try:
        if sys.platform.startswith("win"):
            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32
            # Try GetDpiForWindow (Windows 10 creators update+)
            try:
                hwnd = user32.GetForegroundWindow()
                get_dpi_for_window = user32.GetDpiForWindow
                get_dpi_for_window.restype = ctypes.c_uint
                dpi = get_dpi_for_window(hwnd)
            except AttributeError:
                # Try GetDpiForSystem (Windows 10+)
                try:
                    dpi = user32.GetDpiForSystem()
                except AttributeError:
                    # Fallback: GetDeviceCaps
                    hdc = user32.GetDC(0)
                    LOGPIXELSX = 88
                    dpi = gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
                    user32.ReleaseDC(0, hdc)
        else:
            dpi = 96
    except Exception:
        dpi = 96  # fallback
    
    cm_per_inch = 2.54
    image_height_px = int((image_size_cm / cm_per_inch) * dpi)

    # --- Load resources ---
    img = display_image_fullscreen(image_path, image_height_px)
    logging.info(f"| Loaded base image for display")
    icon_surfaces = load_icon_surfaces(icon_size_px)
    icon_surface_paths = [
        os.path.join(os.path.dirname(__file__), "res", "icons", f)
        for f in sorted([f for f in os.listdir(os.path.join(os.path.dirname(__file__), "res", "icons")) if f.lower().endswith('.png')])
    ]
    logging.info(f"| Loaded {len(icon_surfaces)} icon surfaces")

    # --- State ---
    phase = "icons"
    icon_coords = []
    icon_assignments = []
    guesses = []
    question_answer = ""
    running = True
    font = pygame.font.SysFont(None, 36)
    input_font = pygame.font.SysFont(None, 48)
    input_text = ""
    esc_counter = 0

    # --- Question phase logic variables ---
    incorrect_attempts = 0
    question_locked_until = 0  # timestamp until which input is locked after wrong answer

    # --- Correct answer feedback state ---
    correct_feedback_until = 0  # timestamp until which to show green feedback
    correct_feedback_active = False
    empty_input_feedback_until = 0  # timestamp for empty input feedback

    # --- Grace period and debounce state ---
    phase_grace_until = 0  # timestamp until which Enter is ignored after phase change
    last_enter_time = 0    # for debounce: last time Enter was processed

    # --- Icon scatter ---
    center_x, center_y = screen_width // 2, screen_height // 2
    icon_coords = scatter_icons_around_center(center_x, center_y, distances, n_icons)
    icon_assignments = [random.choice(icon_surfaces) for _ in range(n_icons)]
    icon_assignment_paths = [icon_surface_paths[icon_surfaces.index(icon)] for icon in icon_assignments]
    logging.debug(f"| Icon coordinates: {icon_coords}")

    # --- Timers ---
    # Use tuple for phase durations
    icon_phase_duration = icon_display_time_sec_tuple[0]
    question_phase_duration = icon_display_time_sec_tuple[1]
    guess_phase_duration = icon_display_time_sec_tuple[2]
    phase_start_time = time.time()
    phase_end_time = phase_start_time + icon_phase_duration

    def get_exp_result_for_debug():
        # Helper to get the current exp_result dict for debug overlay
        icon_points = [(x, y, angle) for (x, y, angle, dist) in icon_coords]
        icon_attrs = []
        for (x, y, angle, dist) in icon_coords:
            x_c_dist = x - center_x
            y_c_dist = y - center_y
            icon_attrs.append({"distance": dist, "angle": angle, "x_c_dist": x_c_dist, "y_c_dist": y_c_dist})
        guesses_local = guesses[:]
        icon_to_guess = optimal_icon_guess_assignment(icon_points, guesses_local)
        exp_result = {}
        for idx, (icon, attrs) in enumerate(zip(icon_points, icon_attrs)):
            icon_x, icon_y, icon_angle = icon
            icon_label = f"icon_{idx+1}"
            guess_idx = icon_to_guess[idx] if idx < len(icon_to_guess) else None
            if guess_idx is not None and guess_idx < len(guesses_local):
                guess_x, guess_y = guesses_local[guess_idx]
                dist_x = guess_x - icon_x
                dist_y = guess_y - icon_y
                euclidian = math.hypot(dist_x, dist_y)
                icon_guess = (guess_x, guess_y)
                dist_to_icon = (dist_x, dist_y, euclidian)
                guess_x_c_dist = guess_x - center_x
                guess_y_c_dist = guess_y - center_y
            else:
                icon_guess = None
                dist_to_icon = (None, None, None)
                euclidian = None
                guess_x_c_dist = None
                guess_y_c_dist = None
            exp_result[icon_label] = {
                "icon_pos": (icon_x, icon_y, icon_angle),
                "icon_guess": icon_guess,
                "dist_to_icon": dist_to_icon,
                "euclidian_dist": euclidian,
                "metadata": attrs,
                "guess_x_c_dist": guess_x_c_dist,
                "guess_y_c_dist": guess_y_c_dist
            }
        return exp_result

    # --- Main loop ---
    while running:
        screen.fill((0, 0, 0))
        now = time.time()
        elapsed = now - phase_start_time
        remaining = max(0, int(phase_end_time - now))
        timer_text = f"Time: {int(elapsed)}s / {int(phase_end_time - phase_start_time)}s (left: {remaining}s)"

        # --- Handle phase grace period ---
        # If just entered a phase, set grace period
        if phase_grace_until == 0:
            phase_grace_until = now + 3  # 3 seconds grace at phase start

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logging.info("| Received QUIT event")
                running = False
            elif event.type == pygame.KEYDOWN:
                # Debounce Enter: only process if enough time since last Enter
                is_enter = event.key == pygame.K_RETURN
                enter_allowed = (not is_enter) or (now > phase_grace_until and now - last_enter_time > 0.2)
                if not enter_allowed:
                    continue  # Ignore Enter if grace period or debounce

                if is_enter:
                    last_enter_time = now  # Update debounce timestamp

                if phase == "icons":
                    if event.key == pygame.K_RETURN:
                        logging.info("| Phase changed: icons -> question")
                        phase = "question"
                        input_text = ""
                        phase_start_time = time.time()
                        phase_end_time = phase_start_time + question_phase_duration
                        phase_grace_until = phase_start_time + 3  # Set grace period for new phase
                    elif event.key == pygame.K_ESCAPE:
                        esc_counter += 1
                        if esc_counter >= 3:
                            logging.info("| Escape pressed 3 times, exiting ENTIRE SESSION")
                            return None  # <--- Return None to signal abort
                    else:
                        esc_counter = 0
                elif phase == "question":
                    if now < question_locked_until or correct_feedback_active or now < empty_input_feedback_until:
                        continue  # Ignore input while locked or feedback active
                    if event.key == pygame.K_RETURN:
                        user_ans = input_text.strip().lower()
                        if not user_ans:
                            # Empty input: show feedback
                            empty_input_feedback_until = now + 2
                            continue
                        if any(user_ans == ans.strip().lower() for ans in correct_answers):
                            question_answer = input_text
                            logging.info(f"| Question answered correctly: {question_answer}")
                            input_text = ""
                            correct_feedback_active = True
                            correct_feedback_until = now + 2  # Show green feedback for 2 seconds
                        else:
                            incorrect_attempts += 1
                            logging.info(f"| Incorrect answer attempt {incorrect_attempts}: {input_text}")
                            if incorrect_attempts >= 3:
                                logging.info("| Three incorrect answers, moving to guess phase")
                                question_answer = input_text
                                input_text = ""
                                phase = "guess"
                                phase_start_time = time.time()
                                phase_end_time = phase_start_time + guess_phase_duration
                                phase_grace_until = phase_start_time + 3  # Set grace period for new phase
                            else:
                                question_locked_until = now + 5  # Lock input for 3 seconds
                                input_text = ""  # Clear input so user can start fresh
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        esc_counter += 1
                        if esc_counter >= 3:
                            logging.info("| Escape pressed 3 times, exiting ENTIRE SESSION")
                            return None  # <--- Return None to signal abort
                    else:
                        esc_counter = 0
                        if event.unicode and len(event.unicode) == 1:
                            input_text += event.unicode
                elif phase == "guess":
                    if event.key == pygame.K_RETURN:
                        logging.info("| Experiment finished, exiting")
                        running = False
                    elif event.key == pygame.K_ESCAPE:
                        esc_counter += 1
                        if esc_counter >= 3:
                            logging.info("| Escape pressed 3 times, exiting ENTIRE SESSION")
                            return None  # <--- Return None to signal abort
                    else:
                        esc_counter = 0
            elif event.type == pygame.MOUSEBUTTONDOWN and phase == "guess":
                mx, my = event.pos
                # --- Only allow guesses within the image bounds ---
                img_rect = img.get_rect(center=(center_x, center_y))
                if event.button == 1:  # Left click to add
                    if len(guesses) < n_icons:
                        if img_rect.collidepoint(mx, my):
                            guesses.append((mx, my))
                            logging.info(f"| Guess added at ({mx}, {my})")
                        else:
                            logging.info(f"| Guess at ({mx}, {my}) ignored (out of image bounds)")
                elif event.button == 3:  # Right click to remove nearest
                    if guesses:
                        dists = [math.hypot(mx - gx, my - gy) for gx, gy in guesses]
                        idx = dists.index(min(dists))
                        removed = guesses.pop(idx)
                        logging.info(f"| Guess removed at {removed}")

        # --- Phase logic ---
        if phase == "icons":
            # Draw main image
            img_rect = img.get_rect(center=(center_x, center_y))
            screen.blit(img, img_rect)
            # Draw icons
            for (x, y, angle, dist), icon_surf in zip(icon_coords, icon_assignments):
                icon_rect = icon_surf.get_rect(center=(x, y))
                screen.blit(icon_surf, icon_rect)
            # Timer: auto-advance
            if now >= phase_end_time:
                phase = "question"
                input_text = ""
                phase_start_time = time.time()
                phase_end_time = phase_start_time + question_phase_duration
                phase_grace_until = phase_start_time + 3  # Set grace period for new phase
                logging.info("| Icon phase timed out, moving to question phase")
            # Draw timer at bottom
            if show_info_text:
                instr = "Press Enter to continue, Esc x3 to exit"
                t_surf = font.render(f"{instr} | {timer_text}", True, (200, 200, 200))
                screen.blit(t_surf, (center_x - t_surf.get_width() // 2, screen_height - 60))
        elif phase == "question":
            # Draw question text (support line breaks)
            q_lines = question.split('\n')
            y0 = center_y - 100 - (len(q_lines)-1)*22  # Adjust vertical position for multiple lines

            # --- Correct answer feedback (green) ---
            if correct_feedback_active:
                if now < correct_feedback_until:
                    for i, qline in enumerate(q_lines):
                        qsurf = font.render(qline, True, (0, 255, 0))
                        screen.blit(qsurf, (center_x - qsurf.get_width() // 2, y0 + i*44))
                    # No input box, just feedback
                else:
                    # End feedback, advance phase
                    correct_feedback_active = False
                    phase = "guess"
                    logging.info("| Phase changed: question -> guess (after correct answer feedback)")
                    phase_start_time = time.time()
                    phase_end_time = phase_start_time + guess_phase_duration
                    phase_grace_until = phase_start_time + 3  # Set grace period for new phase
            else:
                for i, qline in enumerate(q_lines):
                    qsurf = font.render(qline, True, (255, 255, 255))
                    screen.blit(qsurf, (center_x - qsurf.get_width() // 2, y0 + i*44))
                # Draw input text
                insurf = input_font.render(input_text, True, (255, 255, 0))
                screen.blit(insurf, (center_x - insurf.get_width() // 2, center_y))
                # Show feedback if locked
                if now < question_locked_until:
                    lock_msg = f"Incorrect! Try again in {int(question_locked_until - now)}s"
                    locksurf = font.render(lock_msg, True, (255, 80, 80))
                    screen.blit(locksurf, (center_x - locksurf.get_width() // 2, center_y + 60))
                # Show empty input feedback
                if now < empty_input_feedback_until:
                    empty_msg = "Bitte gib eine Antwort ein!"
                    emptysurf = font.render(empty_msg, True, (255, 80, 80))
                    screen.blit(emptysurf, (center_x - emptysurf.get_width() // 2, center_y + 100))
            # Timer: auto-advance
            if not correct_feedback_active and now >= phase_end_time:
                question_answer = input_text
                input_text = ""
                phase = "guess"
                phase_start_time = time.time()
                phase_end_time = phase_start_time + guess_phase_duration
                phase_grace_until = phase_start_time + 3  # Set grace period for new phase
                logging.info("| Question phase timed out, moving to guess phase")
            # Draw timer at bottom
            if show_info_text:
                instr = "Type answer, Enter to continue, Esc x3 to exit"
                t_surf = font.render(f"{instr} | {timer_text}", True, (200, 200, 200))
                screen.blit(t_surf, (center_x - t_surf.get_width() // 2, screen_height - 60))
        elif phase == "guess":
            # Draw main image
            img_rect = img.get_rect(center=(center_x, center_y))
            screen.blit(img, img_rect)
            # Draw guesses as red dots (if not debug)
            if not debug:
                for gx, gy in guesses:
                    pygame.draw.circle(screen, (255, 0, 0), (gx, gy), 7)
            # Draw debug overlay if debug is True
            if debug:
                exp_result = get_exp_result_for_debug()
                draw_debug_overlay(exp_result, screen, icon_size_px, font, center_x, center_y)
            # Timer: auto-advance
            if now >= phase_end_time:
                logging.info("| Guess phase timed out, finishing experiment")
                running = False
            # Draw instructions and timer
            if show_info_text:
                instr = "Left click: add guess, Right click: remove nearest, Enter: finish, Esc x3 to exit"
                t_surf = font.render(f"{instr} | {timer_text}", True, (200, 200, 200))
                screen.blit(t_surf, (center_x - t_surf.get_width() // 2, screen_height - 60))

        pygame.display.flip()
        pygame.time.wait(10)

    # --- DO NOT call pygame.quit() here ---
    # Prepare flat experiment result for saving
    icon_points = [(x, y, angle) for (x, y, angle, dist) in icon_coords]
    icon_attrs = []
    for (x, y, angle, dist) in icon_coords:
        x_c_dist = x - center_x
        y_c_dist = y - center_y
        icon_attrs.append({"distance": dist, "angle": angle, "x_c_dist": x_c_dist, "y_c_dist": y_c_dist})
    guesses_local = guesses[:]
    icon_to_guess = optimal_icon_guess_assignment(icon_points, guesses_local)

    # --- Flat structure for CSV ---
    points = {}
    for idx, (icon, attrs) in enumerate(zip(icon_points, icon_attrs)):
        icon_x, icon_y, icon_angle = icon
        icon_label = str(idx + 1)
        guess_idx = icon_to_guess[idx] if idx < len(icon_to_guess) else None

        # Target info
        target_x = icon_x
        target_y = icon_y
        target_degrees = math.degrees(icon_angle)
        target_cd_x = attrs["x_c_dist"]
        target_cd_y = attrs["y_c_dist"]
        icon_path = icon_assignment_paths[idx] if idx < len(icon_assignment_paths) else None

        # Guess info
        if guess_idx is not None and guess_idx < len(guesses_local):
            guess_x, guess_y = guesses_local[guess_idx]
            dist_x = guess_x - icon_x
            dist_y = guess_y - icon_y
            euclidian = math.hypot(dist_x, dist_y)
            guess_angle = math.degrees(math.atan2(guess_y - center_y, guess_x - center_x))
            guess_cd_x = guess_x - center_x
            guess_cd_y = guess_y - center_y
        else:
            guess_x = None
            guess_y = None
            guess_angle = None
            guess_cd_x = None
            guess_cd_y = None
            euclidian = None

        points[icon_label] = {
            "target_x": target_x,
            "target_y": target_y,
            "target_degrees": target_degrees,
            "target_cd_x": target_cd_x,
            "target_cd_y": target_cd_y,
            "icon_path": icon_path,
            "guess_x": guess_x,
            "guess_y": guess_y,
            "guess_degrees": guess_angle,
            "guess_cd_x": guess_cd_x,
            "guess_cd_y": guess_cd_y,
            "guess_eucldist": euclidian,
        }

    flat_result = {
        "participant": participant_name,
        "test_type": test_type,
        "n_test": n_test,
        "base_image_path": image_path,
        "center_x": center_x,
        "center_y": center_y,
        "metadata": {k: v for k, v in enumerate(icon_attrs, 1)},
        "points": points,
        # Optionally: "question_answer": question_answer,
    }

    # --- Add experiment statistics ---
    stats = calculate_experiment_statistics(flat_result)
    flat_result["statistics"] = stats

    logging.debug("| Experiment result dictionary: %s", flat_result)
    return flat_result

# --- New statistics function ---
def calculate_experiment_statistics(flat_result):
    """
    Calculate experiment-level statistics for summary and CSV export.
    Returns a dict of statistics.
    """
    points = flat_result["points"]
    center_x = flat_result["center_x"]
    center_y = flat_result["center_y"]

    # Prepare lists
    euclid_errors = []
    angular_devs = []
    cd_xs = []
    cd_ys = []

    for idx, pdata in points.items():
        tx, ty = pdata["target_x"], pdata["target_y"]
        gx, gy = pdata["guess_x"], pdata["guess_y"]
        tcdx, tcdy = pdata["target_cd_x"], pdata["target_cd_y"]
        gcdx, gcdy = pdata["guess_cd_x"], pdata["guess_cd_y"]
        eucl = pdata["guess_eucldist"]

        # Only if guess exists
        if gx is not None and gy is not None:
            # Euclidean error
            euclid_errors.append(eucl)

            # Angular deviation from nearest cardinal axis (0/90/180/270)
            angle = math.degrees(math.atan2(gy - center_y, gx - center_x)) % 360
            cardinal_angles = [0, 90, 180, 270]
            min_dev = min([abs((angle - ca + 180) % 360 - 180) for ca in cardinal_angles])
            angular_devs.append(min_dev)

            # Center distances
            cd_xs.append(gcdx)
            cd_ys.append(gcdy)

    # Aggregate statistics
    stats = {
        "avg_euclid_error": float(np.mean(euclid_errors)) if euclid_errors else None,
        "avg_angular_deviation_cardinal": float(np.mean(angular_devs)) if angular_devs else None,
        "avg_cd_x": float(np.mean(np.abs(cd_xs))) if cd_xs else None,
        "avg_cd_y": float(np.mean(np.abs(cd_ys))) if cd_ys else None,
    }
    return stats

def merge_to_csv(exp_data, csv_path, user_data):
    """
    Merge experiment data into a CSV file. If file does not exist, create header.
    Columns are named as: {point}{type}_{field}, e.g. 1t_pos_x, 2g_cd_x, 3m_euclidistance.
    Also includes experiment-level statistics.
    user_data: dict with keys 'age', 'gender', 'codeword'
    """
    
    assert isinstance(exp_data, dict), "exp_data must be a dictionary"
    assert isinstance(csv_path, str), "csv_path must be a string"
    assert os.path.isdir(os.path.dirname(csv_path)), "Directory for csv_path does not exist"
    assert isinstance(user_data, dict) or user_data is None, "user_data must be a dictionary or None"

    # Prepare row with global experiment info
    row = {}
    row["codeword"] = user_data.get("codeword", "")
    row["age"] = user_data.get("age", "")
    row["gender"] = user_data.get("gender", "")
    row.update({
        "test_type": exp_data["test_type"],
        "n_test": exp_data["n_test"],
        "base_image_path": exp_data["base_image_path"],
        "center_x": exp_data["center_x"],
        "center_y": exp_data["center_y"],
    })

    # Prepare per-point fields with new naming convention
    point_fields = []
    n_points = len(exp_data["points"])
    for idx in range(1, n_points + 1):
        meta = exp_data["metadata"][idx]
        point = exp_data["points"][str(idx)]
        fields = {
            f"{idx}t_pos_x": point["target_x"],
            f"{idx}t_pos_y": point["target_y"],
            f"{idx}t_angle_deg": point["target_degrees"],
            f"{idx}t_cd_x": point["target_cd_x"],
            f"{idx}t_cd_y": point["target_cd_y"],
            f"{idx}t_icon_path": point["icon_path"],
            f"{idx}m_distance": meta["distance"],
            f"{idx}m_angle": meta["angle"],
            f"{idx}m_cd_x": meta["x_c_dist"],
            f"{idx}m_cd_y": meta["y_c_dist"],
            f"{idx}g_pos_x": point["guess_x"],
            f"{idx}g_pos_y": point["guess_y"],
            f"{idx}g_angle_deg": point["guess_degrees"],
            f"{idx}g_cd_x": point["guess_cd_x"],
            f"{idx}g_cd_y": point["guess_cd_y"],
            f"{idx}g_euclidistance": point["guess_eucldist"],
        }
        point_fields.append(fields)

    # Flatten all fields into the row, keeping each point's fields together
    for fields in point_fields:
        row.update(fields)

    # Add statistics (flattened)
    stats = exp_data.get("statistics", {})
    # Flatten nested dicts (e.g. accuracy_by_angular_bucket) for CSV columns
    flat_stats = {}
    for k, v in stats.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                flat_stats[f"stat_{k}_{subk}"] = subv
        elif isinstance(v, list):
            # For lists, store as stringified list (e.g. for guess_accuracy_vs_target_distance)
            flat_stats[f"stat_{k}"] = str(v)
        else:
            flat_stats[f"stat_{k}"] = v
    row.update(flat_stats)

    # Extend header with stat keys (if not present)
    stat_keys = list(flat_stats.keys())
    header = []
    if user_data:
        header.extend(["codeword", "age", "gender"])
    header.extend([
        "test_type", "n_test", "base_image_path", "center_x", "center_y"
    ])
    for idx in range(1, n_points + 1):
        header.extend([
            f"{idx}t_pos_x",
            f"{idx}t_pos_y",
            f"{idx}t_angle_deg",
            f"{idx}t_cd_x",
            f"{idx}t_cd_y",
            f"{idx}t_icon_path",
            f"{idx}m_distance",
            f"{idx}m_angle",
            f"{idx}m_cd_x",
            f"{idx}m_cd_y",
            f"{idx}g_pos_x",
            f"{idx}g_pos_y",
            f"{idx}g_angle_deg",
            f"{idx}g_cd_x",
            f"{idx}g_cd_y",
            f"{idx}g_euclidistance",
        ])
    header.extend(stat_keys)

    # Write to CSV
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    logging.info(f"Results saved to CSV: {csv_path}")

def get_participant_info(screen=None, screen_width=None, screen_height=None):
    """
    Interactive pygame window to collect participant info: age, gender, codeword.
    Returns a dict: {"age": ..., "gender": ..., "codeword": ...}
    """
    logging.info("| Starting participant info acquisition")
    # Only init and set display if not provided
    if screen is None or screen_width is None or screen_height is None:
        pygame.init()
        info = pygame.display.Info()
        screen_width, screen_height = info.current_w, info.current_h
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)

    font = pygame.font.SysFont(None, 48)
    small_font = pygame.font.SysFont(None, 36)
    fields = [("Alter", ""), ("Geschlecht (M/W/D/O)", ""), ("Ein Codewort/Nutzername etc", "")]
    field_idx = 0
    input_active = True
    input_text = ""
    esc_counter = 0

    # Calculate the height of the full window content to center everything
    total_height = 80 + len(fields) * 80 + 80  # Padding and spacing
    y_offset = (screen_height - total_height) // 2  # Center the content vertically

    while input_active:
        screen.fill((30, 30, 30))
        
        # Instructions
        instr = "Bitte gib deine Informationen ein"
        instr_surf = small_font.render(instr, True, (200, 200, 200))
        screen.blit(instr_surf, (screen_width // 2 - instr_surf.get_width() // 2, y_offset + 20))

        # Draw fields
        for i, (label, value) in enumerate(fields):
            color = (255, 255, 0) if i == field_idx else (255, 255, 255)
            text = f"{label}: {value}" if i != field_idx else f"{label}: {input_text}"
            surf = font.render(text, True, color)
            screen.blit(surf, (screen_width // 2 - surf.get_width() // 2, y_offset + 80 + i * 80))

        # Draw submit hint
        if field_idx >= len(fields):
            done_surf = font.render("Drücke Enter, um fortzufahren...", True, (0, 255, 0))
            screen.blit(done_surf, (screen_width // 2 - done_surf.get_width() // 2, y_offset + 200 + len(fields) * 80))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logging.info("| Participant info window closed by user")
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if field_idx < len(fields):
                    if event.key == pygame.K_RETURN:
                        logging.info(f"| Field '{fields[field_idx][0]}' entered: {input_text.strip()}")
                        fields[field_idx] = (fields[field_idx][0], input_text.strip())
                        input_text = ""
                        field_idx += 1
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        esc_counter += 1
                        if esc_counter >= 3:
                            logging.info("| Escape pressed 3 times during participant info, exiting")
                            pygame.quit()
                            sys.exit(0)
                    else:
                        esc_counter = 0
                        if event.unicode and len(event.unicode) == 1:
                            input_text += event.unicode
                else:
                    if event.key == pygame.K_RETURN:
                        logging.info("| Participant info acquisition complete, continuing")
                        input_active = False
                    elif event.key == pygame.K_ESCAPE:
                        esc_counter += 1
                        if esc_counter >= 3:
                            logging.info("| Escape pressed 3 times during participant info, exiting")
                            pygame.quit()
                            sys.exit(0)
                    else:
                        esc_counter = 0
        pygame.time.wait(10)

    # --- convert to lower cases ---
    fields = [(label, value.lower().strip()) for label, value in fields]

    # --- Check inputs ---
    if not fields[0][1].isdigit() or int(fields[0][1]) < 0:
        logging.info("| Invalid age input detected")
        raise ValueError("Invalid age input. Please enter a valid number.")
    if fields[1][1].strip() not in ["m", "w", "d", "o"]:
        logging.info("| Invalid gender input detected")
        raise ValueError("Ungültige Eingabe für Geschlecht. Bitte gib 'M' - Männlich, 'W' - Weiblich, 'D' - Divers oder 'O' - Sonstiges ein.")
    if not fields[2][1].strip():
        logging.info("| Empty codeword input detected")
        raise ValueError("Codewort darf nicht leer sein. Bitte gib ein gültiges Codewort ein.")

    logging.info("| Participant info collected: Age=%s, Gender=%s, Codeword=%s", fields[0][1], fields[1][1], fields[2][1])
    return {
        "age": fields[0][1],
        "gender": fields[1][1],
        "codeword": fields[2][1]
    }


# --- Thank you screen ---
def show_thank_you_screen(screen):
    """
    Displays a thank you screen with a random image from res/trophy in the center.
    Waits for any key or mouse press to exit.
    """
    font = pygame.font.SysFont(None, 72)
    small_font = pygame.font.SysFont(None, 48)
    # Load random trophy image
    trophy_dir = os.path.join(os.path.dirname(__file__), "res", "rewards")
    trophy_files = [f for f in os.listdir(trophy_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webm'))]
    trophy_img = None
    img_rect = None
    if trophy_files:
        trophy_path = os.path.join(trophy_dir, random.choice(trophy_files))
        trophy_img = pygame.image.load(trophy_path).convert_alpha()
        # Scale trophy image to fit (max 40% of screen height)
        sw, sh = screen.get_width(), screen.get_height()
        max_dim = int(sh * 0.4)
        img_rect = trophy_img.get_rect()
        scale_factor = min(max_dim / img_rect.width, max_dim / img_rect.height, 1.0)
        new_size = (int(img_rect.width * scale_factor), int(img_rect.height * scale_factor))
        trophy_img = pygame.transform.smoothscale(trophy_img, new_size)
        img_rect = trophy_img.get_rect()
    thank_you_active = True
    while thank_you_active:
        screen.fill((0, 0, 0))
        sw, sh = screen.get_width(), screen.get_height()
        # Calculate positions
        if trophy_img:
            img_rect = trophy_img.get_rect()
            img_rect.center = (sw // 2, sh // 2)
            # Text above image
            thank_surf = font.render("Thank you kindly for participating!", True, (255, 255, 0))
            thank_y = img_rect.top - thank_surf.get_height() - 40
            if thank_y < 0:
                thank_y = 20
            screen.blit(thank_surf, (sw // 2 - thank_surf.get_width() // 2, thank_y))
            # Draw trophy image
            screen.blit(trophy_img, img_rect)
            # Text below image
            instr_surf = small_font.render("Press any key to exit.", True, (200, 200, 200))
            instr_y = img_rect.bottom + 40
            if instr_y + instr_surf.get_height() > sh:
                instr_y = sh - instr_surf.get_height() - 20
            screen.blit(instr_surf, (sw // 2 - instr_surf.get_width() // 2, instr_y))
        else:
            # If no image, center text vertically
            thank_surf = font.render("Thank you kindly for participating!", True, (255, 255, 0))
            instr_surf = small_font.render("Press any key to exit.", True, (200, 200, 200))
            screen.blit(thank_surf, (sw // 2 - thank_surf.get_width() // 2, sh // 2 - thank_surf.get_height()))
            screen.blit(instr_surf, (sw // 2 - instr_surf.get_width() // 2, sh // 2 + 20))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                thank_you_active = False
            elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                thank_you_active = False
        pygame.time.wait(10)

# --- Simple message screen ---
def show_simple_message_screen(screen):
    """
    Displays a simple message in the center of the screen.
    Waits for any key or mouse press to exit.
    """
    font = pygame.font.SysFont(None, 72)
    small_font = pygame.font.SysFont(None, 48)
    
    message = "The first section of the experiment is over"
    instruction = "Remove the cover from the screen"

    # Wait for exit event
    message_active = True
    while message_active:
        screen.fill((0, 0, 0))  # Fill the screen with black

        # Get screen width and height
        sw, sh = screen.get_width(), screen.get_height()
        
        # Render the message text
        message_surf = font.render(message, True, (255, 255, 0))
        instr_surf = small_font.render(instruction, True, (200, 200, 200))

        # Calculate center positions for the text
        message_rect = message_surf.get_rect(center=(sw // 2, sh // 2 - 30))
        instr_rect = instr_surf.get_rect(center=(sw // 2, sh // 2 + 30))
        
        # Blit the text to the screen
        screen.blit(message_surf, message_rect)
        screen.blit(instr_surf, instr_rect)

        # Update the display
        pygame.display.flip()

        # Check for any key or mouse press event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                message_active = False
            elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                message_active = False
        
        pygame.time.wait(10)  # Small delay to avoid high CPU usage

# --- Experiment intro screen ---
def show_experiment_intro_screen(screen):
    """
    Zeigt einen Einführungsscreen, der die Phasen des Experiments erklärt.
    Wartet auf eine beliebige Tasten- oder Maustaste, um fortzufahren.
    """
    font = pygame.font.SysFont(None, 60)
    small_font = pygame.font.SysFont(None, 40)
    sw, sh = screen.get_width(), screen.get_height()
    lines = [
        "Willkommen zum Experiment!",
        "",
        "Du wirst mehrere Durchgänge absolvieren, jeder mit drei Phasen:",
        "",
        "1. Icon-Phase:",
        "   - Mehrere Symbole erscheinen auf einer Karte.",
        "   - Merke dir ihre Positionen.",
        "",
        "2. Frage-Phase:",
        "   - Beantworte eine einfache räumliche Frage.",
        "   - Keine Stress, du hast ein paar Versuche.",
        "   - Irgendwann geht es automatisch weiter.",
        "",
        "3. Schätz-Phase:",
        "   - Platziere die Symbole dort, wo du sie erinnerst:",
        "   - Linksklick: Hinzufügen einer Schätzung",
        "   - Rechtsklick: Entfernen der nächsten Schätzung",
        "",
        "Folge den Anweisungen auf dem Bildschirm.",
        "",
        "Drücke Enter nach jedem Schritt - und habe kurz Geduld.",
        "",
        "Es gibt kein Tutorial, es geht mit Enter sofort los.",
        "",
        "Drücke eine beliebige Taste, um zu beginnen."
    ]
    intro_active = True
    while intro_active:
        screen.fill((0, 0, 0))
        y = sh // 2 - (len(lines) * 32) // 2
        for i, line in enumerate(lines):
            if i == 0:
                surf = font.render(line, True, (255, 255, 0))
            else:
                surf = small_font.render(line, True, (220, 220, 220))
            screen.blit(surf, (sw // 2 - surf.get_width() // 2, y))
            y += surf.get_height() + 4
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                intro_active = False
            elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                intro_active = False
        pygame.time.wait(10)

# --- CLI/Module entry point ---
def main():
    # --- Prepare logging ---
    setup_experiment_logging()
    logging.info("=== SESSION START ===")

    # --- Initialize pygame and display ONCE ---
    pygame.init()
    info = pygame.display.Info()
    screen_width, screen_height = info.current_w, info.current_h
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
    pygame.display.set_caption("Experiment")

    # --- Get participant info interactively ---
    user_data = get_participant_info(screen, screen_width, screen_height)

    # --- Show experiment intro screen ---
    show_experiment_intro_screen(screen)

    # --- Experiment configuration (defaults) ---
    participant_name = user_data["codeword"] if user_data.get("codeword") else "unknown"  # Use codeword as participant name
    test_type = "circle"             # Test type (folder in /res/base/)
    debug = False                     # Enable debug overlay
    rebuild_svg = False              # Rebuild PNGs from SVGs
    log_file_level = logging.INFO    # Log file level
    log_console_level = logging.INFO # Log console level
    results_path = os.path.join(os.path.dirname(__file__), "results.csv")  # Path to results CSV file
    icon_display_time_sec_tuple = (25, 30, 50)  # Phase times in seconds (icons, question, guess)
    show_info_text = False            # <--- NEW CONFIGURATION OPTION

    # --- Start single session ---

    # test list without occlusion
    test_list = ["circle", "circle", "circle", "circle", "circle", "circle", "circle", "square", "square", "square", "square", "square", "square", "square"]

    # test list with occlusion
    # test_list = ["occluded_circle", "occluded_circle", "occluded_circle", "occluded_circle", "occluded_circle", "occluded_circle", "occluded_circle"]

    # test list short
    occlusion_done = False
    test_list = ["occluded_circle", "occluded_circle", "occluded_circle", "circle", "circle", "circle", "square", "square", "square"]

    for i, test_nr in enumerate(test_list):
        logging.info(f"Experiment {i+1}/{len(test_list)}: {test_nr} START")

        if test_nr != "occluded_circle" and occlusion_done == False:
            occlusion_done = True
            show_simple_message_screen(screen)

        # Run the experiment for each test type
        exp_data = run_experiment(
            participant_name=participant_name,
            test_type=test_nr,
            n_test=i + 1,
            debug=debug,
            rebuild_svg=rebuild_svg,
            log_file_level=log_file_level,
            log_console_level=log_console_level,
            icon_display_time_sec_tuple=icon_display_time_sec_tuple,
            results_path=results_path,
            screen=screen,
            screen_width=screen_width,
            screen_height=screen_height,
            show_info_text=show_info_text,  # <--- PASS TO FUNCTION
        )
        if exp_data is None:
            logging.warning("Experiment aborted by user (3x ESC). Aborting session.")
            break  # Abort the entire session loop
        # Save results to CSV
        merge_to_csv(exp_data, results_path, user_data)
        logging.info(f"Experiment {i+1}/{len(test_list)}: {test_nr} END")

    logging.info("=== SESSION END ===")

    show_thank_you_screen(screen)

    pygame.quit()

if __name__ == "__main__":
    main()
