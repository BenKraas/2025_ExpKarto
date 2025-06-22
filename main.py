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
import cairosvg
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
    if cairosvg is None:
        logging.error("cairosvg is required for SVG to PNG conversion. Please install it.")
        print("cairosvg is required for SVG to PNG conversion. Please install it.")
        sys.exit(1)

    icons_dir = os.path.join(os.path.dirname(__file__), "res", "icons")
    for fname in os.listdir(icons_dir):
        if fname.lower().endswith('.svg'):
            svg_path = os.path.join(icons_dir, fname)
            png_name = os.path.splitext(fname)[0] + ".png"
            png_path = os.path.join(icons_dir, png_name)
            if rebuild_svg or not os.path.exists(png_path):
                logging.info(f"Converting {fname} to {png_name}...")
                png_bytes = cairosvg.svg2png(url=svg_path, output_width=png_res, output_height=png_res)
                pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
                pil_img.save(png_path)
            # else: print(f"PNG for {fname} already exists, skipping.")

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
    Returns a random (question, answer) tuple about spatial, map, or cartographic logic.
    All answers are one word, require spatial or map reasoning, and are not trivia.
    """
    qa = [
        ("If you face north and turn right, which direction are you facing?", "east"),
        ("On a standard map, what direction is to the left?", "west"),
        ("If you walk south, then west, then north, which direction must you walk to return?", "east"),
        ("A river flows from north to south. If you stand on the east bank, which way is west?", "across"),
        ("If you are in the center of a square and walk to a corner, what shape is your path?", "diagonal"),
        ("If you move from the bottom left to the top right of a map, which direction are you moving?", "northeast"),
        ("If you rotate a map 180 degrees, which direction does north point?", "south"),
        ("If you are facing east and turn left, which direction are you facing?", "north"),
        ("If you are at the intersection of two perpendicular roads, what angle do they form?", "right"),
        ("If you walk in a circle and end where you started, what is your displacement?", "zero"),
        ("If you fold a square map in half, what shape do you get?", "rectangle"),
        ("If you are at the north pole, which direction is south?", "all"),
        ("If you are in a rectangular room and walk from one corner to the opposite, what is the shortest path?", "diagonal"),
        ("If you face west and turn around, which direction are you facing?", "east"),
        ("If you are in the center of a compass rose, how many cardinal directions are there?", "four"),
        ("If you walk north, then east, then south, what direction is left to return?", "west"),
        ("If you are on a map and move up, which direction are you going?", "north"),
        ("If you are in a triangle and walk along all sides, how many turns do you make?", "three"),
        ("If you are at the equator and move north, what hemisphere are you in?", "northern"),
        ("If you are at the center of a circle and move to the edge, what is the path called?", "radius"),
        ("If you are at the intersection of two straight roads, what shape do they form?", "cross"),
        ("If you are in a cube and move from one corner to the farthest, what is the path called?", "diagonal"),
        ("If you are on a map and move down, which direction are you going?", "south"),
        ("If you are on a map and move right, which direction are you going?", "east"),
        ("If you are in a square and walk all four sides, what shape do you make?", "square"),
        ("If you are at the center of a clock and point to 3, which direction is that?", "east"),
        ("If you are at the center of a clock and point to 12, which direction is that?", "north"),
        ("If you are at the center of a clock and point to 6, which direction is that?", "south"),
        ("If you are at the center of a clock and point to 9, which direction is that?", "west"),
        ("If you are on a map and move left, which direction are you going?", "west"),
        ("If you are at the intersection of three roads, what is the shape called?", "t"),
        ("If you are in a rectangle and walk from one short side to the other, what is the path called?", "width"),
        ("If you are in a rectangle and walk from one long side to the other, what is the path called?", "length"),
        ("If you are at the center of a circle and draw a line to the edge, what is that line called?", "radius"),
        ("If you are at the edge of a circle and draw a line through the center to the other edge, what is that line called?", "diameter"),
        ("If you are in a triangle and walk from one corner to the midpoint of the opposite side, what is the path called?", "median"),
        ("If you are in a square and connect two opposite corners, what is the line called?", "diagonal"),
        ("If you are in a cube and connect two opposite corners, what is the line called?", "diagonal"),
        ("If you are in a pentagon and connect every other corner, what shape do you make?", "star"),
        ("If you are in a circle and walk all the way around, what is the path called?", "circumference"),
    ]
    return random.choice(qa)

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
    icon_size_px=24,
    n_icons=5,
    distances=None,
    icon_display_time_sec_tuple=(25, 15, 25),  # (icons, question, guess)
    results_path=None,
    screen=None,
    screen_width=None,
    screen_height=None
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
    question, correct_answer = get_easy_spatial_question()
    logging.info(f"| Selected spatial question: {question} (answer: {correct_answer})")

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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logging.info("| Received QUIT event")
                running = False
            elif event.type == pygame.KEYDOWN:
                if phase == "icons":
                    if event.key == pygame.K_RETURN:
                        logging.info("| Phase changed: icons -> question")
                        phase = "question"
                        input_text = ""
                        phase_start_time = time.time()
                        phase_end_time = phase_start_time + question_phase_duration
                    elif event.key == pygame.K_ESCAPE:
                        esc_counter += 1
                        if esc_counter >= 3:
                            logging.info("| Escape pressed 3 times, exiting ENTIRE SESSION")
                            return None  # <--- Return None to signal abort
                    else:
                        esc_counter = 0
                elif phase == "question":
                    if now < question_locked_until:
                        continue  # Ignore input while locked
                    if event.key == pygame.K_RETURN:
                        if input_text.strip().lower() == correct_answer.lower():
                            question_answer = input_text
                            logging.info(f"| Question answered correctly: {question_answer}")
                            input_text = ""
                            phase = "guess"
                            logging.info("| Phase changed: question -> guess")
                            phase_start_time = time.time()
                            phase_end_time = phase_start_time + guess_phase_duration
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
                            else:
                                question_locked_until = now + 3  # Lock input for 3 seconds
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
                logging.info("| Icon phase timed out, moving to question phase")
            # Draw timer at bottom
            instr = "Press Enter to continue, Esc x3 to exit"
            t_surf = font.render(f"{instr} | {timer_text}", True, (200, 200, 200))
            screen.blit(t_surf, (center_x - t_surf.get_width() // 2, screen_height - 60))
        elif phase == "question":
            # Draw question text
            qsurf = font.render(question, True, (255, 255, 255))
            screen.blit(qsurf, (center_x - qsurf.get_width() // 2, center_y - 100))
            # Draw input text
            insurf = input_font.render(input_text, True, (255, 255, 0))
            screen.blit(insurf, (center_x - insurf.get_width() // 2, center_y))
            # Show feedback if locked
            if now < question_locked_until:
                lock_msg = f"Incorrect! Try again in {int(question_locked_until - now)}s"
                locksurf = font.render(lock_msg, True, (255, 80, 80))
                screen.blit(locksurf, (center_x - locksurf.get_width() // 2, center_y + 60))
            # Timer: auto-advance
            if now >= phase_end_time:
                question_answer = input_text
                input_text = ""
                phase = "guess"
                phase_start_time = time.time()
                phase_end_time = phase_start_time + guess_phase_duration
                logging.info("| Question phase timed out, moving to guess phase")
            # Draw timer at bottom
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
    fields = [("Age", ""), ("Gender (M/F/D/O)", ""), ("Pseudonym or codeword of your choosing - be sure to remember it!", "")]
    field_idx = 0
    input_active = True
    input_text = ""
    esc_counter = 0

    while input_active:
        screen.fill((30, 30, 30))
        # Instructions
        instr = "Please enter your information. Press Enter to confirm each field."
        instr_surf = small_font.render(instr, True, (200, 200, 200))
        screen.blit(instr_surf, (screen_width // 2 - instr_surf.get_width() // 2, 80))
        # Draw fields
        for i, (label, value) in enumerate(fields):
            color = (255, 255, 0) if i == field_idx else (255, 255, 255)
            text = f"{label}: {value}" if i != field_idx else f"{label}: {input_text}"
            surf = font.render(text, True, color)
            screen.blit(surf, (screen_width // 2 - surf.get_width() // 2, 200 + i * 80))
        # Draw submit hint
        if field_idx >= len(fields):
            done_surf = font.render("Press Enter to continue...", True, (0, 255, 0))
            screen.blit(done_surf, (screen_width // 2 - done_surf.get_width() // 2, 200 + len(fields) * 80))
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
    if fields[1][1].strip() not in ["m", "f", "d", "o"]:
        logging.info("| Invalid gender input detected")
        raise ValueError("Invalid gender input. Please enter 'M' - Male, 'F' - Female, 'D' - Diverse, or 'O' - Other.")
    if not fields[2][1].strip():
        logging.info("| Empty codeword input detected")
        raise ValueError("Codeword cannot be empty. Please enter a valid codeword.")

    logging.info("| Participant info collected: Age=%s, Gender=%s, Codeword=%s", fields[0][1], fields[1][1], fields[2][1])
    return {
        "age": fields[0][1],
        "gender": fields[1][1],
        "codeword": fields[2][1]
    }

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

    # --- Experiment configuration (defaults) ---
    participant_name = user_data["codeword"] if user_data.get("codeword") else "unknown"  # Use codeword as participant name
    test_type = "circle"             # Test type (folder in /res/base/)
    debug = True                     # Enable debug overlay
    rebuild_svg = False              # Rebuild PNGs from SVGs
    log_file_level = logging.INFO    # Log file level
    log_console_level = logging.INFO # Log console level
    results_path = os.path.join(os.path.dirname(__file__), "results.csv")  # Path to results CSV file
    icon_display_time_sec_tuple = (25, 20, 15)  # Phase times in seconds (icons, question, guess)

    # --- Start single session ---
    test_list = ["circle", "circle", "circle", "circle", "circle", "circle", "circle", "square", "square", "square", "square", "square", "square", "square", ]

    for i, test_nr in enumerate(test_list):
        logging.info(f"Experiment {i+1}/{len(test_list)}: {test_nr} START")
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
            screen_height=screen_height
        )
        if exp_data is None:
            logging.warning("Experiment aborted by user (3x ESC). Aborting session.")
            break  # Abort the entire session loop
        # Save results to CSV
        merge_to_csv(exp_data, results_path, user_data)
        logging.info(f"Experiment {i+1}/{len(test_list)}: {test_nr} END")

    logging.info("=== SESSION END ===")
    pygame.quit()

if __name__ == "__main__":
    main()