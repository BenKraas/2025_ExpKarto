from PyQt5 import QtWidgets, QtGui, QtCore, QtSvg
import sys
import os
import random
import math
import json
import time
import pygame
import ctypes
import io
from PIL import Image
import cairosvg
from scipy.optimize import linear_sum_assignment
import logging

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

def save_results(icon_coords, guesses, results_path, participant_name, test_type, n_test, center_x=None, center_y=None):
    # Match guesses to icons (optimal, one-to-one)
    icon_points = [(x, y, angle) for (x, y, angle, dist) in icon_coords]
    icon_attrs = []
    for (x, y, angle, dist) in icon_coords:
        x_c_dist = x - center_x if center_x is not None else None
        y_c_dist = y - center_y if center_y is not None else None
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
            guess_x_c_dist = guess_x - center_x if center_x is not None else None
            guess_y_c_dist = guess_y - center_y if center_y is not None else None
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
    try:
        with open(results_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logging.warning(f"Could not read results file: {e}")
        data = {}
    if participant_name not in data:
        data[participant_name] = {}
    if test_type not in data[participant_name]:
        data[participant_name][test_type] = {}
    exp_key = f"exp_{n_test}"
    data[participant_name][test_type][exp_key] = exp_result
    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"Results saved for participant={participant_name}, test_type={test_type}, n_test={n_test}")

def draw_debug_overlay(exp_result, screen, icon_size_px, font, center_x, center_y):
    """
    Draw debug overlay: icons, guesses, lines, and distances, including cardinals.
    """
    import pygame  # Delayed import
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

def main(rebuild_svg=False, png_res=128, debug=False, log_file_level=logging.DEBUG, log_console_level=logging.INFO):
    # --- Logging ---
    log_path = os.path.join(os.path.dirname(__file__), "experiment.log")
    setup_logging(log_path, file_level=log_file_level, console_level=log_console_level)
    logging.info("Experiment started")
    ensure_svg_icons_converted_to_png(rebuild_svg=rebuild_svg, png_res=png_res)
    pygame.init()
    info = pygame.display.Info()
    screen_width, screen_height = info.current_w, info.current_h
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
    pygame.display.set_caption("Experiment")
    logging.info(f"Screen size: {screen_width}x{screen_height}")

    # --- Parameters ---
    image_path = "05_round_cross.png"

    # --- Calculate image height in pixels for 30 cm ---
    # Try to get DPI from system, otherwise use a common default (96)
    try:
        # Windows: use ctypes to get system DPI
        if sys.platform.startswith("win"):
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            dpi = user32.GetDpiForSystem()
        else:
            # Fallback for non-Windows systems
            dpi = 96
    except Exception:
        dpi = 96  # fallback

    cm_per_inch = 2.54
    image_height_cm = 30
    image_height_px = int((image_height_cm / cm_per_inch) * dpi)

    icon_size_px = 24
    n_icons = 5
    distances = [110, 220, 330, 440, 550]
    icon_display_time_sec = 20
    participant_name = "P01"
    test_type = "occluded_circle"
    n_test = 1
    results_path = os.path.join(os.path.dirname(__file__), "results.json")

    # --- Load resources ---
    img = display_image_fullscreen(image_path, image_height_px)
    icon_surfaces = load_icon_surfaces(icon_size_px)
    logging.info(f"Loaded {len(icon_surfaces)} icon surfaces")

    # --- State ---
    phase = "icons"
    icon_coords = []
    icon_assignments = []
    guesses = []
    question_answer = ""
    running = True
    font = pygame.font.SysFont(None, 36)
    input_font = pygame.font.SysFont(None, 48)
    question = "Placeholder question: Please type anything and press Enter."
    input_text = ""
    esc_counter = 0

    # --- Icon scatter ---
    center_x, center_y = screen_width // 2, screen_height // 2
    icon_coords = scatter_icons_around_center(center_x, center_y, distances, n_icons)
    icon_assignments = [random.choice(icon_surfaces) for _ in range(n_icons)]
    logging.debug(f"Icon coordinates: {icon_coords}")

    # --- Timers ---
    icon_phase_start = time.time()
    icon_phase_end = icon_phase_start + icon_display_time_sec

    def save_results_local():
        save_results(icon_coords, guesses, results_path, participant_name, test_type, n_test, center_x=center_x, center_y=center_y)

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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logging.info("Received QUIT event")
                running = False
            elif event.type == pygame.KEYDOWN:
                if phase == "icons":
                    if event.key == pygame.K_RETURN:
                        logging.info("Phase changed: icons -> question")
                        phase = "question"
                        input_text = ""
                    elif event.key == pygame.K_ESCAPE:
                        esc_counter += 1
                        if esc_counter >= 3:
                            logging.info("Escape pressed 3 times, exiting")
                            running = False
                    else:
                        esc_counter = 0
                elif phase == "question":
                    if event.key == pygame.K_RETURN:
                        question_answer = input_text
                        logging.info(f"Question answered: {question_answer}")
                        input_text = ""
                        phase = "guess"
                        logging.info("Phase changed: question -> guess")
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        esc_counter += 1
                        if esc_counter >= 3:
                            logging.info("Escape pressed 3 times, exiting")
                            running = False
                    else:
                        esc_counter = 0
                        if event.unicode and len(event.unicode) == 1:
                            input_text += event.unicode
                elif phase == "guess":
                    if event.key == pygame.K_RETURN:
                        save_results_local()
                        logging.info("Experiment finished, exiting")
                        running = False
                    elif event.key == pygame.K_ESCAPE:
                        esc_counter += 1
                        if esc_counter >= 3:
                            logging.info("Escape pressed 3 times, exiting")
                            running = False
                    else:
                        esc_counter = 0
            elif event.type == pygame.MOUSEBUTTONDOWN and phase == "guess":
                mx, my = event.pos
                if event.button == 1:  # Left click to add
                    if len(guesses) < n_icons:
                        guesses.append((mx, my))
                        save_results_local()
                        logging.info(f"Guess added at ({mx}, {my})")
                elif event.button == 3:  # Right click to remove nearest
                    if guesses:
                        dists = [math.hypot(mx - gx, my - gy) for gx, gy in guesses]
                        idx = dists.index(min(dists))
                        removed = guesses.pop(idx)
                        save_results_local()
                        logging.info(f"Guess removed at {removed}")

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
            if time.time() >= icon_phase_end:
                phase = "question"
                input_text = ""
                logging.info("Icon phase timed out, moving to question phase")
        elif phase == "question":
            # Draw question text
            qsurf = font.render(question, True, (255, 255, 255))
            screen.blit(qsurf, (center_x - qsurf.get_width() // 2, center_y - 100))
            # Draw input text
            insurf = input_font.render(input_text, True, (255, 255, 0))
            screen.blit(insurf, (center_x - insurf.get_width() // 2, center_y))
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
            # Draw instructions
            instr = "Left click: add guess, Right click: remove nearest, Enter: finish"
            isurf = font.render(instr, True, (200, 200, 200))
            screen.blit(isurf, (center_x - isurf.get_width() // 2, screen_height - 60))

        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()
    logging.info("Pygame quit, program exiting")
    sys.exit()

if __name__ == "__main__":
    # Set png_res for icon quality, and rebuild_svg to force conversion if needed
    png_res = 128
    # Set debug=True to enable debug visualization after experiment
    # Set log levels as needed
    main(rebuild_svg=False, png_res=png_res, debug=True,
         log_file_level=logging.DEBUG, log_console_level=logging.INFO)