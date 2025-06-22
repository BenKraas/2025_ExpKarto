from PyQt5 import QtWidgets, QtGui, QtCore, QtSvg
import sys
import os
import random
import math
import json
import time
import pygame

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
    import io
    from PIL import Image
    try:
        import cairosvg
    except ImportError:
        print("cairosvg is required for SVG to PNG conversion. Please install it.")
        sys.exit(1)

    icons_dir = os.path.join(os.path.dirname(__file__), "res", "icons")
    for fname in os.listdir(icons_dir):
        if fname.lower().endswith('.svg'):
            svg_path = os.path.join(icons_dir, fname)
            png_name = os.path.splitext(fname)[0] + ".png"
            png_path = os.path.join(icons_dir, png_name)
            if rebuild_svg or not os.path.exists(png_path):
                print(f"Converting {fname} to {png_name}...")
                png_bytes = cairosvg.svg2png(url=svg_path, output_width=png_res, output_height=png_res)
                pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
                pil_img.save(png_path)
            # else: print(f"PNG for {fname} already exists, skipping.")

def save_results(icon_coords, guesses, results_path, participant_name, test_type, n_test):
    # Match guesses to icons (nearest, one-to-one)
    icon_points = [(x, y, angle) for (x, y, angle, dist) in icon_coords]
    icon_attrs = [{"distance": dist, "angle": angle} for (x, y, angle, dist) in icon_coords]
    guesses_local = guesses[:]
    icon_to_guess = [None] * len(icon_points)
    guess_used = set()
    for i, (ix, iy, _) in enumerate(icon_points):
        min_dist = float('inf')
        min_j = None
        for j, (gx, gy) in enumerate(guesses_local):
            if j in guess_used:
                continue
            dist = math.hypot(ix - gx, iy - gy)
            if dist < min_dist:
                min_dist = dist
                min_j = j
        if min_j is not None:
            icon_to_guess[i] = min_j
            guess_used.add(min_j)
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
        else:
            icon_guess = None
            dist_to_icon = (None, None, None)
            euclidian = None
        exp_result[icon_label] = {
            "icon_pos": (icon_x, icon_y, icon_angle),
            "icon_guess": icon_guess,
            "dist_to_icon": dist_to_icon,
            "euclidian_dist": euclidian,
            "metadata": attrs
        }
    try:
        with open(results_path, "r") as f:
            data = json.load(f)
    except Exception:
        data = {}
    if participant_name not in data:
        data[participant_name] = {}
    if test_type not in data[participant_name]:
        data[participant_name][test_type] = {}
    exp_key = f"exp_{n_test}"
    data[participant_name][test_type][exp_key] = exp_result
    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)

def visualize_debug(results_path, participant_name, test_type, n_test, image_path, image_height_px, icon_size_px, screen, screen_width, screen_height):
    """
    Visualize the icons, guesses, and distances for the given experiment.
    """
    # Load results
    try:
        with open(results_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Could not load results: {e}")
        return

    exp_key = f"exp_{n_test}"
    try:
        exp_result = data[participant_name][test_type][exp_key]
    except Exception as e:
        print(f"Could not find experiment result: {e}")
        return

    # Load base image
    img = pygame.image.load(image_path).convert_alpha()
    img = pygame.transform.smoothscale(img, (image_height_px, image_height_px))
    center_x, center_y = screen_width // 2, screen_height // 2
    img_rect = img.get_rect(center=(center_x, center_y))

    # Prepare font
    font = pygame.font.SysFont(None, 28)

    # Gather icon and guess data
    icons = []
    guesses = []
    lines = []
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

    # Draw everything
    running = True
    while running:
        screen.fill((0, 0, 0))
        screen.blit(img, img_rect)

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
                pygame.draw.line(screen, (0, 255, 0), icon_pos, guess_pos, 2)
                # Label with distance
                mid_x = (icon_pos[0] + guess_pos[0]) // 2
                mid_y = (icon_pos[1] + guess_pos[1]) // 2
                label = f"{dist:.1f}" if dist is not None else "?"
                text_surf = font.render(label, True, (255, 255, 0))
                screen.blit(text_surf, (mid_x, mid_y))
        # Instructions
        instr = "Debug visualization: Press any key or close window to exit"
        isurf = font.render(instr, True, (200, 200, 200))
        screen.blit(isurf, (screen_width // 2 - isurf.get_width() // 2, screen_height - 40))

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                running = False
        pygame.time.wait(10)

def main(rebuild_svg=False, png_res=128, debug=False):
    ensure_svg_icons_converted_to_png(rebuild_svg=rebuild_svg, png_res=png_res)
    pygame.init()
    info = pygame.display.Info()
    screen_width, screen_height = info.current_w, info.current_h
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
    pygame.display.set_caption("Experiment")

    # --- Parameters ---
    image_path = "05_round_cross.png"
    image_height_px = int(screen_height * 0.7)
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

    # --- Timers ---
    icon_phase_start = time.time()
    icon_phase_end = icon_phase_start + icon_display_time_sec

    def save_results_local():
        save_results(icon_coords, guesses, results_path, participant_name, test_type, n_test)

    # --- Main loop ---
    while running:
        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if phase == "icons":
                    if event.key == pygame.K_RETURN:
                        phase = "question"
                        input_text = ""
                    elif event.key == pygame.K_ESCAPE:
                        esc_counter += 1
                        if esc_counter >= 3:
                            running = False
                    else:
                        esc_counter = 0
                elif phase == "question":
                    if event.key == pygame.K_RETURN:
                        question_answer = input_text
                        input_text = ""
                        phase = "guess"
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        esc_counter += 1
                        if esc_counter >= 3:
                            running = False
                    else:
                        esc_counter = 0
                        if event.unicode and len(event.unicode) == 1:
                            input_text += event.unicode
                elif phase == "guess":
                    if event.key == pygame.K_RETURN:
                        save_results_local()
                        running = False
                        # --- Debug visualization ---
                        if debug:
                            visualize_debug(
                                results_path, participant_name, test_type, n_test,
                                image_path, image_height_px, icon_size_px,
                                screen, screen_width, screen_height
                            )
                    elif event.key == pygame.K_ESCAPE:
                        esc_counter += 1
                        if esc_counter >= 3:
                            running = False
                    else:
                        esc_counter = 0
            elif event.type == pygame.MOUSEBUTTONDOWN and phase == "guess":
                mx, my = event.pos
                if event.button == 1:  # Left click to add
                    if len(guesses) < n_icons:
                        guesses.append((mx, my))
                        save_results_local()
                elif event.button == 3:  # Right click to remove nearest
                    if guesses:
                        dists = [math.hypot(mx - gx, my - gy) for gx, gy in guesses]
                        idx = dists.index(min(dists))
                        guesses.pop(idx)
                        save_results_local()

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
            # Draw guesses as red dots
            for gx, gy in guesses:
                pygame.draw.circle(screen, (255, 0, 0), (gx, gy), 7)
            # Draw instructions
            instr = "Left click: add guess, Right click: remove nearest, Enter: finish"
            isurf = font.render(instr, True, (200, 200, 200))
            screen.blit(isurf, (center_x - isurf.get_width() // 2, screen_height - 60))

        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    # Set png_res for icon quality, and rebuild_svg to force conversion if needed
    png_res = 128
    # Set debug=True to enable debug visualization after experiment
    main(rebuild_svg=False, png_res=png_res, debug=True)

# You cannot natively load SVGs directly with pygame.
# To use SVG icons in pygame, you must first convert them to raster images (e.g., PNG).
# This can be done ahead of time, or at runtime using a library like cairosvg or svglib+Pillow.

# Example: Convert SVG to PNG at runtime for pygame (requires cairosvg and Pillow)
# (Uncomment and use as needed)

# import io
# from PIL import Image
# import cairosvg

# def svg_to_surface(svg_path, size_px):
#     # Convert SVG to PNG bytes
#     png_bytes = cairosvg.svg2png(url=svg_path, output_width=size_px, output_height=size_px)
#     # Load PNG bytes into PIL Image
#     pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
#     # Convert PIL Image to pygame Surface
#     mode = pil_img.mode
#     size = pil_img.size
#     data = pil_img.tobytes()
#     return pygame.image.fromstring(data, size, mode)

# Then, in your icon loading:
# def load_icon_surfaces(icon_size_px):
#     icons_dir = os.path.join(os.path.dirname(__file__), "res", "icons")
#     icon_files = [f for f in os.listdir(icons_dir) if f.lower().endswith('.svg')]
#     assert icon_files, "No SVG icons found in /res/icons"
#     surfaces = []
#     for fname in icon_files:
#         surf = svg_to_surface(os.path.join(icons_dir, fname), icon_size_px)
#         surfaces.append(surf)
#     return surfaces
