from PyQt5 import QtWidgets, QtGui, QtCore, QtSvg
import sys
import os
import random
import math
import json
import time

def scatter_icons_around_center(parent_widget, center_x, center_y, distances, icon_size_px, n_items=5):
	"""
	Scatter SVG icons around a center point at specified distances.
	Returns a list of (x, y, attrs, label) for each icon.
	"""
	assert len(distances) > 0, "distances array must have at least 1 element"
	icons_dir = os.path.join(os.path.dirname(__file__), "res", "icons")
	icon_files = [f for f in os.listdir(icons_dir) if f.lower().endswith('.svg')]
	assert icon_files, "No SVG icons found in /res/icons"

	coords_and_attrs = []
	for i in range(n_items):
		dist = distances[i % len(distances)]
		angle = random.uniform(0, 2 * math.pi)
		x = center_x + dist * math.cos(angle)
		y = center_y + dist * math.sin(angle)
		icon_path = os.path.join(icons_dir, random.choice(icon_files))

		# Render SVG icon to pixmap
		svg_renderer = QtSvg.QSvgRenderer(icon_path)
		pixmap = QtGui.QPixmap(icon_size_px, icon_size_px)
		pixmap.fill(QtCore.Qt.transparent)
		painter = QtGui.QPainter(pixmap)
		svg_renderer.render(painter)
		painter.end()

		label = QtWidgets.QLabel(parent_widget)
		label.setPixmap(pixmap)
		label.setFixedSize(icon_size_px, icon_size_px)
		label.move(int(x - icon_size_px/2), int(y - icon_size_px/2))
		label.setAttribute(QtCore.Qt.WA_TranslucentBackground)
		label.show()

		coords_and_attrs.append((
			int(x), int(y),
			{
				"path": icon_path,
				"size": icon_size_px,
				"distance": dist,
				"angle": angle
			},
			label
		))
	return coords_and_attrs

def display_image_fullscreen(image_path, image_height_cm):
	"""
	Display an image in a fullscreen window, scaled to a given height in cm.
	Returns the window and the image QLabel.
	"""
	app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
	screen = app.primaryScreen()
	geometry = screen.geometry()
	screen_width_px = geometry.width()
	screen_height_px = geometry.height()
	physical_size_mm = screen.physicalSize()
	screen_height_mm = physical_size_mm.height()
	screen_width_mm = physical_size_mm.width()

	# Calculate pixels per cm using physical screen size
	if screen_height_mm > 0 and screen_width_mm > 0:
		px_per_cm_h = screen_height_px / (screen_height_mm / 10)
		px_per_cm_w = screen_width_px / (screen_width_mm / 10)
		px_per_cm = min(px_per_cm_h, px_per_cm_w)
	else:
		px_per_cm = 96 / 2.54  # fallback

	img_size_px = int(image_height_cm * px_per_cm)
	# Ensure image fits in both dimensions (square, max x=y)
	max_img_px = min(screen_width_px, screen_height_px)
	img_size_px = min(img_size_px, max_img_px)

	# Load and resize image
	pixmap = QtGui.QPixmap(image_path)
	pixmap = pixmap.scaled(img_size_px, img_size_px, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

	# Create fullscreen window
	window = QtWidgets.QWidget()
	window.setWindowFlags(QtCore.Qt.FramelessWindowHint)
	window.showFullScreen()
	window.setStyleSheet("background-color: black;")

	# --- UI Layout ---
	main_layout = QtWidgets.QHBoxLayout(window)
	main_layout.setContentsMargins(0, 0, 0, 0)
	main_layout.setSpacing(0)

	# Left panel
	left_panel = QtWidgets.QWidget(window)
	left_panel.setFixedWidth(220)
	left_panel.setStyleSheet("background: #222;")
	left_layout = QtWidgets.QVBoxLayout(left_panel)
	left_layout.setContentsMargins(20, 20, 20, 20)
	left_layout.setSpacing(20)

	timer_label = QtWidgets.QLabel("00:00", left_panel)
	timer_label.setStyleSheet("color: white; font-size: 32px; font-weight: bold;")
	timer_label.setAlignment(QtCore.Qt.AlignCenter)
	status_label = QtWidgets.QLabel("Status: Ready", left_panel)
	status_label.setStyleSheet("color: white; font-size: 18px;")
	status_label.setAlignment(QtCore.Qt.AlignCenter)

	left_layout.addWidget(timer_label)
	left_layout.addWidget(status_label)
	left_layout.addStretch(1)  # for future elements

	# Center panel (image)
	center_panel = QtWidgets.QWidget(window)
	center_layout = QtWidgets.QVBoxLayout(center_panel)
	center_layout.setContentsMargins(0, 0, 0, 0)
	center_layout.setSpacing(0)

	image_label = QtWidgets.QLabel(center_panel)
	image_label.setPixmap(pixmap)
	image_label.setAlignment(QtCore.Qt.AlignCenter)
	center_layout.addWidget(image_label)
	center_panel.setMinimumWidth(img_size_px)
	center_panel.setMinimumHeight(img_size_px)

	main_layout.addWidget(left_panel)
	main_layout.addWidget(center_panel)
	window.setLayout(main_layout)

	window.show()
	return window, image_label, timer_label, status_label

def main():
	"""
	Main entry point: shows image, scatters icons, handles key events.
	"""
	app = QtWidgets.QApplication(sys.argv)
	window, image_label, timer_label, status_label = display_image_fullscreen("05_round_cross.png", 35)

	def get_center():
		return window.width() // 2, window.height() // 2

	icon_params = dict(
		parent_widget=window,
		center_x=0,
		center_y=0,
		distances=[110, 220, 330, 440, 550],
		icon_size_px=36,
		n_items=5
	)
	icon_labels = [[]]
	esc_counter = [0]

	# --- Experiment state ---
	state = {
		"phase": "icons",
		"start_time": None,
		"icon_coords": None,
		"guesses": [],
		"guess_labels": [],
		"question_answer": "",
		"json_data": {}
	}
	icon_display_time_sec = 20  # set display time for icons (20 sec)
	participant_name = "P01"
	test_type = "occluded_circle"
	n_test = 1

	# --- Placeholder question widgets ---
	question_widget = QtWidgets.QWidget(window)
	question_layout = QtWidgets.QVBoxLayout(question_widget)
	question_label = QtWidgets.QLabel("Placeholder question: Please type anything and press Enter.")
	question_input = QtWidgets.QLineEdit()
	question_layout.addWidget(question_label)
	question_layout.addWidget(question_input)
	question_widget.hide()

	# --- Guessing phase ---
	def clear_guess_labels():
		for lbl in state["guess_labels"]:
			lbl.setParent(None)
			lbl.deleteLater()
		state["guess_labels"] = []

	def add_guess(x, y):
		dot = QtWidgets.QLabel(window)
		dot.setStyleSheet("background: red; border-radius: 7px;")
		dot.setFixedSize(14, 14)
		dot.move(int(x-7), int(y-7))
		dot.show()
		state["guess_labels"].append(dot)
		state["guesses"].append((x, y))
		save_results()  # Save after every click

	def remove_guess(x, y):
		if not state["guesses"]:
			return
		dists = [math.hypot(gx-x, gy-y) for gx, gy in state["guesses"]]
		idx = dists.index(min(dists))
		state["guess_labels"][idx].setParent(None)
		state["guess_labels"][idx].deleteLater()
		del state["guess_labels"][idx]
		del state["guesses"][idx]
		save_results()  # Save after every correction

	def save_results():
		icons = state["icon_coords"]
		guesses = state["guesses"]
		icon_points = [(x, y, attrs["angle"]) for x, y, attrs, _ in icons] if icons else []
		icon_attrs = [attrs for _, _, attrs, _ in icons] if icons else []

		# --- Match guesses to icons (one-to-one, nearest) ---
		if icons:
			icon_centers = [(x, y) for x, y, _, _ in icons]
			guess_points = guesses[:]
			assigned = set()
			icon_to_guess = [None] * len(icon_centers)
			guess_used = set()
			# For each icon, find nearest unused guess
			for i, (ix, iy) in enumerate(icon_centers):
				min_dist = float('inf')
				min_j = None
				for j, (gx, gy) in enumerate(guess_points):
					if j in guess_used:
						continue
					dist = math.hypot(ix - gx, iy - gy)
					if dist < min_dist:
						min_dist = dist
						min_j = j
				if min_j is not None:
					icon_to_guess[i] = min_j
					guess_used.add(min_j)
			# If more guesses than icons, extra guesses are ignored

		else:
			icon_to_guess = []

		# --- Build per-icon result dict ---
		exp_result = {}
		for idx, (icon, attrs) in enumerate(zip(icon_points, icon_attrs)):
			icon_x, icon_y, icon_angle = icon
			icon_label = f"icon_{idx+1}"
			guess_idx = icon_to_guess[idx] if icons and idx < len(icon_to_guess) else None
			if guess_idx is not None and guess_idx < len(guesses):
				guess_x, guess_y = guesses[guess_idx]
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

		# --- Save all metadata ---
		json_path = os.path.join(os.path.dirname(__file__), "results.json")
		try:
			with open(json_path, "r") as f:
				data = json.load(f)
		except Exception:
			data = {}

		if participant_name not in data:
			data[participant_name] = {}
		if test_type not in data[participant_name]:
			data[participant_name][test_type] = {}

		exp_key = f"exp_{n_test}"
		data[participant_name][test_type][exp_key] = exp_result

		with open(json_path, "w") as f:
			json.dump(data, f, indent=2)

	# --- Timer logic ---
	timer = QtCore.QTimer()
	timer.setInterval(100)
	timer_running = {"active": False, "end_time": 0}

	def update_timer():
		if not timer_running["active"]:
			return
		remaining = max(0, int(timer_running["end_time"] - time.time()))
		mins = remaining // 60
		secs = remaining % 60
		timer_label.setText(f"{mins:02d}:{secs:02d}")
		if remaining <= 0:
			timer.stop()
			timer_running["active"] = False

	timer.timeout.connect(update_timer)

	# --- Icon scatter phase ---
	def scatter_icons():
		for lbl in icon_labels[0]:
			lbl.setParent(None)
			lbl.deleteLater()
		icon_labels[0] = []
		center_x, center_y = get_center()
		icon_params['center_x'] = center_x
		icon_params['center_y'] = center_y
		coords = scatter_icons_around_center(**icon_params)
		icon_labels[0] = [item[3] for item in coords]
		state["icon_coords"] = coords
		status_label.setText("Status: Memorize icon locations")
		timer_label.setText(f"{icon_display_time_sec:02d}:00")

	def start_icon_phase():
		state["phase"] = "icons"
		state["guesses"] = []
		clear_guess_labels()
		image_label.show()
		for lbl in icon_labels[0]:
			lbl.show()
		question_widget.hide()
		window.setCursor(QtCore.Qt.ArrowCursor)
		scatter_icons()
		state["start_time"] = time.time()
		timer_running["active"] = True
		timer_running["end_time"] = time.time() + icon_display_time_sec
		timer.start()
		QtCore.QTimer.singleShot(icon_display_time_sec * 1000, start_question_phase)

	def start_question_phase():
		state["phase"] = "question"
		image_label.hide()
		for lbl in icon_labels[0]:
			lbl.hide()
		question_widget.show()
		question_input.setFocus()
		window.setCursor(QtCore.Qt.ArrowCursor)
		status_label.setText("Status: Answer the question")
		timer_running["active"] = False
		timer_label.setText("")

	def start_guess_phase():
		state["phase"] = "guess"
		question_widget.hide()
		image_label.show()
		for lbl in icon_labels[0]:
			lbl.hide()
		clear_guess_labels()
		state["guesses"] = []
		window.setCursor(QtCore.Qt.CrossCursor)
		status_label.setText("Status: Click to guess icon locations")
		timer_label.setText("")

	# --- Question input triggers guess phase ---
	def on_question_enter():
		state["question_answer"] = question_input.text()
		start_guess_phase()

	question_input.returnPressed.connect(on_question_enter)

	def mousePressEvent(event):
		if state["phase"] != "guess":
			return
		if event.button() == QtCore.Qt.LeftButton:
			add_guess(event.x(), event.y())
		elif event.button() == QtCore.Qt.RightButton:
			remove_guess(event.x(), event.y())

	def keyPressEvent(event):
		if event.key() == QtCore.Qt.Key_Escape:
			esc_counter[0] += 1
			if esc_counter[0] >= 3:
				window.close()
				app.quit()
		else:
			esc_counter[0] = 0

	window.keyPressEvent = keyPressEvent
	window.mousePressEvent = mousePressEvent

	center_panel = image_label.parent()
	center_panel.layout().addWidget(question_widget)

	QtCore.QTimer.singleShot(100, start_icon_phase)

	window.show()
	sys.exit(app.exec_())

if __name__ == "__main__":
	main()
