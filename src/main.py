# minesweeper_ai_continued.py
# Continued from user's version: added minesweeper.online color palette support,
# adjustable HSV/color thresholds via UI, and a "Show Grid" preview of captured region.

import pyautogui
import numpy as np
import cv2
from PIL import Image, ImageGrab, ImageDraw, ImageFont, ImageTk
import time
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict
import sys


@dataclass
class Cell:
    x: int
    y: int
    value: int  # -2: unknown, -1: mine, 0-8: number
    screen_x: int
    screen_y: int
    probability: float = 0.0


class CellDetector:
    """Computer Vision untuk deteksi state dan angka pada sel Minesweeper

    This detector uses HSV ranges tuned toward minesweeper.online colors, and
    allows runtime threshold adjustments from the GUI (passed in via `params`).
    """

    def __init__(self, params: Dict[str, float] = None):
        # Default parameters (can be updated by GUI sliders)
        defaults = {
            'unopened_h_low': 0, 'unopened_s_low': 0, 'unopened_v_low': 150,
            'unopened_h_high': 180, 'unopened_s_high': 40, 'unopened_v_high': 255,
            'flag_sensitivity': 0.05,
            'number_color_tol': 40,
            'unopened_ratio_threshold': 0.45,
            'flag_ratio_threshold': 0.03,
        }
        self.params = defaults
        if params:
            self.params.update(params)

        # Minesweeper.online number colors (BGR approximate values)
        # These are approximate; GUI sliders allow tolerance adjustment.
        self.number_colors = {
            1: ([25, 60, 250], 'blue'),      # 1 - blue
            2: ([35, 180, 60], 'green'),     # 2 - green
            3: ([50, 50, 200], 'red'),       # 3 - redish
            4: ([150, 90, 40], 'darkblue'),  # 4 - dark blue
            5: ([20, 30, 130], 'maroon'),    # 5 - maroon
            6: ([200, 200, 60], 'cyan'),     # 6 - cyan-ish
            7: ([80, 30, 120], 'blackish'),  # 7 - near-black/purple
            8: ([120, 120, 120], 'gray'),    # 8 - gray
        }

        # Flag HSV ranges typical for minesweeper.online (red flag)
        # We keep two ranges for red wrap-around
        self.flag_ranges = [
            (np.array([0, 100, 80]), np.array([10, 255, 255])),
            (np.array([170, 100, 80]), np.array([180, 255, 255])),
        ]

    def update_params(self, new_params: Dict[str, float]):
        self.params.update(new_params)

    def capture_cell(self, x: int, y: int, size: int) -> np.ndarray:
        """Capture screenshot sel tunggal (BGR)"""
        screenshot = ImageGrab.grab(bbox=(x, y, x + size, y + size))
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    def is_unopened(self, img: np.ndarray) -> bool:
        """Deteksi apakah sel masih tertutup (unopened) menggunakan HSV mask

        Uses dynamic thresholds from self.params.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        low = np.array([
            int(self.params['unopened_h_low']),
            int(self.params['unopened_s_low']),
            int(self.params['unopened_v_low'])
        ])
        high = np.array([
            int(self.params['unopened_h_high']),
            int(self.params['unopened_s_high']),
            int(self.params['unopened_v_high'])
        ])

        mask = cv2.inRange(hsv, low, high)
        ratio = np.sum(mask > 0) / mask.size
        return ratio > float(self.params['unopened_ratio_threshold'])

    def is_flagged(self, img: np.ndarray) -> bool:
        """Deteksi apakah sel sudah di-flag (merah)"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in self.flag_ranges:
            mask = cv2.inRange(hsv, low, high)
            mask_total = cv2.bitwise_or(mask_total, mask)

        ratio = np.sum(mask_total > 0) / mask_total.size
        return ratio > float(self.params['flag_ratio_threshold'])

    def detect_number(self, img: np.ndarray) -> int:
        """Deteksi angka pada sel yang sudah dibuka.

        Strategy:
        1. Quick test for empty (uniform bright cell)
        2. Color matching against self.number_colors using BGR tolerance
        3. Fallback OCR-like heuristics with contours/edges
        """
        # Preprocess: small blur to reduce noise
        bgr = img.copy()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # Empty cell detection: very bright & low contrast
        mean_val = np.mean(gray_blur)
        std_val = np.std(gray_blur)
        if mean_val > 200 and std_val < 20:
            return 0

        # Color matching
        tol = int(self.params['number_color_tol'])
        for number, (bgr_color, _name) in self.number_colors.items():
            color_arr = np.array(bgr_color, dtype=int)
            diff = np.abs(bgr.astype(int) - color_arr)
            mask = np.all(diff <= tol, axis=2)
            ratio = np.sum(mask) / mask.size
            if ratio > 0.015:  # 1.5% of pixels match
                return number

        # Fallback: try simple template like detection by contours (better than nothing)
        edges = cv2.Canny(gray_blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Heuristic mapping by total contour length
        total_len = sum(cv2.arcLength(cnt, True) for cnt in contours)

        if total_len < 20:
            return 0
        elif total_len < 60:
            return 1
        elif total_len < 130:
            return 2
        elif total_len < 210:
            return 3

        return -2  # unknown

    def get_cell_state(self, x: int, y: int, size: int) -> Tuple[str, int]:
        """Deteksi state dan nilai sel"""
        img = self.capture_cell(x, y, size)

        if self.is_flagged(img):
            return ('flagged', -1)

        if self.is_unopened(img):
            return ('unopened', -2)

        number = self.detect_number(img)
        return ('opened', number)


class RegionSelector:
    """Window untuk select region dengan drag & drop"""

    def __init__(self, callback):
        self.callback = callback
        self.root = tk.Toplevel()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-alpha', 0.25)
        self.root.attributes('-topmost', True)
        self.root.config(cursor='cross')

        self.canvas = tk.Canvas(self.root, highlightthickness=0, bg='black')
        self.canvas.pack(fill='both', expand=True)

        self.start_x = None
        self.start_y = None
        self.rect = None

        self.instruction = self.canvas.create_text(
            self.root.winfo_screenwidth() // 2, 50,
            text="DRAG untuk select region papan Minesweeper\nTekan ESC untuk cancel",
            fill='white', font=('Arial', 16, 'bold'), justify='center'
        )

        self.canvas.bind('<Button-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
        self.root.bind('<Escape>', lambda e: self.cancel())

    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline='red', width=3
        )

    def on_drag(self, event):
        if self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)

        width = x2 - x1
        height = y2 - y1

        if width > 20 and height > 20:
            self.callback(x1, y1, width, height)
            self.root.destroy()
        else:
            messagebox.showwarning("Region Terlalu Kecil", "Drag area yang lebih besar!")
            if self.rect:
                self.canvas.delete(self.rect)
                self.rect = None

    def cancel(self):
        self.root.destroy()


class MinesweeperSolver:
    """Unchanged solver logic (kept same as user but included for completeness)"""

    def __init__(self, board_width: int, board_height: int, total_mines: int):
        self.width = board_width
        self.height = board_height
        self.total_mines = total_mines
        self.board = np.full((board_height, board_width), -2, dtype=int)
        self.flagged = set()
        self.safe_cells = set()

    def get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.height and 0 <= nj < self.width:
                    neighbors.append((ni, nj))
        return neighbors

    def update_cell(self, i: int, j: int, value: int):
        self.board[i, j] = value

    def solve_logic(self) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        safe = set()
        mines = set()
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i, j] >= 0:
                    neighbors = self.get_neighbors(i, j)
                    unknown_neighbors = [(ni, nj) for ni, nj in neighbors if self.board[ni, nj] == -2]
                    flagged_neighbors = [(ni, nj) for ni, nj in neighbors if (ni, nj) in self.flagged]
                    remaining_mines = self.board[i, j] - len(flagged_neighbors)
                    if len(unknown_neighbors) == remaining_mines and remaining_mines > 0:
                        mines.update(unknown_neighbors)
                    elif remaining_mines == 0:
                        safe.update(unknown_neighbors)
        return safe, mines

    def calculate_probability(self) -> dict:
        probabilities = {}
        total_flagged = len(self.flagged)
        remaining_mines = self.total_mines - total_flagged
        unknown_cells = []
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i, j] == -2 and (i, j) not in self.flagged:
                    unknown_cells.append((i, j))
        total_unknown = len(unknown_cells)
        if total_unknown == 0:
            return probabilities
        global_prob = remaining_mines / total_unknown if total_unknown > 0 else 0
        for i, j in unknown_cells:
            neighbors = self.get_neighbors(i, j)
            local_probs = []
            for ni, nj in neighbors:
                if self.board[ni, nj] >= 0:
                    nei_neighbors = self.get_neighbors(ni, nj)
                    unknown_nei = [n for n in nei_neighbors if self.board[n[0], n[1]] == -2]
                    flagged_nei = [n for n in nei_neighbors if n in self.flagged]
                    remaining = self.board[ni, nj] - len(flagged_nei)
                    u = len(unknown_nei)
                    if u > 0:
                        local_prob = remaining / u
                        local_probs.append(local_prob)
            if local_probs:
                probabilities[(i, j)] = sum(local_probs) / len(local_probs)
            else:
                probabilities[(i, j)] = global_prob
        return probabilities

    def get_best_move(self) -> Optional[Tuple[int, int, str]]:
        safe, mines = self.solve_logic()
        if safe:
            cell = safe.pop()
            return (cell[0], cell[1], 'click')
        if mines:
            cell = mines.pop()
            self.flagged.add(cell)
            return (cell[0], cell[1], 'flag')
        probabilities = self.calculate_probability()
        if not probabilities:
            return None
        best_cell = min(probabilities.items(), key=lambda x: x[1])
        return (best_cell[0][0], best_cell[0][1], 'click')


class MinesweeperGUI:
    """Main GUI extended with adjustable detector params and grid preview"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Minesweeper AI Helper (minesweeper.online palette)")
        self.root.geometry("520x820")
        self.root.attributes('-topmost', True)

        # detector parameters
        self.detector_params = {
            'unopened_h_low': 0,
            'unopened_s_low': 0,
            'unopened_v_low': 150,
            'unopened_h_high': 180,
            'unopened_s_high': 50,
            'unopened_v_high': 255,
            'unopened_ratio_threshold': 0.45,
            'flag_ratio_threshold': 0.03,
            'number_color_tol': 40,
        }

        self.detector = CellDetector(self.detector_params.copy())
        self.solver = None
        self.overlay_window = None
        self.running = False
        self.autosolve_thread = None
        self.overlay_thread = None
        self.region_data = {'x': 100, 'y': 100, 'width': 144, 'height': 144, 'cell_size': 16}

        self.setup_ui()

    def setup_ui(self):
        config_frame = ttk.LabelFrame(self.root, text="Konfigurasi Papan", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(config_frame, text="Lebar:").grid(row=0, column=0, sticky='w', padx=5)
        self.width_var = tk.IntVar(value=9)
        ttk.Entry(config_frame, textvariable=self.width_var, width=10).grid(row=0, column=1, padx=5)

        ttk.Label(config_frame, text="Tinggi:").grid(row=0, column=2, sticky='w', padx=5)
        self.height_var = tk.IntVar(value=9)
        ttk.Entry(config_frame, textvariable=self.height_var, width=10).grid(row=0, column=3, padx=5)

        ttk.Label(config_frame, text="Total Ranjau:").grid(row=1, column=0, sticky='w', padx=5)
        self.mines_var = tk.IntVar(value=10)
        ttk.Entry(config_frame, textvariable=self.mines_var, width=10).grid(row=1, column=1, padx=5)

        ttk.Label(config_frame, text="Cell Size:").grid(row=1, column=2, sticky='w', padx=5)
        self.cell_size_var = tk.IntVar(value=16)
        ttk.Entry(config_frame, textvariable=self.cell_size_var, width=10).grid(row=1, column=3, padx=5)

        region_frame = ttk.LabelFrame(self.root, text="Region Papan", padding=10)
        region_frame.pack(fill='x', padx=10, pady=5)

        select_btn = ttk.Button(region_frame, text="🎯 Select Region (Drag & Drop)", command=self.select_region_interactive, width=35)
        select_btn.pack(pady=5)

        self.region_info_label = tk.Label(region_frame, text="Region: Belum diset\nKlik tombol di atas untuk select", justify='left', bg='#f0f0f0', padx=10, pady=10)
        self.region_info_label.pack(fill='x', pady=5)

        mode_frame = ttk.LabelFrame(self.root, text="Mode", padding=10)
        mode_frame.pack(fill='x', padx=10, pady=5)

        self.auto_btn = ttk.Button(mode_frame, text="🤖 Auto Solve (CV)", command=self.start_autosolve, width=35)
        self.auto_btn.pack(pady=3)

        self.overlay_btn = ttk.Button(mode_frame, text="👁️ Guide Overlay (CV)", command=self.start_overlay, width=35)
        self.overlay_btn.pack(pady=3)

        self.scan_btn = ttk.Button(mode_frame, text="🔍 Test: Scan Board", command=self.test_scan_board, width=35)
        self.scan_btn.pack(pady=3)

        # New: Show Grid preview
        self.grid_btn = ttk.Button(mode_frame, text="🧭 Show Grid Preview", command=self.show_grid_preview, width=35)
        self.grid_btn.pack(pady=3)

        self.stop_btn = ttk.Button(mode_frame, text="⛔ Stop", command=self.stop, width=35, state='disabled')
        self.stop_btn.pack(pady=3)

        # Detector parameter sliders
        param_frame = ttk.LabelFrame(self.root, text="Detector - Adjust Thresholds", padding=10)
        param_frame.pack(fill='x', padx=10, pady=5)

        # Unopened V low
        ttk.Label(param_frame, text="Unopened V low:").grid(row=0, column=0, sticky='w')
        self.vlow_var = tk.IntVar(value=self.detector_params['unopened_v_low'])
        ttk.Scale(param_frame, from_=0, to=255, variable=self.vlow_var, command=lambda e: self.param_changed()).grid(row=0, column=1, sticky='we', padx=5)

        ttk.Label(param_frame, text="Unopened S high:").grid(row=1, column=0, sticky='w')
        self.shigh_var = tk.IntVar(value=self.detector_params['unopened_s_high'])
        ttk.Scale(param_frame, from_=0, to=255, variable=self.shigh_var, command=lambda e: self.param_changed()).grid(row=1, column=1, sticky='we', padx=5)

        ttk.Label(param_frame, text="Unopened ratio:").grid(row=2, column=0, sticky='w')
        self.unopened_ratio_var = tk.DoubleVar(value=self.detector_params['unopened_ratio_threshold'])
        ttk.Scale(param_frame, from_=0.0, to=1.0, variable=self.unopened_ratio_var, command=lambda e: self.param_changed()).grid(row=2, column=1, sticky='we', padx=5)

        ttk.Label(param_frame, text="Flag ratio:").grid(row=3, column=0, sticky='w')
        self.flag_ratio_var = tk.DoubleVar(value=self.detector_params['flag_ratio_threshold'])
        ttk.Scale(param_frame, from_=0.0, to=0.5, variable=self.flag_ratio_var, command=lambda e: self.param_changed()).grid(row=3, column=1, sticky='we', padx=5)

        ttk.Label(param_frame, text="Number color tolerance:").grid(row=4, column=0, sticky='w')
        self.num_tol_var = tk.IntVar(value=self.detector_params['number_color_tol'])
        ttk.Scale(param_frame, from_=5, to=120, variable=self.num_tol_var, command=lambda e: self.param_changed()).grid(row=4, column=1, sticky='we', padx=5)

        # Status & log
        status_frame = ttk.LabelFrame(self.root, text="Status & Log", padding=10)
        status_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.log_text = tk.Text(status_frame, height=14, width=60, wrap='word')
        self.log_text.pack(side='left', fill='both', expand=True)
        scrollbar = ttk.Scrollbar(status_frame, command=self.log_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.log_text.config(yscrollcommand=scrollbar.set)

        info_frame = ttk.Frame(self.root)
        info_frame.pack(fill='x', padx=10, pady=5)
        info_text = """CARA PAKAI:
1. Buka Minesweeper (minesweeper.online)
2. Klik "Select Region" → drag area papan
3. Set ukuran board & jumlah mines
4. Adjust thresholds jika perlu, klik "Show Grid Preview" untuk memeriksa
5. Test "Scan Board" sebelum auto solve
"""
        ttk.Label(info_frame, text=info_text, justify='left', font=('Arial', 8), foreground='#666').pack()

    def log(self, message: str):
        self.log_text.insert('end', f"{message}\n")
        self.log_text.see('end')
        self.root.update()

    def select_region_interactive(self):
        self.log("Minimalisir window ini, lalu DRAG area papan...")

        def on_region_selected(x, y, width, height):
            self.region_data['x'] = x
            self.region_data['y'] = y
            self.region_data['width'] = width
            self.region_data['height'] = height

            board_w = self.width_var.get()
            board_h = self.height_var.get()
            cell_size = int((width / board_w + height / board_h) / 2)

            self.region_data['cell_size'] = cell_size
            self.cell_size_var.set(cell_size)

            info_text = f"""✅ Region berhasil diset!\nX: {x}, Y: {y}\nWidth: {width}px, Height: {height}px\nCell Size: {cell_size}px (auto-detected)"""
            self.region_info_label.config(text=info_text, bg='#d4edda')
            self.log(f"✅ Region: ({x},{y}) {width}x{height}, Cell: {cell_size}px")

        self.root.withdraw()
        time.sleep(0.25)
        selector = RegionSelector(on_region_selected)
        self.root.wait_window(selector.root)
        self.root.deiconify()

    def param_changed(self):
        newp = {
            'unopened_v_low': int(self.vlow_var.get()),
            'unopened_s_high': int(self.shigh_var.get()),
            'unopened_ratio_threshold': float(self.unopened_ratio_var.get()),
            'flag_ratio_threshold': float(self.flag_ratio_var.get()),
            'number_color_tol': int(self.num_tol_var.get()),
        }
        self.detector.update_params(newp)
        self.log(f"🔧 Updated detector params: {newp}")

    def scan_board_state(self) -> bool:
        try:
            region_x = self.region_data['x']
            region_y = self.region_data['y']
            cell_size = self.region_data['cell_size']

            updated_count = 0
            width = self.width_var.get()
            height = self.height_var.get()

            for i in range(height):
                for j in range(width):
                    cell_x = region_x + j * cell_size
                    cell_y = region_y + i * cell_size
                    state, value = self.detector.get_cell_state(cell_x, cell_y, cell_size)
                    if state == 'flagged':
                        self.solver.flagged.add((i, j))
                        self.solver.update_cell(i, j, -1)
                        updated_count += 1
                    elif state == 'opened' and value >= 0:
                        if self.solver.board[i, j] != value:
                            self.solver.update_cell(i, j, value)
                            updated_count += 1
            return updated_count > 0
        except Exception as e:
            self.log(f"❌ Scan error: {e}")
            return False

    def test_scan_board(self):
        if self.region_data['width'] < 50:
            messagebox.showwarning("Region Belum Diset", "Select region terlebih dahulu!")
            return
        self.log("=" * 40)
        self.log("🔍 TEST SCAN BOARD")
        self.log("=" * 40)
        width = self.width_var.get()
        height = self.height_var.get()
        mines = self.mines_var.get()
        self.solver = MinesweeperSolver(width, height, mines)
        self.log("📷 Scanning board state...")
        success = self.scan_board_state()
        if success:
            self.log("✅ Scan berhasil!")
            self.log("\n📊 Board State:")
            for i in range(height):
                row_str = ""
                for j in range(width):
                    val = self.solver.board[i, j]
                    if val == -2:
                        row_str += "? "
                    elif val == -1:
                        row_str += "F "
                    else:
                        row_str += f"{val} "
                self.log(row_str)
        else:
            self.log("⚠️ Tidak ada perubahan terdeteksi")

    def show_grid_preview(self):
        """Capture the entire selected region and show each cell image in a preview window

        This helps the user tune thresholds visually.
        """
        if self.region_data['width'] < 50:
            messagebox.showwarning("Region Belum Diset", "Select region terlebih dahulu!")
            return

        region_x = self.region_data['x']
        region_y = self.region_data['y']
        cell_size = self.region_data['cell_size']
        width = self.width_var.get()
        height = self.height_var.get()

        preview = tk.Toplevel(self.root)
        preview.title("Grid Preview")
        canvas = tk.Canvas(preview, width=cell_size * width + 20, height=cell_size * height + 60)
        canvas.pack()

        imgs = []
        for i in range(height):
            for j in range(width):
                x = region_x + j * cell_size
                y = region_y + i * cell_size
                try:
                    arr = self.detector.capture_cell(x, y, cell_size)
                    pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
                    pil = pil.resize((cell_size, cell_size), Image.NEAREST)
                    tkimg = ImageTk.PhotoImage(pil)
                    imgs.append(tkimg)  # keep reference
                    canvas.create_image(j * cell_size + 10, i * cell_size + 10, anchor='nw', image=tkimg)

                    # overlay detection text
                    state, value = self.detector.get_cell_state(x, y, cell_size)
                    txt = '?'
                    if state == 'flagged':
                        txt = 'F'
                    elif state == 'unopened':
                        txt = 'U'
                    elif state == 'opened' and value >= 0:
                        txt = str(value)
                    canvas.create_text(j * cell_size + 6 + cell_size // 2, i * cell_size + 6 + cell_size // 2, text=txt, fill='yellow', font=('Arial', 8, 'bold'))
                except Exception as e:
                    canvas.create_rectangle(j * cell_size + 10, i * cell_size + 10, j * cell_size + 10 + cell_size, i * cell_size + 10 + cell_size, outline='red')
                    canvas.create_text(j * cell_size + 10 + cell_size//2, i * cell_size + 10 + cell_size//2, text='ERR', fill='red')

        # Info
        canvas.create_text(cell_size * width // 2, cell_size * height + 30, text='Preview captured cells (U=unopened, F=flag, numbers=open)', fill='white')

        # Keep window on top briefly
        preview.attributes('-topmost', True)
        preview.after(3000, lambda: preview.attributes('-topmost', False))

    def start_autosolve(self):
        if self.running:
            self.log('⚠️ Program sudah berjalan!')
            return
        if self.region_data['width'] < 50:
            messagebox.showwarning('Region Belum Diset', 'Select region terlebih dahulu!')
            return
        self.running = True
        self.auto_btn.config(state='disabled')
        self.overlay_btn.config(state='disabled')
        self.scan_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.log('=' * 40)
        self.log('🤖 MODE: AUTO SOLVE (CV)')
        self.log('=' * 40)
        width = self.width_var.get()
        height = self.height_var.get()
        mines = self.mines_var.get()
        self.solver = MinesweeperSolver(width, height, mines)
        self.log(f'Board: {width}x{height}, Mines: {mines}')
        self.autosolve_thread = threading.Thread(target=self.autosolve_loop, daemon=True)
        self.autosolve_thread.start()

    def autosolve_loop(self):
        try:
            region_x = self.region_data['x']
            region_y = self.region_data['y']
            cell_size = self.region_data['cell_size']
            self.log('⏳ Memulai dalam 3 detik...')
            for i in range(3, 0, -1):
                if not self.running:
                    return
                self.log(f'   {i}...')
                time.sleep(1)
            self.log('▶️ Mulai!')
            move_count = 0
            first_click = False
            while self.running:
                self.log('📷 Scanning...')
                self.scan_board_state()
                if not first_click:
                    mid_i = self.height_var.get() // 2
                    mid_j = self.width_var.get() // 2
                    click_x = region_x + mid_j * cell_size + cell_size // 2
                    click_y = region_y + mid_i * cell_size + cell_size // 2
                    self.log(f'🎯 Klik pertama: ({mid_i},{mid_j})')
                    pyautogui.click(click_x, click_y)
                    time.sleep(0.8)
                    first_click = True
                    continue
                move = self.solver.get_best_move()
                if move is None:
                    self.log('✅ Tidak ada langkah yang bisa dihitung')
                    self.log('🎉 Game mungkin sudah selesai!')
                    break
                i, j, action = move
                click_x = region_x + j * cell_size + cell_size // 2
                click_y = region_y + i * cell_size + cell_size // 2
                probs = self.solver.calculate_probability()
                prob = probs.get((i, j), 0)
                if action == 'click':
                    self.log(f'✅ Klik ({i},{j}) - P(mine)={prob:.1%}')
                    pyautogui.click(click_x, click_y)
                elif action == 'flag':
                    self.log(f'🚩 Flag ({i},{j}) - Mine terdeteksi')
                    pyautogui.rightClick(click_x, click_y)
                move_count += 1
                time.sleep(0.8)
                if move_count > 200:
                    self.log('⚠️ Limit 200 moves tercapai')
                    break
        except Exception as e:
            self.log(f'❌ Error: {e}')
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.running = False
            self.auto_btn.config(state='normal')
            self.overlay_btn.config(state='normal')
            self.scan_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.log('⏹️ Auto solve berhenti')

    def start_overlay(self):
        if self.running:
            self.log('⚠️ Program sudah berjalan!')
            return
        if self.region_data['width'] < 50:
            messagebox.showwarning('Region Belum Diset', 'Select region terlebih dahulu!')
            return
        self.running = True
        self.auto_btn.config(state='disabled')
        self.overlay_btn.config(state='disabled')
        self.scan_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.log('=' * 40)
        self.log('👁️ MODE: GUIDE OVERLAY (CV)')
        self.log('=' * 40)
        width = self.width_var.get()
        height = self.height_var.get()
        mines = self.mines_var.get()
        self.solver = MinesweeperSolver(width, height, mines)
        self.log(f'Board: {width}x{height}, Mines: {mines}')
        self.create_overlay()
        self.overlay_thread = threading.Thread(target=self.overlay_loop, daemon=True)
        self.overlay_thread.start()

    def create_overlay(self):
        self.overlay_window = tk.Toplevel(self.root)
        self.overlay_window.attributes('-alpha', 0.7)
        self.overlay_window.attributes('-topmost', True)
        self.overlay_window.overrideredirect(True)
        x = self.region_data['x']
        y = self.region_data['y']
        w = self.region_data['width']
        h = self.region_data['height']
        self.overlay_window.geometry(f"{w}x{h}+{x}+{y}")
        self.overlay_canvas = tk.Canvas(self.overlay_window, bg='black', highlightthickness=0)
        self.overlay_canvas.pack(fill='both', expand=True)
        self.log('✅ Overlay aktif - updating setiap 1.2 detik')

    def overlay_loop(self):
        try:
            cell_size = self.region_data['cell_size']
            update_count = 0
            while self.running and self.overlay_canvas:
                try:
                    self.scan_board_state()
                    self.overlay_canvas.delete('all')
                    probabilities = self.solver.calculate_probability()
                    best_move = self.solver.get_best_move()
                    best_cell = None
                    if best_move:
                        best_cell = (best_move[0], best_move[1])
                    for (i, j), prob in probabilities.items():
                        x = j * cell_size
                        y = i * cell_size
                        # color ramp
                        if prob < 0.15:
                            color = '#00FF00'
                            text_color = 'white'
                        elif prob < 0.30:
                            color = '#7FFF00'
                            text_color = 'black'
                        elif prob < 0.50:
                            color = '#FFFF00'
                            text_color = 'black'
                        elif prob < 0.70:
                            color = '#FFA500'
                            text_color = 'white'
                        else:
                            color = '#FF0000'
                            text_color = 'white'
                        if best_cell and (i, j) == best_cell:
                            self.overlay_canvas.create_rectangle(x, y, x + cell_size, y + cell_size, outline='#00FFFF', width=4, fill='')
                            self.overlay_canvas.create_text(x + cell_size//2, y + 5, text="★", fill='#00FFFF', font=('Arial', 10, 'bold'))
                        self.overlay_canvas.create_rectangle(x + 2, y + 2, x + cell_size - 2, y + cell_size - 2, outline=color, width=2, fill='')
                        self.overlay_canvas.create_text(x + cell_size//2, y + cell_size//2, text=f"{prob:.0%}", fill=text_color, font=('Arial', 8, 'bold'))
                    if best_move:
                        action = best_move[2]
                        action_text = 'CLICK' if action == 'click' else 'FLAG'
                        info = f"Best: ({best_move[0]},{best_move[1]}) {action_text}"
                    else:
                        info = 'No moves available'
                    self.overlay_canvas.create_text(self.region_data['width'] // 2, 10, text=info, fill='cyan', font=('Arial', 10, 'bold'))
                    update_count += 1
                    if update_count % 3 == 0:
                        self.log(f'🔄 Overlay update #{update_count}')
                except tk.TclError:
                    break
                time.sleep(1.2)
        except Exception as e:
            self.log(f'❌ Overlay error: {e}')
        finally:
            if self.overlay_window:
                try:
                    self.overlay_window.destroy()
                except:
                    pass
                self.overlay_window = None

    def stop(self):
        self.log('⏹️ Menghentikan...')
        self.running = False
        if self.overlay_window:
            try:
                self.overlay_window.destroy()
            except:
                pass
            self.overlay_window = None
        self.auto_btn.config(state='normal')
        self.overlay_btn.config(state='normal')
        self.scan_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.log('✅ Stopped')

    def run(self):
        self.log('🎮 Minesweeper AI Helper - Continued')
        self.log('📚 Perubahan: minesweeper.online palette support, adjustable thresholds, grid preview')
        self.log('-' * 40)
        self.log("💡 Gunakan 'Select Region' untuk memulai")
        self.log("")
        self.root.mainloop()


def main():
    print('=' * 50)
    print('MINESWEEPER AI SOLVER - Continued')
    print('=' * 50)
    try:
        import pyautogui
        import cv2
        print('✓ Dependencies OK')
    except ImportError as e:
        print(f'✗ Missing dependency: {e}')
        print('\nInstall dengan:')
        print('pip install pyautogui opencv-python pillow numpy')
        return
    print('🚀 Launching GUI...')
    app = MinesweeperGUI()
    app.run()


if __name__ == '__main__':
    main()
