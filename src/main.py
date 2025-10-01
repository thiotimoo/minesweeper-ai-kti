import pyautogui
import numpy as np
import cv2
from PIL import Image, ImageGrab
import time
import tkinter as tk
from tkinter import ttk
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import sys

@dataclass
class Cell:
    """Representasi sel dalam matriks Minesweeper"""
    x: int
    y: int
    value: int  # -2: unknown, -1: mine, 0-8: number
    screen_x: int
    screen_y: int
    probability: float = 0.0

class MinesweeperSolver:
    """
    Solver Minesweeper berbasis Matriks dan Probabilitas
    Mengimplementasikan konsep dari makalah Stima ITB
    """
    
    def __init__(self, board_width: int, board_height: int, total_mines: int):
        self.width = board_width
        self.height = board_height
        self.total_mines = total_mines
        self.board = np.full((board_height, board_width), -2, dtype=int)  # -2 = unknown
        self.flagged = set()
        self.safe_cells = set()
        
    def get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        """Mendapatkan 8 tetangga dari sel (i,j)"""
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
        """Update nilai sel pada matriks"""
        self.board[i, j] = value
    
    def solve_logic(self) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        """
        Menyelesaikan menggunakan logika deduktif (sistem persamaan linier)
        Returns: (safe_cells, mine_cells)
        """
        safe = set()
        mines = set()
        
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i, j] >= 0:  # Sel dengan angka
                    neighbors = self.get_neighbors(i, j)
                    unknown_neighbors = [(ni, nj) for ni, nj in neighbors 
                                       if self.board[ni, nj] == -2]
                    flagged_neighbors = [(ni, nj) for ni, nj in neighbors 
                                        if (ni, nj) in self.flagged]
                    
                    remaining_mines = self.board[i, j] - len(flagged_neighbors)
                    
                    # Jika jumlah unknown = sisa ranjau, semua adalah ranjau
                    if len(unknown_neighbors) == remaining_mines and remaining_mines > 0:
                        mines.update(unknown_neighbors)
                    
                    # Jika tidak ada sisa ranjau, semua unknown aman
                    elif remaining_mines == 0:
                        safe.update(unknown_neighbors)
        
        return safe, mines
    
    def calculate_probability(self) -> dict:
        """
        Menghitung probabilitas setiap sel mengandung ranjau
        Menggunakan P(mine) = R/U (global probability)
        """
        probabilities = {}
        
        # Hitung jumlah ranjau tersisa (R) dan sel tertutup (U)
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
        
        # Global probability: P(mine) = R/U
        global_prob = remaining_mines / total_unknown if total_unknown > 0 else 0
        
        # Hitung probabilitas lokal untuk setiap sel
        for i, j in unknown_cells:
            # Cek apakah ada sel bernomor di sekitarnya
            neighbors = self.get_neighbors(i, j)
            local_probs = []
            
            for ni, nj in neighbors:
                if self.board[ni, nj] >= 0:  # Sel dengan angka
                    nei_neighbors = self.get_neighbors(ni, nj)
                    unknown_nei = [n for n in nei_neighbors if self.board[n[0], n[1]] == -2]
                    flagged_nei = [n for n in nei_neighbors if n in self.flagged]
                    
                    remaining = self.board[ni, nj] - len(flagged_nei)
                    u = len(unknown_nei)
                    
                    if u > 0:
                        # P(x_pq = 1) ≈ c/u
                        local_prob = remaining / u
                        local_probs.append(local_prob)
            
            # Gunakan rata-rata dari probabilitas lokal atau global probability
            if local_probs:
                probabilities[(i, j)] = sum(local_probs) / len(local_probs)
            else:
                probabilities[(i, j)] = global_prob
        
        return probabilities
    
    def get_best_move(self) -> Optional[Tuple[int, int, str]]:
        """
        Mendapatkan langkah terbaik berikutnya
        Returns: (i, j, action) dimana action = 'click' atau 'flag'
        """
        # 1. Coba deduksi logis
        safe, mines = self.solve_logic()
        
        if safe:
            cell = safe.pop()
            return (cell[0], cell[1], 'click')
        
        if mines:
            cell = mines.pop()
            self.flagged.add(cell)
            return (cell[0], cell[1], 'flag')
        
        # 2. Gunakan probabilitas
        probabilities = self.calculate_probability()
        
        if not probabilities:
            return None
        
        # Pilih sel dengan probabilitas terendah (paling aman)
        best_cell = min(probabilities.items(), key=lambda x: x[1])
        return (best_cell[0][0], best_cell[0][1], 'click')


class MinesweeperDetector:
    """Deteksi dan parsing papan Minesweeper dari layar"""
    
    def __init__(self):
        self.cell_size = 16  # Default Windows Minesweeper
        self.board_region = None
        
    def find_board(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Mencari lokasi papan Minesweeper di layar
        Returns: (x, y, width, height) atau None
        """
        print("Mencari papan Minesweeper di layar...")
        print("Pastikan Minesweeper terbuka dan terlihat!")
        
        # Ambil screenshot
        screenshot = ImageGrab.grab()
        screenshot_np = np.array(screenshot)
        screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)
        
        # Deteksi grid pattern (simplified)
        # Di implementasi nyata, gunakan template matching atau edge detection
        # Untuk demo, gunakan input manual
        return None
    
    def detect_cells(self, region: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Deteksi nilai setiap sel dalam region
        Returns: numpy array berisi nilai sel
        """
        # Implementasi computer vision untuk mendeteksi angka/state sel
        # Untuk demo, ini simplified version
        pass


class MinesweeperGUI:
    """GUI untuk overlay guide dan control panel"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Minesweeper AI Helper")
        self.root.geometry("400x600")
        self.root.attributes('-topmost', True)
        
        self.solver = None
        self.overlay_window = None
        self.running = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup UI controls"""
        # Frame konfigurasi
        config_frame = ttk.LabelFrame(self.root, text="Konfigurasi Papan", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(config_frame, text="Lebar:").grid(row=0, column=0, sticky='w')
        self.width_var = tk.IntVar(value=9)
        ttk.Entry(config_frame, textvariable=self.width_var, width=10).grid(row=0, column=1)
        
        ttk.Label(config_frame, text="Tinggi:").grid(row=1, column=0, sticky='w')
        self.height_var = tk.IntVar(value=9)
        ttk.Entry(config_frame, textvariable=self.height_var, width=10).grid(row=1, column=1)
        
        ttk.Label(config_frame, text="Total Ranjau:").grid(row=2, column=0, sticky='w')
        self.mines_var = tk.IntVar(value=10)
        ttk.Entry(config_frame, textvariable=self.mines_var, width=10).grid(row=2, column=1)
        
        # Frame region
        region_frame = ttk.LabelFrame(self.root, text="Region Papan (x, y, width, height)", padding=10)
        region_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(region_frame, text="X:").grid(row=0, column=0, sticky='w')
        self.region_x = tk.IntVar(value=100)
        ttk.Entry(region_frame, textvariable=self.region_x, width=10).grid(row=0, column=1)
        
        ttk.Label(region_frame, text="Y:").grid(row=1, column=0, sticky='w')
        self.region_y = tk.IntVar(value=100)
        ttk.Entry(region_frame, textvariable=self.region_y, width=10).grid(row=1, column=1)
        
        ttk.Label(region_frame, text="Cell Size:").grid(row=2, column=0, sticky='w')
        self.cell_size = tk.IntVar(value=16)
        ttk.Entry(region_frame, textvariable=self.cell_size, width=10).grid(row=2, column=1)
        
        ttk.Button(region_frame, text="Detect Region", command=self.detect_region).grid(row=3, column=0, columnspan=2, pady=5)
        
        # Frame mode
        mode_frame = ttk.LabelFrame(self.root, text="Mode", padding=10)
        mode_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(mode_frame, text="Mode 1: Auto Solve", 
                  command=self.start_autosolve, width=30).pack(pady=5)
        ttk.Button(mode_frame, text="Mode 2: Guide Overlay", 
                  command=self.start_overlay, width=30).pack(pady=5)
        ttk.Button(mode_frame, text="Stop", 
                  command=self.stop, width=30).pack(pady=5)
        
        # Status
        status_frame = ttk.LabelFrame(self.root, text="Status & Log", padding=10)
        status_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.log_text = tk.Text(status_frame, height=15, width=45)
        self.log_text.pack(fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(status_frame, command=self.log_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # Info
        info_frame = ttk.Frame(self.root)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        info_text = """
INSTRUKSI:
1. Buka game Minesweeper
2. Set konfigurasi papan (width, height, mines)
3. Detect region atau set manual
4. Pilih mode:
   - Auto Solve: AI main otomatis
   - Guide Overlay: Tampilkan saran di layar
        """
        ttk.Label(info_frame, text=info_text, justify='left', 
                 font=('Arial', 8)).pack()
    
    def log(self, message: str):
        """Log message ke text widget"""
        self.log_text.insert('end', f"{message}\n")
        self.log_text.see('end')
        self.root.update()
    
    def detect_region(self):
        """Deteksi region papan Minesweeper"""
        self.log("Mendeteksi region papan...")
        self.log("Fitur ini memerlukan implementasi computer vision")
        self.log("Untuk demo, gunakan input manual")
    
    def start_autosolve(self):
        """Mode 1: Auto solve"""
        if self.running:
            self.log("Sudah berjalan!")
            return
        
        self.running = True
        self.log("=== MODE 1: AUTO SOLVE ===")
        self.log("Memulai auto solve...")
        
        # Inisialisasi solver
        width = self.width_var.get()
        height = self.height_var.get()
        mines = self.mines_var.get()
        
        self.solver = MinesweeperSolver(width, height, mines)
        self.log(f"Board: {width}x{height}, Mines: {mines}")
        
        # Jalankan di thread terpisah
        thread = threading.Thread(target=self.autosolve_loop, daemon=True)
        thread.start()
    
    def autosolve_loop(self):
        """Loop utama auto solve"""
        try:
            region_x = self.region_x.get()
            region_y = self.region_y.get()
            cell_size = self.cell_size.get()
            
            self.log("Tekan Ctrl+C untuk stop")
            time.sleep(2)
            
            move_count = 0
            
            while self.running:
                # 1. Scan papan (simplified - perlu computer vision)
                self.log(f"Scan #{move_count + 1}...")
                
                # Demo: simulasi update board
                # Dalam implementasi nyata, gunakan computer vision
                
                # 2. Hitung langkah terbaik
                move = self.solver.get_best_move()
                
                if move is None:
                    self.log("Tidak ada langkah aman! Menggunakan probabilitas...")
                    time.sleep(1)
                    continue
                
                i, j, action = move
                
                # 3. Eksekusi langkah
                click_x = region_x + j * cell_size + cell_size // 2
                click_y = region_y + i * cell_size + cell_size // 2
                
                if action == 'click':
                    self.log(f"Klik ({i},{j}) - Prob: {self.solver.calculate_probability().get((i,j), 0):.2%}")
                    pyautogui.click(click_x, click_y)
                elif action == 'flag':
                    self.log(f"Flag ({i},{j}) sebagai ranjau")
                    pyautogui.rightClick(click_x, click_y)
                
                move_count += 1
                time.sleep(0.5)  # Delay antar langkah
                
                # Safety limit
                if move_count >= 100:
                    self.log("Mencapai batas maksimum langkah")
                    break
                    
        except Exception as e:
            self.log(f"Error: {str(e)}")
        finally:
            self.running = False
            self.log("Auto solve selesai")
    
    def start_overlay(self):
        """Mode 2: Guide overlay"""
        if self.running:
            self.log("Sudah berjalan!")
            return
        
        self.running = True
        self.log("=== MODE 2: GUIDE OVERLAY ===")
        self.log("Menampilkan overlay guide...")
        
        # Inisialisasi solver
        width = self.width_var.get()
        height = self.height_var.get()
        mines = self.mines_var.get()
        
        self.solver = MinesweeperSolver(width, height, mines)
        
        # Buat overlay window
        self.create_overlay()
        
        # Update overlay di thread terpisah
        thread = threading.Thread(target=self.overlay_loop, daemon=True)
        thread.start()
    
    def create_overlay(self):
        """Membuat transparent overlay window"""
        self.overlay_window = tk.Toplevel(self.root)
        self.overlay_window.attributes('-alpha', 0.7)
        self.overlay_window.attributes('-topmost', True)
        self.overlay_window.overrideredirect(True)
        
        region_x = self.region_x.get()
        region_y = self.region_y.get()
        width = self.width_var.get()
        height = self.height_var.get()
        cell_size = self.cell_size.get()
        
        self.overlay_window.geometry(f"{width*cell_size}x{height*cell_size}+{region_x}+{region_y}")
        
        self.overlay_canvas = tk.Canvas(self.overlay_window, 
                                       bg='black', 
                                       highlightthickness=0)
        self.overlay_canvas.pack(fill='both', expand=True)
        
    def overlay_loop(self):
        """Loop update overlay"""
        try:
            cell_size = self.cell_size.get()
            
            while self.running:
                if self.overlay_canvas:
                    self.overlay_canvas.delete('all')
                    
                    # Hitung probabilitas
                    probabilities = self.solver.calculate_probability()
                    
                    # Tampilkan probability untuk setiap sel
                    for (i, j), prob in probabilities.items():
                        x = j * cell_size
                        y = i * cell_size
                        
                        # Warna berdasarkan probabilitas (hijau=aman, merah=bahaya)
                        if prob < 0.2:
                            color = '#00FF00'  # Hijau - Aman
                        elif prob < 0.4:
                            color = '#FFFF00'  # Kuning
                        elif prob < 0.6:
                            color = '#FFA500'  # Orange
                        else:
                            color = '#FF0000'  # Merah - Bahaya
                        
                        self.overlay_canvas.create_rectangle(
                            x, y, x + cell_size, y + cell_size,
                            outline=color, width=2
                        )
                        
                        # Tampilkan persentase
                        self.overlay_canvas.create_text(
                            x + cell_size//2, y + cell_size//2,
                            text=f"{prob:.0%}",
                            fill='white',
                            font=('Arial', 8, 'bold')
                        )
                
                time.sleep(1)  # Update setiap detik
                
        except Exception as e:
            self.log(f"Overlay error: {str(e)}")
        finally:
            if self.overlay_window:
                self.overlay_window.destroy()
                self.overlay_window = None
    
    def stop(self):
        """Stop semua proses"""
        self.running = False
        if self.overlay_window:
            self.overlay_window.destroy()
            self.overlay_window = None
        self.log("Stopped")
    
    def run(self):
        """Jalankan aplikasi"""
        self.log("Minesweeper AI Helper dimulai")
        self.log("Algoritma berbasis Matriks dan Probabilitas")
        self.log("-" * 40)
        self.root.mainloop()


def main():
    """Entry point"""
    print("=" * 50)
    print("MINESWEEPER AI SOLVER & GUIDE")
    print("Berbasis Matriks dan Teori Probabilitas")
    print("=" * 50)
    print()
    
    # Check dependencies
    try:
        import pyautogui
        import cv2
        print("✓ Dependencies OK")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nInstall dengan:")
        print("pip install pyautogui opencv-python pillow numpy")
        return
    
    # Jalankan GUI
    app = MinesweeperGUI()
    app.run()


if __name__ == "__main__":
    main()