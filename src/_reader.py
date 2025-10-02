import tkinter as tk
from tkinter import ttk, colorchooser
import pyautogui
from PIL import Image, ImageTk, ImageGrab
import numpy as np
from collections import Counter
import threading
import time
import json
import os

class MinesweeperReader:
    def __init__(self, root):
        self.root = root
        self.root.title("Minesweeper Reader - Screen Monitor")
        self.root.geometry("1400x800")
        
        # Parameters
        self.capture_x = tk.IntVar(value=100)
        self.capture_y = tk.IntVar(value=100)
        self.capture_w = tk.IntVar(value=400)
        self.capture_h = tk.IntVar(value=400)
        self.grid_rows = tk.IntVar(value=10)
        self.grid_cols = tk.IntVar(value=10)
        self.color_tolerance = tk.IntVar(value=30)
        self.auto_refresh = tk.BooleanVar(value=False)
        self.refresh_rate = tk.IntVar(value=500)
        
        # Data storage
        self.current_image = None
        self.grid_data = None
        self.eyedropper_mode = False
        self.eyedropper_state = None
        
        # Color profile storage
        self.color_profiles = self.load_color_profiles()
        
        self.setup_ui()
        
    def load_color_profiles(self):
        """Load saved color profiles"""
        if os.path.exists('minesweeper_colors.json'):
            try:
                with open('minesweeper_colors.json', 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Default color profiles for minesweeper.online
        return {
            'unopened': [(127, 141, 155), (115, 131, 146), (100, 120, 140)],
            'opened': [(190, 190, 190), (200, 200, 200), (180, 180, 180)],
            'flag': [(255, 0, 0), (240, 10, 10), (220, 0, 0)],
            '1': [(0, 0, 255), (50, 100, 255), (0, 50, 200)],
            '2': [(0, 128, 0), (50, 150, 50), (0, 100, 0)],
            '3': [(255, 0, 0), (255, 50, 50), (200, 0, 0)],
            '4': [(0, 0, 128), (50, 50, 150), (0, 0, 100)],
            '5': [(128, 0, 0), (150, 50, 50), (100, 0, 0)],
            '6': [(0, 128, 128), (50, 150, 150), (0, 100, 100)],
            '7': [(0, 0, 0), (50, 50, 50), (30, 30, 30)],
            '8': [(128, 128, 128), (100, 100, 100), (140, 140, 140)]
        }
    
    def save_color_profiles(self):
        """Save color profiles to file"""
        with open('minesweeper_colors.json', 'w') as f:
            json.dump(self.color_profiles, f, indent=2)
        
    def setup_ui(self):
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Control Panel
        control_frame = ttk.LabelFrame(main_container, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Capture region controls
        ttk.Label(control_frame, text="Capture Region:").pack(anchor=tk.W)
        
        for label, var in [("X:", self.capture_x), ("Y:", self.capture_y), 
                           ("Width:", self.capture_w), ("Height:", self.capture_h)]:
            frame = ttk.Frame(control_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=label, width=8).pack(side=tk.LEFT)
            ttk.Entry(frame, textvariable=var, width=8).pack(side=tk.LEFT)
            ttk.Button(frame, text="-10", width=4, 
                      command=lambda v=var: v.set(max(0, v.get()-10))).pack(side=tk.LEFT, padx=1)
            ttk.Button(frame, text="+10", width=4, 
                      command=lambda v=var: v.set(v.get()+10)).pack(side=tk.LEFT)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Grid controls
        ttk.Label(control_frame, text="Grid Settings:").pack(anchor=tk.W)
        
        for label, var in [("Rows:", self.grid_rows), ("Columns:", self.grid_cols)]:
            frame = ttk.Frame(control_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=label, width=8).pack(side=tk.LEFT)
            ttk.Spinbox(frame, from_=1, to=50, textvariable=var, width=10).pack(side=tk.LEFT)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Detection controls
        ttk.Label(control_frame, text="Detection:").pack(anchor=tk.W)
        
        frame = ttk.Frame(control_frame)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text="Color Tol:", width=8).pack(side=tk.LEFT)
        ttk.Scale(frame, from_=0, to=100, variable=self.color_tolerance, 
                 orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(frame, textvariable=self.color_tolerance, width=4).pack(side=tk.LEFT)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Action buttons
        ttk.Button(control_frame, text="📷 Capture Screen", 
                  command=self.capture_screen).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="🔍 Analyze Grid", 
                  command=self.analyze_grid).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="📋 Get Region Info", 
                  command=self.get_region_info).pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Auto-refresh
        ttk.Checkbutton(control_frame, text="Auto Refresh", 
                       variable=self.auto_refresh, 
                       command=self.toggle_auto_refresh).pack(anchor=tk.W)
        
        frame = ttk.Frame(control_frame)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text="Rate (ms):", width=8).pack(side=tk.LEFT)
        ttk.Entry(frame, textvariable=self.refresh_rate, width=10).pack(side=tk.LEFT)
        
        # Right side container
        right_container = ttk.Frame(main_container)
        right_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Display Panel
        display_frame = ttk.Frame(right_container)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Image canvas
        self.canvas = tk.Canvas(display_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<Button-1>', self.canvas_click)
        
        # Bottom: Info and Color Picker
        bottom_frame = ttk.Frame(right_container)
        bottom_frame.pack(fill=tk.BOTH, padx=5, pady=5)
        
        # Info Panel
        info_frame = ttk.LabelFrame(bottom_frame, text="Detection Info", padding=5)
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.info_text = tk.Text(info_frame, width=50, height=15, wrap=tk.WORD, font=('Consolas', 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(info_frame, command=self.info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(yscrollcommand=scrollbar.set)
        
        # Color Picker Panel
        picker_frame = ttk.LabelFrame(bottom_frame, text="Color Picker", padding=5)
        picker_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        ttk.Label(picker_frame, text="Set colors for states:").pack(anchor=tk.W)
        
        states = ['unopened', 'opened', 'flag', '1', '2', '3', '4', '5', '6', '7', '8']
        
        self.color_buttons = {}
        for state in states:
            frame = ttk.Frame(picker_frame)
            frame.pack(fill=tk.X, pady=2)
            
            btn = tk.Button(frame, text=state, width=10, 
                          command=lambda s=state: self.pick_color(s))
            btn.pack(side=tk.LEFT, padx=2)
            self.color_buttons[state] = btn
            self.update_color_button(state)
            
            ttk.Button(frame, text="👁 Pick", width=6,
                      command=lambda s=state: self.start_eyedropper(s)).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(picker_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        ttk.Button(picker_frame, text="💾 Save Colors", 
                  command=self.save_color_profiles).pack(fill=tk.X, pady=2)
        ttk.Button(picker_frame, text="🔄 Reset Colors", 
                  command=self.reset_colors).pack(fill=tk.X, pady=2)
        
    def update_color_button(self, state):
        """Update color button appearance"""
        if state in self.color_profiles and self.color_profiles[state]:
            color = self.color_profiles[state][0]
            hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
            self.color_buttons[state].config(bg=hex_color)
            # Set text color to white or black based on brightness
            brightness = sum(color) / 3
            text_color = 'white' if brightness < 128 else 'black'
            self.color_buttons[state].config(fg=text_color)
    
    def pick_color(self, state):
        """Pick color using color chooser dialog"""
        color = colorchooser.askcolor(title=f"Choose color for {state}")
        if color[0]:
            rgb = tuple(int(c) for c in color[0])
            if state not in self.color_profiles:
                self.color_profiles[state] = []
            self.color_profiles[state].insert(0, rgb)
            self.update_color_button(state)
            self.log_info(f"✓ Color set for '{state}': RGB{rgb}")
    
    def start_eyedropper(self, state):
        """Start eyedropper mode"""
        self.eyedropper_mode = True
        self.eyedropper_state = state
        self.log_info(f"👁 EYEDROPPER MODE: Click on a GRID CELL to pick color for '{state}'")
        self.canvas.config(cursor="crosshair")
    
    def canvas_click(self, event):
        """Handle canvas click for eyedropper - select grid cell"""
        if not self.eyedropper_mode or self.current_image is None:
            return
        
        # Convert canvas coordinates to image coordinates
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        img_w = self.current_image.width
        img_h = self.current_image.height
        
        # Calculate image position on canvas
        img_ratio = img_w / img_h
        canvas_ratio = canvas_w / canvas_h
        
        if img_ratio > canvas_ratio:
            display_w = canvas_w
            display_h = int(canvas_w / img_ratio)
        else:
            display_h = canvas_h
            display_w = int(canvas_h * img_ratio)
        
        offset_x = (canvas_w - display_w) // 2
        offset_y = (canvas_h - display_h) // 2
        
        # Check if click is within image bounds
        if (event.x < offset_x or event.x > offset_x + display_w or
            event.y < offset_y or event.y > offset_y + display_h):
            return
        
        # Convert to image coordinates
        img_x = int((event.x - offset_x) * img_w / display_w)
        img_y = int((event.y - offset_y) * img_h / display_h)
        
        # Determine which grid cell was clicked
        rows, cols = self.grid_rows.get(), self.grid_cols.get()
        cell_w = img_w / cols
        cell_h = img_h / rows
        
        cell_col = int(img_x / cell_w)
        cell_row = int(img_y / cell_h)
        
        if 0 <= cell_row < rows and 0 <= cell_col < cols:
            # Extract the cell
            img_array = np.array(self.current_image)
            y1 = int(cell_row * cell_h)
            y2 = int((cell_row + 1) * cell_h)
            x1 = int(cell_col * cell_w)
            x2 = int((cell_col + 1) * cell_w)
            
            cell = img_array[y1:y2, x1:x2]
            
            # Get multi-point sampled colors from the cell
            sampled_colors = self.get_cell_sample_colors(cell)
            
            # Add all sampled colors to the profile
            state = self.eyedropper_state
            if state not in self.color_profiles:
                self.color_profiles[state] = []
            
            # Add unique colors from this cell
            for color in sampled_colors:
                if color not in self.color_profiles[state]:
                    self.color_profiles[state].append(color)
            
            self.update_color_button(state)
            
            self.log_info(f"✓ Cell [{cell_row},{cell_col}] colors picked for '{state}':\n   " + 
                         "\n   ".join([f"RGB{c}" for c in sampled_colors]))
        
        # Exit eyedropper mode
        self.eyedropper_mode = False
        self.eyedropper_state = None
        self.canvas.config(cursor="")
    
    def reset_colors(self):
        """Reset to default colors"""
        self.color_profiles = self.load_color_profiles()
        for state in self.color_buttons:
            self.update_color_button(state)
        self.log_info("✓ Colors reset to defaults")
    
    def capture_screen(self):
        """Capture the specified screen region"""
        x, y = self.capture_x.get(), self.capture_y.get()
        w, h = self.capture_w.get(), self.capture_h.get()
        
        try:
            screenshot = ImageGrab.grab(bbox=(x, y, x+w, y+h))
            self.current_image = screenshot
            self.display_image(screenshot)
            self.log_info(f"✓ Captured region: ({x}, {y}, {w}, {h})")
        except Exception as e:
            self.log_info(f"✗ Error capturing: {str(e)}")
    
    def display_image(self, image):
        """Display image on canvas with grid overlay"""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 100:
            canvas_w, canvas_h = 800, 600
        
        img_ratio = image.width / image.height
        canvas_ratio = canvas_w / canvas_h
        
        if img_ratio > canvas_ratio:
            new_w = canvas_w
            new_h = int(canvas_w / img_ratio)
        else:
            new_h = canvas_h
            new_w = int(canvas_h * img_ratio)
        
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w//2, canvas_h//2, image=self.photo)
        
        # Draw grid
        rows, cols = self.grid_rows.get(), self.grid_cols.get()
        cell_w = new_w / cols
        cell_h = new_h / rows
        
        for i in range(1, rows):
            y = i * cell_h + (canvas_h - new_h) // 2
            self.canvas.create_line(
                (canvas_w - new_w) // 2, y, 
                (canvas_w + new_w) // 2, y, 
                fill='red', width=1
            )
        
        for i in range(1, cols):
            x = i * cell_w + (canvas_w - new_w) // 2
            self.canvas.create_line(
                x, (canvas_h - new_h) // 2,
                x, (canvas_h + new_h) // 2,
                fill='red', width=1
            )
    
    def analyze_grid(self):
        """Analyze the grid and detect cell states"""
        if self.current_image is None:
            self.log_info("✗ No image captured. Capture screen first!")
            return
        
        rows, cols = self.grid_rows.get(), self.grid_cols.get()
        img_array = np.array(self.current_image)
        
        h, w = img_array.shape[:2]
        cell_h = h / rows
        cell_w = w / cols
        
        self.grid_data = []
        
        for r in range(rows):
            row_data = []
            for c in range(cols):
                y1 = int(r * cell_h)
                y2 = int((r + 1) * cell_h)
                x1 = int(c * cell_w)
                x2 = int((c + 1) * cell_w)
                
                cell = img_array[y1:y2, x1:x2]
                
                # Multi-point sampling for better detection
                state, confidence, detected_color = self.detect_cell_state_multipoint(cell)
                row_data.append({
                    'state': state,
                    'confidence': confidence,
                    'color': detected_color,
                    'position': (r, c)
                })
            
            self.grid_data.append(row_data)
        
        self.display_grid_results()
        self.log_info(f"✓ Grid analyzed: {rows}x{cols}")
    
    def get_cell_sample_colors(self, cell):
        """Get sampled colors from a cell using multi-point sampling"""
        h, w = cell.shape[:2]
        
        if h < 5 or w < 5:
            return []
        
        # Sample multiple points in the cell
        sample_points = [
            (h//2, w//2),           # Center
            (h//3, w//3),           # Top-left inner
            (h//3, 2*w//3),         # Top-right inner
            (2*h//3, w//3),         # Bottom-left inner
            (2*h//3, 2*w//3),       # Bottom-right inner
        ]
        
        sampled_colors = []
        for y, x in sample_points:
            if 0 <= y < h and 0 <= x < w:
                color = tuple(cell[y, x])
                if color not in sampled_colors:  # Only unique colors
                    sampled_colors.append(color)
        
        return sampled_colors
    
    def detect_cell_state_multipoint(self, cell):
        """Detect cell state using multi-point sampling"""
        sampled_colors = self.get_cell_sample_colors(cell)
        
        if not sampled_colors:
            return 'unknown', 0, (0, 0, 0)
        
        # Find the most distinctive/non-background color
        tolerance = self.color_tolerance.get()
        best_match = 'unknown'
        best_confidence = 0
        detected_color = (0, 0, 0)
        
        # Check each sampled color against known states
        for sample_color in sampled_colors:
            for state, ref_colors in self.color_profiles.items():
                for ref_color in ref_colors:
                    diff = sum(abs(a - b) for a, b in zip(sample_color, ref_color))
                    confidence = max(0, 100 - (diff / 3))
                    
                    if confidence > best_confidence and diff < tolerance * 3:
                        best_confidence = confidence
                        best_match = state
                        detected_color = sample_color
        
        return best_match, best_confidence, detected_color
    
    def display_grid_results(self):
        """Display grid analysis results"""
        if not self.grid_data:
            return
        
        info = "═══ GRID ANALYSIS ═══\n\n"
        
        # Count states
        state_counts = {}
        for row in self.grid_data:
            for cell in row:
                state = cell['state']
                state_counts[state] = state_counts.get(state, 0) + 1
        
        info += "State Distribution:\n"
        for state, count in sorted(state_counts.items()):
            info += f"  {state}: {count}\n"
        
        info += "\n" + "─" * 40 + "\n\n"
        
        # Grid visualization
        info += "Grid Map:\n"
        for r, row in enumerate(self.grid_data):
            info += f"Row {r:2d}: "
            for cell in row:
                state = cell['state']
                if state == 'unopened':
                    char = '□'
                elif state == 'opened':
                    char = '·'
                elif state == 'flag':
                    char = '⚑'
                elif state.isdigit():
                    char = state
                else:
                    char = '?'
                info += char + ' '
            info += "\n"
        
        info += "\n" + "─" * 40 + "\n\n"
        
        # Detailed cell info (first 10 cells)
        info += "Cell Details (first 10):\n"
        count = 0
        for r in range(len(self.grid_data)):
            for c in range(len(self.grid_data[r])):
                if count >= 10:
                    break
                cell = self.grid_data[r][c]
                info += f"[{r},{c}] {cell['state']}: RGB{cell['color']} (conf: {cell['confidence']:.0f}%)\n"
                count += 1
            if count >= 10:
                break
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)
    
    def get_region_info(self):
        """Get info about the current capture region"""
        if self.current_image is None:
            self.log_info("✗ No image captured.")
            return
        
        img_array = np.array(self.current_image)
        colors = img_array.reshape(-1, 3)
        
        info = "═══ REGION INFO ═══\n\n"
        info += f"Image Size: {self.current_image.width} x {self.current_image.height}\n\n"
        
        # Most common colors
        color_counts = Counter(map(tuple, colors))
        info += "Top 15 Colors (RGB):\n"
        for color, count in color_counts.most_common(15):
            pct = (count / len(colors)) * 100
            info += f"  {color} - {pct:.1f}%\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info)
    
    def log_info(self, message):
        """Log information message"""
        current = self.info_text.get(1.0, tk.END)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, f"{message}\n\n{current}")
    
    def toggle_auto_refresh(self):
        """Toggle auto-refresh mode"""
        if self.auto_refresh.get():
            self.auto_refresh_thread = threading.Thread(target=self.auto_refresh_loop, daemon=True)
            self.auto_refresh_thread.start()
    
    def auto_refresh_loop(self):
        """Auto-refresh loop"""
        while self.auto_refresh.get():
            self.capture_screen()
            if self.grid_rows.get() > 0 and self.grid_cols.get() > 0:
                self.analyze_grid()
            time.sleep(self.refresh_rate.get() / 1000)
    
    def get_grid_matrix(self):
        """Export grid data as matrix (for solver.py)"""
        if not self.grid_data:
            return None
        
        matrix = []
        for row in self.grid_data:
            matrix.append([cell['state'] for cell in row])
        return matrix

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = MinesweeperReader(root)
    root.mainloop()