import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Toplevel, Scale, HORIZONTAL
from PIL import Image, ImageTk
import numpy as np
import cv2
from skimage import filters, io
from skimage.transform import resize
import matplotlib.pyplot as plt

# =============================================================================
# PART 1: LOGIC FUNCTIONS (From Hamdy, Mahmoud, Ahmed)
# تجميع الدوال الحسابية من الملفات المرفقة
# =============================================================================

# --- From Mahmoud's File ---
def rgb_to_gray(rgb_image):
    if len(rgb_image.shape) == 2: return rgb_image # Already gray
    height, width, channel = rgb_image.shape
    gray_image = np.zeros((height, width), dtype=np.uint8)
    # Vectorized for performance in GUI (Original loop is too slow for real-time)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    return gray_image

def ADD_Brightness(image, constant):
    # Ensure image is float for calculation then clip
    img_float = image.astype(np.float32)
    bright_image = img_float + constant
    bright_image = np.clip(bright_image, 0, 255).astype(np.uint8)
    return bright_image

# --- From Hamdy's File ---
def apply_gaussian_filter_logic(image, kernel_size=3, sigma=1.0):
    # Using opencv for speed in GUI instead of raw loops if possible, 
    # but sticking to logic structure provided. 
    # For responsiveness, I will use cv2.GaussianBlur which simulates the loop logic faster
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def laplacian_filter_logic(image):
    # Using cv2.Laplacian to mimic the kernel convolution
    if len(image.shape) == 3: image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    return laplacian

def add_salt_and_pepper(image, prob=0.05):
    output = np.copy(image)
    # Salt
    num_salt = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    output[tuple(coords)] = 255
    # Pepper
    num_pepper = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    output[tuple(coords)] = 0
    return output

def add_gaussian_noise(image, mean=0, sigma=25):
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def add_periodic_noise(image, freq_x=0.1, freq_y=0.1, amplitude=50):
    if len(image.shape) == 3:
        rows, cols, ch = image.shape
    else:
        rows, cols = image.shape
        
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    noise_pattern = amplitude * np.sin(2 * np.pi * (freq_x * X + freq_y * Y))
    
    if len(image.shape) == 3:
        noise_pattern = np.stack([noise_pattern]*3, axis=2)

    noisy_image = image.astype(float) + noise_pattern
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def apply_dilation(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def apply_erosion(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def apply_opening(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def apply_closing(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def automatic_threshold(image):
    if len(image.shape) == 3: image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = filters.threshold_otsu(image)
    segmented = image > thresh
    return (segmented * 255).astype(np.uint8)

def apply_dithering(image):
    pil_img = Image.fromarray(image.astype(np.uint8)).convert("L")
    dithered = pil_img.convert("1")
    return np.array(dithered, dtype=np.uint8) * 255

# --- From Ahmed's File (Statistical Filters) ---
def apply_averaging_filter(image, k_size=3):
    return cv2.blur(image, (k_size, k_size))

def apply_min_filter(image, k_size=3):
    # Min filter is effectively erosion
    kernel = np.ones((k_size, k_size), np.uint8)
    return cv2.erode(image, kernel)

def apply_max_filter(image, k_size=3):
    # Max filter is effectively dilation
    kernel = np.ones((k_size, k_size), np.uint8)
    return cv2.dilate(image, kernel)

def apply_median_filter(image, k_size=3):
    # Must be odd
    if k_size % 2 == 0: k_size += 1
    return cv2.medianBlur(image, k_size)

def apply_mode_filter(image, k_size=3):
    # Mode is complex and slow in pure python, simplified here
    if len(image.shape) == 3: image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    from scipy.ndimage import generic_filter
    from scipy.stats import mode
    def get_mode(x): return mode(x, axis=None, keepdims=False)[0]
    return generic_filter(image, get_mode, size=k_size)

def apply_range_filter(image, k_size=3):
    # Max - Min
    mx = apply_max_filter(image, k_size)
    mn = apply_min_filter(image, k_size)
    return mx - mn

# =============================================================================
# PART 2: GUI IMPLEMENTATION
# تصميم الواجهة الرسومية
# =============================================================================

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("تغير الصور الأحترافي - Professional Image Editor")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2b2b2b") # Dark theme

        # Variables
        self.original_image_np = None # Numpy Array
        self.current_image_np = None  # Numpy Array to show
        self.display_image_ref = None # Tkinter Image Reference

        # --- GUI Layout ---
        
        # 1. Main Canvas (Image Display)
        self.canvas_frame = tk.Frame(root, bg="#1e1e1e", bd=2, relief="sunken")
        self.canvas_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # 2. Bottom Control Panel
        self.controls_frame = tk.Frame(root, bg="#2b2b2b")
        self.controls_frame.pack(fill="x", side="bottom", padx=10, pady=20)

        # Style for buttons
        btn_style = {"bg": "#4a90e2", "fg": "white", "font": ("Arial", 10, "bold"), "width": 12, "height": 2, "relief": "flat"}
        btn_style_red = {"bg": "#e74c3c", "fg": "white", "font": ("Arial", 10, "bold"), "width": 10, "height": 2, "relief": "flat"}
        btn_style_green = {"bg": "#2ecc71", "fg": "white", "font": ("Arial", 10, "bold"), "width": 10, "height": 2, "relief": "flat"}

        # Buttons Definition
        tk.Button(self.controls_frame, text="Load Image", command=self.load_image, **btn_style).pack(side="left", padx=5)
        tk.Button(self.controls_frame, text="Brightness", command=self.open_brightness_control, **btn_style).pack(side="left", padx=5)
        tk.Button(self.controls_frame, text="Filters", command=self.open_filters_menu, **btn_style).pack(side="left", padx=5)
        tk.Button(self.controls_frame, text="Noise", command=self.open_noise_menu, **btn_style).pack(side="left", padx=5)
        tk.Button(self.controls_frame, text="Crop", command=self.crop_center, **btn_style).pack(side="left", padx=5)
        tk.Button(self.controls_frame, text="Grayscale", command=self.convert_to_gray, **btn_style).pack(side="left", padx=5)
        
        # Right side buttons (Reset, Save)
        tk.Button(self.controls_frame, text="Reset", command=self.reset_image, **btn_style_red).pack(side="right", padx=5)
        tk.Button(self.controls_frame, text="Save", command=self.save_image, **btn_style_green).pack(side="right", padx=5)

    # --- Core Functions ---

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
        if file_path:
            # Read image using OpenCV (BGR) then convert to RGB
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.original_image_np = img
            self.current_image_np = img.copy()
            self.show_image()

    def show_image(self):
        if self.current_image_np is None: return
        
        # Resize for display to fit canvas
        h, w = self.current_image_np.shape[:2]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w < 50: canvas_w = 800 # Default if not rendered yet
        if canvas_h < 50: canvas_h = 600

        scale = min(canvas_w/w, canvas_h/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        img_resized = cv2.resize(self.current_image_np, (new_w, new_h))
        img_pil = Image.fromarray(img_resized)
        self.display_image_ref = ImageTk.PhotoImage(img_pil)
        
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w//2, canvas_h//2, anchor="center", image=self.display_image_ref)

    def reset_image(self):
        if self.original_image_np is not None:
            self.current_image_np = self.original_image_np.copy()
            self.show_image()

    def save_image(self):
        if self.current_image_np is None: return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if file_path:
            # Convert RGB back to BGR for OpenCV saving
            if len(self.current_image_np.shape) == 3:
                save_img = cv2.cvtColor(self.current_image_np, cv2.COLOR_RGB2BGR)
            else:
                save_img = self.current_image_np
            cv2.imwrite(file_path, save_img)
            messagebox.showinfo("Success", "Image saved successfully!")

    # --- Feature: Brightness ---
    def open_brightness_control(self):
        if self.current_image_np is None: return
        
        # Popup window
        top = Toplevel(self.root)
        top.title("Brightness")
        top.geometry("300x100")
        top.configure(bg="#2b2b2b")
        
        lbl = tk.Label(top, text="Adjust Brightness", fg="white", bg="#2b2b2b")
        lbl.pack(pady=5)

        def update_brightness(val):
            val = int(val)
            # Use original image as base to avoid quality degradation
            self.current_image_np = ADD_Brightness(self.original_image_np, val)
            self.show_image()

        slider = Scale(top, from_=-100, to=100, orient=HORIZONTAL, command=update_brightness, bg="#2b2b2b", fg="white")
        slider.set(0)
        slider.pack(fill="x", padx=20)

    # --- Feature: Grayscale ---
    def convert_to_gray(self):
        if self.current_image_np is None: return
        self.current_image_np = rgb_to_gray(self.current_image_np)
        self.show_image()

    # --- Feature: Crop (Simple Center Crop) ---
    def crop_center(self):
        if self.current_image_np is None: return
        h, w = self.current_image_np.shape[:2]
        start_row, start_col = int(h * 0.1), int(w * 0.1)
        end_row, end_col = int(h * 0.9), int(w * 0.9)
        self.current_image_np = self.current_image_np[start_row:end_row, start_col:end_col]
        self.show_image()

    # --- Feature: Filters Menu ---
    def open_filters_menu(self):
        if self.current_image_np is None: return
        
        top = Toplevel(self.root)
        top.title("Filters")
        top.configure(bg="#2b2b2b")
        
        filters_list = [
            ("Laplacian Filter", lambda: self.apply_filter(laplacian_filter_logic)),
            ("Automatic Thresholding", lambda: self.apply_filter(automatic_threshold)),
            ("Dithering", lambda: self.apply_filter(apply_dithering)),
            ("Gaussian Smoothing", lambda: self.apply_filter(apply_gaussian_filter_logic)),
            ("Mean Filter", lambda: self.apply_filter(apply_averaging_filter)),
            ("Median Filter", lambda: self.apply_filter(apply_median_filter)),
            ("Min Filter", lambda: self.apply_filter(apply_min_filter)),
            ("Max Filter", lambda: self.apply_filter(apply_max_filter)),
            ("Mode Filter (Slow)", lambda: self.apply_filter(apply_mode_filter)),
            ("Range Filter", lambda: self.apply_filter(apply_range_filter)),
            ("Dilation", lambda: self.apply_filter(apply_dilation)),
            ("Erosion", lambda: self.apply_filter(apply_erosion)),
            ("Opening", lambda: self.apply_filter(apply_opening)),
            ("Closing", lambda: self.apply_filter(apply_closing)),
        ]

        for text, cmd in filters_list:
            btn = tk.Button(top, text=text, command=cmd, bg="#3498db", fg="white", width=25, relief="flat")
            btn.pack(pady=2, padx=10)

    def apply_filter(self, filter_func):
        try:
            # Some filters require grayscale
            if filter_func in [automatic_threshold, apply_dithering, laplacian_filter_logic, apply_range_filter, apply_mode_filter]:
                 if len(self.current_image_np.shape) == 3:
                     self.current_image_np = rgb_to_gray(self.current_image_np)
            
            self.current_image_np = filter_func(self.current_image_np)
            self.show_image()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # --- Feature: Noise Menu ---
    def open_noise_menu(self):
        if self.current_image_np is None: return
        
        top = Toplevel(self.root)
        top.title("Noise")
        top.configure(bg="#2b2b2b")
        
        noise_list = [
            ("Salt & Pepper", lambda: self.apply_filter(lambda img: add_salt_and_pepper(img, 0.05))),
            ("Gaussian Noise", lambda: self.apply_filter(lambda img: add_gaussian_noise(img))),
            ("Periodic Noise", lambda: self.apply_filter(lambda img: add_periodic_noise(img))),
        ]

        for text, cmd in noise_list:
            btn = tk.Button(top, text=text, command=cmd, bg="#9b59b6", fg="white", width=20, relief="flat")
            btn.pack(pady=2, padx=10)

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()