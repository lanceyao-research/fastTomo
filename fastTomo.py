import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import mss
import time
import cv2
import os
import datetime
import threading
import json
import queue
import socket

# Try to import temscript, but don't fail if not available
try:
    import temscript
    TEMSCRIPT_AVAILABLE = True
except ImportError:
    TEMSCRIPT_AVAILABLE = False

# Try to import ultralytics, but don't fail if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

CONFIG_FILE = "configure.json"


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)

        vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_configure(self, _):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")


class ScreenGrabberApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fast Tomo v0.0.7")
        self.root.geometry("1400x950")

        scroll = ScrollableFrame(self.root)
        scroll.pack(fill="both", expand=True)

        self.ui = scroll.inner

        # State variables
        self.enable = tk.BooleanVar(value=False)
        self.tracking = tk.BooleanVar(value=True)
        self.connected = False
        self.M = None
        self.tilt_trigger = False
        self.tilt_status = False
        self.tlist = None
        self.t_counter = None
        self.logs = None
        self.pos_init = {}

        self.last_tilt_time = time.time()
        self.time = time.time()
        self.display_size = 256
        self.img_size = 512
        self.corrected = None

        self.FOV = 512
        self.multiplier = 2.5
        self.scale = self.img_size / self.FOV
        self.tilt_LB = -50
        self.tilt_UB = 50
        self.tilt_int = 5
        self.delay = 2
        self.transThres = 25

        # Capture source: 'mss' or 'capture_card'
        self.capture_source = tk.StringVar(value='mss')
        self.capture_device_index = tk.IntVar(value=0)
        self.capture_resolution = tk.StringVar(value="")
        self.available_devices = []
        self.available_resolutions = []
        self.capture_card = None
        self.capture_card_latest_frame = None
        self.capture_card_lock = threading.Lock()
        self.capture_card_running = False
        self.capture_card_thread = None

        # Tracking method: 'classical' or 'ml'
        self.tracking_method = tk.StringVar(value='classical')

        # Slider variables
        self.x_var = tk.IntVar(value=0)
        self.y_var = tk.IntVar(value=0)
        self.w_var = tk.IntVar(value=512)
        self.h_var = tk.IntVar(value=512)
        self.blur_var = tk.IntVar(value=15)
        self.thresh_var = tk.DoubleVar(value=0.5)
        self.square_var = tk.BooleanVar(value=True)
        self.invert_var = tk.BooleanVar(value=True)  # Invert contrast checkbox
        self.margin_var = tk.IntVar(value=5)
        self.area_lb_var = tk.IntVar(value=400)
        self.area_ub_var = tk.IntVar(value=90000)
        self.confidence_var = tk.DoubleVar(value=0.5)

        # YOLO model
        self.yolo_model = None
        self.yolo_model_path = tk.StringVar(value="No model selected")
        self.yolo_model_loaded = False
        
        # YOLO threading
        self.yolo_input_queue = queue.Queue(maxsize=2)
        self.yolo_output_queue = queue.Queue(maxsize=2)
        self.yolo_thread = None
        self.yolo_running = False
        self.yolo_fps = 0
        self.yolo_last_time = time.time()
        self.yolo_frame_count = 0
        self.latest_yolo_result = None

        # Load config before building UI
        self.load_config()

        # Build UI
        self.build_ui()
        
        # Apply loaded config to UI
        self.apply_config_to_ui()

        # Bind variable changes to save config
        self.bind_config_save()

        self.sct = mss.mss()
        self.last_time = time.time()
        self.frame_count = 0

        # Scan for available capture devices
        self.scan_capture_devices()

        self.update_status_circle()
        self.update_tracking_status_label()
        self.update_image()

    def load_config(self):
        """Load configuration from JSON file"""
        self.config = {}
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
                self.config = {}

    def save_config(self, *args):
        """Save current configuration to JSON file"""
        config = {
            # Capture source
            'capture_source': self.capture_source.get(),
            'capture_device_index': self.capture_device_index.get(),
            'capture_resolution': self.capture_resolution.get(),
            
            # Screen capture
            'x': self.x_var.get(),
            'y': self.y_var.get(),
            'w': self.w_var.get(),
            'h': self.h_var.get(),
            'square': self.square_var.get(),
            'invert': self.invert_var.get(),
            
            # Tracking
            'tracking_method': self.tracking_method.get(),
            'blur': self.blur_var.get(),
            'thresh': self.thresh_var.get(),
            'margin': self.margin_var.get(),
            'area_lb': self.area_lb_var.get(),
            'area_ub': self.area_ub_var.get(),
            'confidence': self.confidence_var.get(),
            'yolo_model_path': self.yolo_model_path.get(),
            
            # Configuration text entries
            'microscopy_ip': self.text_entries.get("Microscopy IP", ttk.Entry()).get() if hasattr(self, 'text_entries') else "192.168.0.1",
            'port': self.text_entries.get("Port", ttk.Entry()).get() if hasattr(self, 'text_entries') else "8080",
            'tilt_start': self.text_entries.get("Tilt angle start", ttk.Entry()).get() if hasattr(self, 'text_entries') else str(self.tilt_LB),
            'tilt_end': self.text_entries.get("Tilt angle end", ttk.Entry()).get() if hasattr(self, 'text_entries') else str(self.tilt_UB),
            'tilt_interval': self.text_entries.get("Tilt interval", ttk.Entry()).get() if hasattr(self, 'text_entries') else str(self.tilt_int),
            'fov': self.text_entries.get("FOV (nm)", ttk.Entry()).get() if hasattr(self, 'text_entries') else "512",
            'delay': self.text_entries.get("Delay time (s)", ttk.Entry()).get() if hasattr(self, 'text_entries') else str(self.delay),
            'trans_threshold': self.text_entries.get("Trans threshold (px)", ttk.Entry()).get() if hasattr(self, 'text_entries') else str(self.transThres),
            'multiplier': self.text_entries.get("Multiplier", ttk.Entry()).get() if hasattr(self, 'text_entries') else str(self.multiplier),
            
            # Control states
            'tracking_enabled': self.tracking.get(),
        }
        
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def apply_config_to_ui(self):
        """Apply loaded config values to UI elements"""
        if not self.config:
            return
        
        # Apply capture source
        if 'capture_source' in self.config:
            self.capture_source.set(self.config['capture_source'])
        if 'capture_device_index' in self.config:
            self.capture_device_index.set(self.config['capture_device_index'])
        if 'capture_resolution' in self.config:
            self.capture_resolution.set(self.config['capture_resolution'])
            
        # Apply slider values
        if 'x' in self.config:
            self.x_var.set(self.config['x'])
        if 'y' in self.config:
            self.y_var.set(self.config['y'])
        if 'w' in self.config:
            self.w_var.set(self.config['w'])
        if 'h' in self.config:
            self.h_var.set(self.config['h'])
        if 'square' in self.config:
            self.square_var.set(self.config['square'])
        if 'invert' in self.config:
            self.invert_var.set(self.config['invert'])
        if 'blur' in self.config:
            self.blur_var.set(self.config['blur'])
        if 'thresh' in self.config:
            self.thresh_var.set(self.config['thresh'])
        if 'margin' in self.config:
            self.margin_var.set(self.config['margin'])
        if 'area_lb' in self.config:
            self.area_lb_var.set(self.config['area_lb'])
        if 'area_ub' in self.config:
            self.area_ub_var.set(self.config['area_ub'])
        if 'confidence' in self.config:
            self.confidence_var.set(self.config['confidence'])
        if 'yolo_model_path' in self.config:
            self.yolo_model_path.set(self.config['yolo_model_path'])
        if 'tracking_method' in self.config:
            self.tracking_method.set(self.config['tracking_method'])
        if 'tracking_enabled' in self.config:
            self.tracking.set(self.config['tracking_enabled'])
            
        # Apply text entry values
        if hasattr(self, 'text_entries'):
            if 'microscopy_ip' in self.config:
                self.set_entry_value("Microscopy IP", self.config['microscopy_ip'])
            if 'port' in self.config:
                self.set_entry_value("Port", self.config['port'])
            if 'tilt_start' in self.config:
                self.set_entry_value("Tilt angle start", self.config['tilt_start'])
            if 'tilt_end' in self.config:
                self.set_entry_value("Tilt angle end", self.config['tilt_end'])
            if 'tilt_interval' in self.config:
                self.set_entry_value("Tilt interval", self.config['tilt_interval'])
            if 'fov' in self.config:
                self.set_entry_value("FOV (nm)", self.config['fov'])
            if 'delay' in self.config:
                self.set_entry_value("Delay time (s)", self.config['delay'])
            if 'trans_threshold' in self.config:
                self.set_entry_value("Trans threshold (px)", self.config['trans_threshold'])
            if 'multiplier' in self.config:
                self.set_entry_value("Multiplier", self.config['multiplier'])

        # Update UI state
        self.toggle_height_slider()
        self.on_tracking_method_change()
        self.on_capture_source_change()
        
        # Update slider value labels
        self.root.after(100, self.update_all_slider_labels)

    def update_all_slider_labels(self):
        """Update all slider value labels to reflect current values"""
        for name, label in self.slider_labels.items():
            if name in ["Thresh", "Confidence"]:
                var = self.thresh_var if name == "Thresh" else self.confidence_var
                label.config(text=f"{var.get():.2f}")
            elif name == "X":
                label.config(text=str(self.x_var.get()))
            elif name == "Y":
                label.config(text=str(self.y_var.get()))
            elif name == "W":
                label.config(text=str(self.w_var.get()))
            elif name == "H":
                label.config(text=str(self.h_var.get()))
            elif name == "Blur":
                label.config(text=str(self.blur_var.get()))
            elif name == "Margin":
                label.config(text=str(self.margin_var.get()))
            elif name == "Area LB":
                label.config(text=str(self.area_lb_var.get()))
            elif name == "Area UB":
                label.config(text=str(self.area_ub_var.get()))

    def set_entry_value(self, name, value):
        """Set a text entry value"""
        if name in self.text_entries:
            entry = self.text_entries[name]
            entry.delete(0, tk.END)
            entry.insert(0, str(value))

    def bind_config_save(self):
        """Bind variable changes to save config"""
        # Bind capture source variables
        self.capture_source.trace_add('write', self.save_config)
        self.capture_device_index.trace_add('write', self.save_config)
        self.capture_resolution.trace_add('write', self.save_config)
        
        # Bind slider variables
        self.x_var.trace_add('write', self.save_config)
        self.y_var.trace_add('write', self.save_config)
        self.w_var.trace_add('write', self.save_config)
        self.h_var.trace_add('write', self.save_config)
        self.square_var.trace_add('write', self.save_config)
        self.invert_var.trace_add('write', self.save_config)
        self.blur_var.trace_add('write', self.save_config)
        self.thresh_var.trace_add('write', self.save_config)
        self.margin_var.trace_add('write', self.save_config)
        self.area_lb_var.trace_add('write', self.save_config)
        self.area_ub_var.trace_add('write', self.save_config)
        self.confidence_var.trace_add('write', self.save_config)
        self.yolo_model_path.trace_add('write', self.save_config)
        self.tracking_method.trace_add('write', self.save_config)
        self.tracking.trace_add('write', self.save_config)

        # Bind text entry changes
        for name, entry in self.text_entries.items():
            entry.bind('<KeyRelease>', lambda e: self.save_config())
            entry.bind('<FocusOut>', lambda e: self.save_config())

    def get_common_resolutions(self):
        """Return a list of common resolutions to try"""
        return [
            (1920, 1080),  # 1080p
            (1280, 720),   # 720p
            (1024, 768),   # XGA
            (800, 600),    # SVGA
            (640, 480),    # VGA
            (320, 240),    # QVGA
            (2560, 1440),  # 1440p
            (3840, 2160),  # 4K
        ]

    def probe_device_resolutions(self, device_index):
        """Probe a device for supported resolutions"""
        supported = []
        cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)  # Use DirectShow on Windows
        
        if not cap.isOpened():
            # Try without DirectShow
            cap = cv2.VideoCapture(device_index)
            if not cap.isOpened():
                return supported
        
        # Get current resolution as default
        current_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        current_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if current_w > 0 and current_h > 0:
            supported.append((current_w, current_h))
        
        # Try common resolutions
        for w, h in self.get_common_resolutions():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if (actual_w, actual_h) not in supported and actual_w > 0 and actual_h > 0:
                supported.append((actual_w, actual_h))
        
        cap.release()
        
        # Sort by resolution (highest first)
        supported.sort(key=lambda x: x[0] * x[1], reverse=True)
        return supported

    def scan_capture_devices(self):
        """Scan for available video capture devices"""
        self.available_devices = []
        
        # Test up to 10 device indices
        for i in range(10):
            # Try DirectShow first (better for Windows capture cards)
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                device_info = {
                    'index': i,
                    'name': f"Device {i}",
                    'width': width,
                    'height': height,
                    'resolution': f"{width}x{height}"
                }
                self.available_devices.append(device_info)
                cap.release()
        
        # Update device dropdown if it exists
        if hasattr(self, 'device_dropdown'):
            self.update_device_dropdown()
        
        self.log(f"Found {len(self.available_devices)} capture device(s)")
        return self.available_devices

    def update_device_dropdown(self):
        """Update the device dropdown with available devices"""
        if not self.available_devices:
            device_names = ["No devices found"]
        else:
            device_names = [f"{d['index']}: {d['name']} ({d['resolution']})" for d in self.available_devices]
        
        self.device_dropdown['values'] = device_names
        
        # Set current selection
        current_index = self.capture_device_index.get()
        matching_devices = [i for i, d in enumerate(self.available_devices) if d['index'] == current_index]
        if matching_devices:
            self.device_dropdown.current(matching_devices[0])
        elif self.available_devices:
            self.device_dropdown.current(0)
            self.capture_device_index.set(self.available_devices[0]['index'])
        
        # Update resolution dropdown for selected device
        self.update_resolution_dropdown()

    def update_resolution_dropdown(self):
        """Update the resolution dropdown for the current device"""
        device_index = self.capture_device_index.get()
        
        # Probe resolutions in background to avoid UI freeze
        def probe_thread():
            resolutions = self.probe_device_resolutions(device_index)
            self.root.after(0, lambda: self._update_resolution_dropdown_ui(resolutions))
        
        self.resolution_dropdown['values'] = ["Scanning..."]
        self.resolution_dropdown.current(0)
        
        thread = threading.Thread(target=probe_thread, daemon=True)
        thread.start()

    def _update_resolution_dropdown_ui(self, resolutions):
        """Update resolution dropdown UI (called from main thread)"""
        self.available_resolutions = resolutions
        
        if not resolutions:
            res_names = ["Default"]
        else:
            res_names = [f"{w}x{h}" for w, h in resolutions]
        
        self.resolution_dropdown['values'] = res_names
        
        # Try to select saved resolution
        saved_res = self.capture_resolution.get()
        if saved_res in res_names:
            self.resolution_dropdown.set(saved_res)
        elif res_names:
            self.resolution_dropdown.current(0)
            self.capture_resolution.set(res_names[0])

    def on_device_selected(self, event=None):
        """Handle device selection from dropdown"""
        selection_index = self.device_dropdown.current()
        if 0 <= selection_index < len(self.available_devices):
            device_index = self.available_devices[selection_index]['index']
            self.capture_device_index.set(device_index)
            
            # Update resolution dropdown
            self.update_resolution_dropdown()
            
            # Restart capture card if currently using it
            if self.capture_source.get() == 'capture_card' and self.capture_card_running:
                self.stop_capture_card()
                self.start_capture_card()

    def on_resolution_selected(self, event=None):
        """Handle resolution selection from dropdown"""
        selection = self.resolution_dropdown.get()
        self.capture_resolution.set(selection)
        
        # Restart capture card if currently running
        if self.capture_source.get() == 'capture_card' and self.capture_card_running:
            self.stop_capture_card()
            self.start_capture_card()

    def build_ui(self):
        # Main container
        main_frame = ttk.Frame(self.ui)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # === ROW 1: Screen Capture & Tracking Settings ===
        sliders_frame = ttk.LabelFrame(main_frame, text="Screen Capture & Tracking Settings")
        sliders_frame.pack(fill=tk.X, padx=5, pady=5)

        sliders_inner = ttk.Frame(sliders_frame)
        sliders_inner.pack(fill=tk.X, padx=5, pady=5)

        # Left column: Capture Source Selection and X, Y, W, H, W==H checkbox
        left_slider_frame = ttk.LabelFrame(sliders_inner, text="Capture Source & Screen Region")
        left_slider_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Capture source selection
        source_frame = ttk.Frame(left_slider_frame)
        source_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(source_frame, text="Capture Source:").pack(side=tk.LEFT)
        ttk.Radiobutton(source_frame, text="Screenshot (MSS)", variable=self.capture_source,
                        value='mss', command=self.on_capture_source_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(source_frame, text="Capture Card", variable=self.capture_source,
                        value='capture_card', command=self.on_capture_source_change).pack(side=tk.LEFT, padx=5)

        # Capture card device selection frame
        self.capture_card_options_frame = ttk.Frame(left_slider_frame)
        
        # Device selection row
        device_row = ttk.Frame(self.capture_card_options_frame)
        device_row.pack(fill=tk.X, pady=2)
        
        ttk.Label(device_row, text="Device:", width=10).pack(side=tk.LEFT)
        self.device_dropdown = ttk.Combobox(device_row, state="readonly", width=25)
        self.device_dropdown.pack(side=tk.LEFT, padx=5)
        self.device_dropdown.bind("<<ComboboxSelected>>", self.on_device_selected)
        
        self.refresh_devices_btn = ttk.Button(device_row, text="Refresh", command=self.refresh_devices)
        self.refresh_devices_btn.pack(side=tk.LEFT, padx=2)
        
        # Resolution selection row
        resolution_row = ttk.Frame(self.capture_card_options_frame)
        resolution_row.pack(fill=tk.X, pady=2)
        
        ttk.Label(resolution_row, text="Resolution:", width=10).pack(side=tk.LEFT)
        self.resolution_dropdown = ttk.Combobox(resolution_row, state="readonly", width=15)
        self.resolution_dropdown.pack(side=tk.LEFT, padx=5)
        self.resolution_dropdown.bind("<<ComboboxSelected>>", self.on_resolution_selected)
        
        # Start/Stop button
        self.capture_card_btn = ttk.Button(resolution_row, text="Start", command=self.toggle_capture_card)
        self.capture_card_btn.pack(side=tk.LEFT, padx=5)
        
        # Capture card status row
        status_row = ttk.Frame(self.capture_card_options_frame)
        status_row.pack(fill=tk.X, pady=2)
        
        self.capture_card_status_label = ttk.Label(status_row, text="Status: Stopped", foreground="gray")
        self.capture_card_status_label.pack(side=tk.LEFT)

        # Screen region sliders (always visible for cropping)
        self.slider_widgets = {}
        self.slider_labels = {}
        
        # Region sliders frame
        region_frame = ttk.Frame(left_slider_frame)
        region_frame.pack(fill=tk.X, pady=5)
        
        self.add_slider(region_frame, "X", self.x_var, 0, 2000)
        self.add_slider(region_frame, "Y", self.y_var, 0, 2000)
        self.add_slider(region_frame, "W", self.w_var, 10, 1920)
        self.add_slider(region_frame, "H", self.h_var, 10, 1080)

        self.checkbox = ttk.Checkbutton(region_frame, text="W == H?", variable=self.square_var,
                                        command=self.toggle_height_slider)
        self.checkbox.pack(anchor='w', pady=2)
        
        self.toggle_height_slider()

        # Right column: Tracking parameters
        right_slider_frame = ttk.LabelFrame(sliders_inner, text="Tracking Parameters")
        right_slider_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Tracking method switch
        method_frame = ttk.Frame(right_slider_frame)
        method_frame.pack(fill=tk.X, pady=5)
        ttk.Label(method_frame, text="Tracking Method:").pack(side=tk.LEFT)
        ttk.Radiobutton(method_frame, text="Classical", variable=self.tracking_method,
                        value='classical', command=self.on_tracking_method_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(method_frame, text="ML (YOLO)", variable=self.tracking_method,
                        value='ml', command=self.on_tracking_method_change).pack(side=tk.LEFT, padx=5)

        # Classical tracking sliders
        self.classical_sliders_frame = ttk.Frame(right_slider_frame)
        self.classical_sliders_frame.pack(fill=tk.X)

        self.add_slider(self.classical_sliders_frame, "Blur", self.blur_var, 1, 31)
        self.add_slider(self.classical_sliders_frame, "Thresh", self.thresh_var, 0.0, 1.0)
        
        # Invert contrast checkbox for classical method
        self.invert_checkbox = ttk.Checkbutton(self.classical_sliders_frame, text="Invert Contrast", 
                                                variable=self.invert_var)
        self.invert_checkbox.pack(anchor='w', pady=2)

        # ML tracking sliders
        self.ml_sliders_frame = ttk.Frame(right_slider_frame)
        self.add_slider(self.ml_sliders_frame, "Confidence", self.confidence_var, 0.0, 1.0)

        # YOLO model path in ML frame
        model_frame = ttk.Frame(self.ml_sliders_frame)
        model_frame.pack(fill=tk.X, pady=5)

        ttk.Label(model_frame, text="YOLO Model:").pack(side=tk.LEFT)
        self.model_path_label = ttk.Label(model_frame, textvariable=self.yolo_model_path, width=25,
                                          relief="sunken", anchor="w")
        self.model_path_label.pack(side=tk.LEFT, padx=5)

        self.browse_btn = ttk.Button(model_frame, text="Browse", command=self.browse_model)
        self.browse_btn.pack(side=tk.LEFT, padx=2)

        self.load_model_btn = ttk.Button(model_frame, text="Load", command=self.load_yolo_model)
        self.load_model_btn.pack(side=tk.LEFT, padx=2)

        self.model_status_label = ttk.Label(model_frame, text="Not loaded", foreground="red")
        self.model_status_label.pack(side=tk.LEFT, padx=5)

        # YOLO FPS label
        yolo_fps_frame = ttk.Frame(self.ml_sliders_frame)
        yolo_fps_frame.pack(fill=tk.X, pady=2)
        self.yolo_fps_label = ttk.Label(yolo_fps_frame, text="YOLO FPS: 0")
        self.yolo_fps_label.pack(side=tk.LEFT)

        # Common sliders
        self.add_slider(right_slider_frame, "Margin", self.margin_var, 1, 100)
        self.add_slider(right_slider_frame, "Area LB", self.area_lb_var, 10, 10000)
        self.add_slider(right_slider_frame, "Area UB", self.area_ub_var, 10000, 100000)

        # === ROW 2: Configuration and Controls side by side ===
        row2_frame = ttk.Frame(main_frame)
        row2_frame.pack(fill=tk.X, padx=5, pady=5)

        # === TEXT INPUTS SECTION ===
        text_frame = ttk.LabelFrame(row2_frame, text="Configuration")
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        text_inner = ttk.Frame(text_frame)
        text_inner.pack(fill=tk.X, padx=5, pady=5)

        left_text_frame = ttk.Frame(text_inner)
        left_text_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        right_text_frame = ttk.Frame(text_inner)
        right_text_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.text_entries = {}
        self.add_text_input(left_text_frame, "Microscopy IP", self.config.get('microscopy_ip', "192.168.0.1"))
        self.add_text_input(left_text_frame, "Port", self.config.get('port', "8080"))
        self.add_text_input(left_text_frame, "Tilt angle start", self.config.get('tilt_start', str(self.tilt_LB)))
        self.add_text_input(left_text_frame, "Tilt angle end", self.config.get('tilt_end', str(self.tilt_UB)))
        self.add_text_input(left_text_frame, "Tilt interval", self.config.get('tilt_interval', str(self.tilt_int)))

        self.add_text_input(right_text_frame, "FOV (nm)", self.config.get('fov', "512"))
        self.add_text_input(right_text_frame, "Delay time (s)", self.config.get('delay', str(self.delay)))
        self.add_text_input(right_text_frame, "Trans threshold (px)", self.config.get('trans_threshold', str(self.transThres)))
        self.add_text_input(right_text_frame, "Multiplier", self.config.get('multiplier', str(self.multiplier)))

        # === CONTROLS SECTION (horizontal layout) ===
        controls_frame = ttk.LabelFrame(row2_frame, text="Controls")
        controls_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)

        controls_inner = ttk.Frame(controls_frame)
        controls_inner.pack(fill=tk.BOTH, padx=5, pady=5)

        # Left side of controls: Connection & Tracking
        controls_left = ttk.Frame(controls_inner)
        controls_left.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Enable checkbox with status indicator
        enable_frame = ttk.Frame(controls_left)
        enable_frame.pack(fill=tk.X, pady=3)

        self.enable_check = ttk.Checkbutton(enable_frame, text="Enable Microscopy Control",
                                            variable=self.enable, command=self.on_enable_change)
        self.enable_check.pack(side=tk.LEFT)

        self.status_canvas = tk.Canvas(enable_frame, width=20, height=20, highlightthickness=0)
        self.status_canvas.pack(side=tk.LEFT, padx=5)
        self.status_circle = self.status_canvas.create_oval(2, 2, 18, 18, fill='yellow')

        # Connection
        connect_frame = ttk.Frame(controls_left)
        connect_frame.pack(fill=tk.X, pady=3)

        self.connect_btn = ttk.Button(connect_frame, text="Connect", command=self.connect_microscope)
        self.connect_btn.pack(side=tk.LEFT, padx=2)

        self.connection_label = ttk.Label(connect_frame, text="Disconnected")
        self.connection_label.pack(side=tk.LEFT, padx=5)

        # Tracking checkbox
        track_frame = ttk.Frame(controls_left)
        track_frame.pack(fill=tk.X, pady=3)

        self.track_check = ttk.Checkbutton(track_frame, text="Track On/Off", variable=self.tracking,
                                           command=self.on_tracking_change)
        self.track_check.pack(side=tk.LEFT)

        # Tracking status label
        self.tracking_status_label = ttk.Label(controls_left, text="Tracking: ON | Idle", foreground="blue")
        self.tracking_status_label.pack(fill=tk.X, pady=3)

        # Separator
        ttk.Separator(controls_inner, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Right side of controls: Stage Controls
        controls_right = ttk.Frame(controls_inner)
        controls_right.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        stage_label = ttk.Label(controls_right, text="Stage Controls", font=('TkDefaultFont', 9, 'bold'))
        stage_label.pack(anchor='w', pady=3)

        self.register_btn = ttk.Button(controls_right, text="Register Starting Pose",
                                       command=self.record_pose, state=tk.DISABLED)
        self.register_btn.pack(fill=tk.X, pady=2)

        self.goto_btn = ttk.Button(controls_right, text="Go to Starting Pose",
                                   command=self.goto_pose, state=tk.DISABLED)
        self.goto_btn.pack(fill=tk.X, pady=2)

        self.tilt_btn = ttk.Button(controls_right, text="Tilt Start/Stop",
                                   command=self.toggle_tilting, state=tk.DISABLED)
        self.tilt_btn.pack(fill=tk.X, pady=2)

        # === FIGURES SECTION ===
        figures_frame = ttk.LabelFrame(main_frame, text="Live View")
        figures_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Info labels next to figures
        info_frame = ttk.Frame(figures_frame)
        info_frame.pack(fill=tk.X, padx=5, pady=2)

        self.obj_pos_label = ttk.Label(info_frame, text="Object @ (x, y): (—, —)")
        self.obj_pos_label.pack(side=tk.LEFT, padx=10)

        self.fps_label = ttk.Label(info_frame, text="FPS: 0")
        self.fps_label.pack(side=tk.LEFT, padx=10)

        # Matplotlib figures in one row
        self.fig, self.ax = plt.subplots(1, 4, figsize=(12, 3))
        self.ims = []

        for i in range(4):
            im = self.ax[i].imshow(np.zeros((self.display_size, self.display_size), dtype='uint8'),
                                   vmin=0, vmax=255, cmap='gray')
            self.ax[i].axis('off')
            self.ims.append(im)

        titles = ['Original', 'Blurred', 'Binary', 'Overlay']
        for i, title in enumerate(titles):
            self.ax[i].set_title(title, fontsize=10)

        self.im1, self.im2, self.im3, self.im4 = self.ims

        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=figures_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # === LOG SECTION ===
        log_frame = ttk.LabelFrame(main_frame, text="Log")
        log_frame.pack(fill=tk.X, padx=5, pady=5)

        self.log_box = tk.Text(log_frame, height=6, width=100, state='disabled')
        self.log_box.pack(padx=5, pady=5)
        
        # Update device dropdown after building UI
        self.update_device_dropdown()

    def on_capture_source_change(self):
        """Handle capture source change"""
        source = self.capture_source.get()
        
        if source == 'mss':
            # Hide capture card options
            self.capture_card_options_frame.pack_forget()
            self.stop_capture_card()
            self.log("Switched to MSS screenshot capture")
        else:
            # Show capture card options
            self.capture_card_options_frame.pack(fill=tk.X, pady=5, before=self.slider_widgets.get("X", None).master if "X" in self.slider_widgets else None)
            self.log("Switched to capture card mode")

    def toggle_capture_card(self):
        """Toggle capture card on/off"""
        if self.capture_card_running:
            self.stop_capture_card()
        else:
            self.start_capture_card()

    def refresh_devices(self):
        """Refresh the list of available capture devices"""
        self.log("Scanning for capture devices...")
        
        # Stop current capture if running
        was_running = self.capture_card_running
        if was_running:
            self.stop_capture_card()
        
        # Scan for devices
        self.scan_capture_devices()
        
        # Update dropdown
        self.update_device_dropdown()
        
        # Restart if it was running
        if was_running and self.capture_source.get() == 'capture_card':
            self.start_capture_card()

    def start_capture_card(self):
        """Start capture card capture thread"""
        if self.capture_card_running:
            self.log("Capture card already running")
            return
        
        device_index = self.capture_device_index.get()
        
        # Open capture device with DirectShow (better Windows support)
        self.capture_card = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        
        if not self.capture_card.isOpened():
            # Try without DirectShow
            self.capture_card = cv2.VideoCapture(device_index)
            if not self.capture_card.isOpened():
                self.log(f"Failed to open capture device {device_index}")
                self.capture_card_status_label.config(text="Status: Failed to open", foreground="red")
                return
        
        # Set resolution if specified
        res_str = self.capture_resolution.get()
        if res_str and 'x' in res_str:
            try:
                w, h = map(int, res_str.split('x'))
                self.capture_card.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                self.capture_card.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            except:
                pass
        
        # Set buffer size to minimum to reduce latency
        self.capture_card.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get actual resolution
        width = int(self.capture_card.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture_card.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Test frame grab
        ret, frame = self.capture_card.read()
        if not ret or frame is None:
            self.log(f"Failed to grab test frame from device {device_index}")
            self.capture_card_status_label.config(text="Status: Cannot grab frame", foreground="red")
            self.capture_card.release()
            self.capture_card = None
            return
        
        self.capture_card_running = True
        self.capture_card_thread = threading.Thread(target=self.capture_card_loop, daemon=True)
        self.capture_card_thread.start()
        
        self.capture_card_status_label.config(text=f"Status: Running ({width}x{height})", foreground="green")
        self.capture_card_btn.config(text="Stop")
        self.log(f"Started capture card (device {device_index}, {width}x{height})")

    def stop_capture_card(self):
        """Stop capture card capture thread"""
        self.capture_card_running = False
        
        # Wait for thread to finish
        if self.capture_card_thread is not None:
            self.capture_card_thread.join(timeout=1.0)
            self.capture_card_thread = None
        
        # Release capture device
        if self.capture_card is not None:
            self.capture_card.release()
            self.capture_card = None
        
        with self.capture_card_lock:
            self.capture_card_latest_frame = None
        
        if hasattr(self, 'capture_card_status_label'):
            self.capture_card_status_label.config(text="Status: Stopped", foreground="gray")
        if hasattr(self, 'capture_card_btn'):
            self.capture_card_btn.config(text="Start")

    def capture_card_loop(self):
        """Background thread for capture card frame capture"""
        consecutive_failures = 0
        max_failures = 10
        
        while self.capture_card_running and self.capture_card is not None:
            try:
                ret, frame = self.capture_card.read()
                if ret and frame is not None:
                    with self.capture_card_lock:
                        self.capture_card_latest_frame = frame.copy()
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        self.root.after(0, lambda: self.log("Capture card: too many consecutive failures"))
                        self.root.after(0, lambda: self.capture_card_status_label.config(
                            text="Status: Frame grab failed", foreground="red"))
                        break
                    time.sleep(0.01)
            except Exception as e:
                print(f"Capture card error: {e}")
                time.sleep(0.1)
        
        self.capture_card_running = False
        self.root.after(0, lambda: self.capture_card_btn.config(text="Start"))

    def get_capture_frame(self):
        """Get the current frame from the selected capture source with cropping"""
        if self.capture_source.get() == 'mss':
            # MSS screenshot capture
            x, y = self.x_var.get(), self.y_var.get()
            w = self.w_var.get()
            h = w if self.square_var.get() else self.h_var.get()

            monitor = {"top": y, "left": x, "width": w, "height": h}
            img = self.sct.grab(monitor)
            img_np = np.array(img)[:, :, :3]  # Remove alpha channel, keep BGR
            return img_np
        else:
            # Capture card
            with self.capture_card_lock:
                if self.capture_card_latest_frame is not None:
                    frame = self.capture_card_latest_frame.copy()
                else:
                    # Return a black frame if no capture available
                    return np.zeros((self.img_size, self.img_size, 3), dtype='uint8')
            
            # Apply cropping to capture card frame
            frame_h, frame_w = frame.shape[:2]
            x = min(self.x_var.get(), frame_w - 1)
            y = min(self.y_var.get(), frame_h - 1)
            w = self.w_var.get()
            h = w if self.square_var.get() else self.h_var.get()
            
            # Ensure we don't go out of bounds
            x2 = min(x + w, frame_w)
            y2 = min(y + h, frame_h)
            
            # Crop the frame
            cropped = frame[y:y2, x:x2]
            
            # If cropped region is too small, pad with black
            if cropped.shape[0] < h or cropped.shape[1] < w:
                padded = np.zeros((h, w, 3), dtype='uint8')
                padded[:cropped.shape[0], :cropped.shape[1]] = cropped
                return padded
            
            return cropped

    def browse_model(self):
        filepath = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("YOLO Model", "*.pt"), ("All files", "*.*")]
        )
        if filepath:
            self.yolo_model_path.set(filepath)
            self.model_status_label.config(text="Not loaded", foreground="red")
            self.yolo_model = None
            self.yolo_model_loaded = False
            self.stop_yolo_thread()

    def load_yolo_model(self):
        if not YOLO_AVAILABLE:
            self.log("ultralytics not installed!")
            self.model_status_label.config(text="ultralytics not available", foreground="red")
            return

        model_path = self.yolo_model_path.get()
        if model_path == "No model selected" or not os.path.exists(model_path):
            self.log("Please select a valid model file first!")
            self.model_status_label.config(text="Invalid path", foreground="red")
            return

        self.model_status_label.config(text="Loading...", foreground="orange")
        self.load_model_btn.config(state=tk.DISABLED)
        self.log(f"Loading YOLO model from: {model_path}")

        def load_thread():
            try:
                model = YOLO(model_path)
                self.yolo_model = model
                self.yolo_model_loaded = True
                
                # Update UI in main thread
                self.root.after(0, self._on_model_loaded_success, model_path)
            except Exception as e:
                self.yolo_model = None
                self.yolo_model_loaded = False
                error_msg = str(e)
                self.root.after(0, self._on_model_loaded_failure, error_msg)

        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()

    def _on_model_loaded_success(self, model_path):
        """Called in main thread when model loads successfully"""
        self.model_status_label.config(text="Loaded", foreground="green")
        self.load_model_btn.config(state=tk.NORMAL)
        self.log(f"YOLO model loaded successfully: {model_path}")
        # Start YOLO prediction thread
        self.start_yolo_thread()

    def _on_model_loaded_failure(self, error_msg):
        """Called in main thread when model fails to load"""
        self.model_status_label.config(text="Load failed", foreground="red")
        self.load_model_btn.config(state=tk.NORMAL)
        self.log(f"Failed to load YOLO model: {error_msg}")

    def start_yolo_thread(self):
        """Start the YOLO prediction thread"""
        if self.yolo_thread is not None and self.yolo_thread.is_alive():
            self.log("YOLO thread already running")
            return
        
        if not self.yolo_model_loaded or self.yolo_model is None:
            self.log("YOLO model not loaded, cannot start thread")
            return
        
        self.yolo_running = True
        self.yolo_thread = threading.Thread(target=self.yolo_prediction_loop, daemon=True)
        self.yolo_thread.start()
        self.log("YOLO prediction thread started")

    def stop_yolo_thread(self):
        """Stop the YOLO prediction thread"""
        self.yolo_running = False
        # Clear queues
        while not self.yolo_input_queue.empty():
            try:
                self.yolo_input_queue.get_nowait()
            except queue.Empty:
                break
        while not self.yolo_output_queue.empty():
            try:
                self.yolo_output_queue.get_nowait()
            except queue.Empty:
                break

    def yolo_prediction_loop(self):
        """Background thread for YOLO predictions"""
        self.log("YOLO prediction loop started")
        while self.yolo_running:
            try:
                if self.yolo_model is None:
                    time.sleep(0.1)
                    continue
                    
                # Get input image (blocking with timeout)
                try:
                    img_data = self.yolo_input_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                img_resized, margin, area_lb, area_ub, confidence = img_data

                # Convert to RGB for YOLO (8-bit)
                img_8bit = (img_resized * 255).astype('uint8')
                img_rgb = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)

                # Run prediction
                results = self.yolo_model(img_rgb, conf=confidence, verbose=False)

                filtered_contours = []
                centroids = []
                bboxes = []
                areas = []

                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                            # Check if bbox touches margin
                            if x1 < margin or y1 < margin or x2 > (self.img_size - margin) or y2 > (self.img_size - margin):
                                continue

                            w = x2 - x1
                            h = y2 - y1
                            area = w * h

                            if area_lb <= area <= area_ub:
                                cx = (x1 + x2) // 2
                                cy = (y1 + y2) // 2

                                cnt = np.array([
                                    [[x1, y1]],
                                    [[x2, y1]],
                                    [[x2, y2]],
                                    [[x1, y2]]
                                ])
                                filtered_contours.append(cnt)
                                centroids.append((cx, cy))
                                bboxes.append((x1, y1, x2, y2))
                                areas.append(area)

                largest_index = int(np.argmax(areas)) if areas else None

                # Put result in output queue (non-blocking)
                # Clear old result first
                while not self.yolo_output_queue.empty():
                    try:
                        self.yolo_output_queue.get_nowait()
                    except queue.Empty:
                        break
                
                try:
                    self.yolo_output_queue.put_nowait((filtered_contours, centroids, bboxes, largest_index))
                except queue.Full:
                    pass

                # Update YOLO FPS
                self.yolo_frame_count += 1
                now = time.time()
                if now - self.yolo_last_time >= 1.0:
                    self.yolo_fps = self.yolo_frame_count / (now - self.yolo_last_time)
                    self.yolo_last_time = now
                    self.yolo_frame_count = 0

            except Exception as e:
                print(f"YOLO thread error: {e}")
                time.sleep(0.1)
        
        self.log("YOLO prediction loop ended")

    def add_slider(self, parent, name, var, from_, to):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)

        ttk.Label(row, text=name, width=10).pack(side=tk.LEFT)
        ttk.Label(row, text=f"{from_:.1f}" if isinstance(from_, float) else str(from_), width=5).pack(side=tk.LEFT)

        slider = ttk.Scale(row, from_=from_, to=to, variable=var, orient=tk.HORIZONTAL, length=200)
        slider.pack(side=tk.LEFT)

        ttk.Label(row, text=f"{to:.1f}" if isinstance(to, float) else str(to), width=5).pack(side=tk.LEFT)

        val_label = ttk.Label(row, text=str(var.get()), width=8)
        val_label.pack(side=tk.LEFT, padx=5)

        def update_label(val):
            val_label.config(text=f"{float(val):.2f}" if isinstance(var, tk.DoubleVar) else str(int(float(val))))

        slider.config(command=update_label)
        self.slider_widgets[name] = row
        self.slider_labels[name] = val_label

    def add_text_input(self, parent, name, default=""):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=name, width=20).pack(side=tk.LEFT)
        entry = ttk.Entry(row, width=30)
        entry.insert(0, default)
        entry.pack(side=tk.LEFT, fill=tk.X)
        self.text_entries[name] = entry

    def toggle_height_slider(self):
        if self.square_var.get():
            self.slider_widgets["H"].pack_forget()
        else:
            self.slider_widgets["H"].pack(fill=tk.X)

    def on_tracking_method_change(self):
        method = self.tracking_method.get()
        if method == 'classical':
            self.ml_sliders_frame.pack_forget()
            self.classical_sliders_frame.pack(fill=tk.X)
        else:
            self.classical_sliders_frame.pack_forget()
            self.ml_sliders_frame.pack(fill=tk.X)
            # Start YOLO thread if model is loaded
            if self.yolo_model_loaded and self.yolo_model is not None:
                self.start_yolo_thread()

    def update_status_circle(self):
        if not self.enable.get():
            color = 'yellow'
        elif not self.connected:
            color = 'red'
        else:
            color = 'green'
        self.status_canvas.itemconfig(self.status_circle, fill=color)

    def update_tracking_status_label(self):
        """Update the tracking status label based on current state"""
        tracking_state = "ON" if self.tracking.get() else "OFF"
        
        if self.tilt_status:
            if self.corrected:
                action = "Position corrected"
            else:
                action = "Tilting"
        else:
            action = "Idle"
        
        self.tracking_status_label.config(text=f"Tracking: {tracking_state} | {action}")

    def on_enable_change(self):
        state = 'ENABLED' if self.enable.get() else 'DISABLED'
        self.log(f"Microscopy Control {state}!")

        if not self.enable.get():
            self.connected = False
            self.M = None
            self.connection_label.config(text="Disconnected")

        self.update_status_circle()
        self.update_button_states()

    def on_tracking_change(self):
        state = 'on' if self.tracking.get() else 'off'
        self.log(f"Tracking {state}.")
        self.update_tracking_status_label()

    def connect_microscope(self):
        if not self.enable.get():
            self.log("Enable microscopy control first!")
            return

        if not TEMSCRIPT_AVAILABLE:
            self.log("temscript module not available! Simulating connection...")
            # Simulate connection for testing
            self.connection_label.config(text="Connecting...")
            self.connect_btn.config(state=tk.DISABLED)
            
            def simulate_connect():
                time.sleep(1)
                self.root.after(0, self._on_connect_simulated)
            
            thread = threading.Thread(target=simulate_connect, daemon=True)
            thread.start()
            return

        self.connection_label.config(text="Connecting...")
        self.connect_btn.config(state=tk.DISABLED)
        self.log("Connecting to microscope (timeout=5s)...")

        def connect_thread():
            IP = self.text_entries["Microscopy IP"].get()
            port_str = self.text_entries["Port"].get()
            
            try:
                port = int(port_str)
            except ValueError:
                self.root.after(0, self._on_connect_failure, "Invalid port number")
                return

            # Set socket timeout
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(5)
            
            try:
                microscope = temscript.RemoteMicroscope((IP, port))
                self.M = microscope
                self.connected = True
                self.root.after(0, self._on_connect_success, IP, port)
            except Exception as e:
                self.connected = False
                self.M = None
                error_msg = str(e)
                self.root.after(0, self._on_connect_failure, error_msg)
            finally:
                socket.setdefaulttimeout(old_timeout)

        thread = threading.Thread(target=connect_thread, daemon=True)
        thread.start()

    def _on_connect_success(self, IP, port):
        """Called in main thread on successful connection"""
        self.connection_label.config(text="Connected")
        self.connect_btn.config(state=tk.NORMAL)
        self.log(f"Connected to microscope {IP}:{port}")
        self.update_status_circle()
        self.update_button_states()

    def _on_connect_failure(self, error_msg):
        """Called in main thread on connection failure"""
        self.connection_label.config(text="Failed")
        self.connect_btn.config(state=tk.NORMAL)
        self.log(f"Connection failed: {error_msg}")
        self.update_status_circle()
        self.update_button_states()

    def _on_connect_simulated(self):
        """Called when simulating connection (no temscript)"""
        self.connected = True
        self.connection_label.config(text="Simulated")
        self.connect_btn.config(state=tk.NORMAL)
        self.log("Simulated connection (temscript not available)")
        self.update_status_circle()
        self.update_button_states()

    def update_button_states(self):
        if self.enable.get() and self.connected:
            self.register_btn.config(state=tk.NORMAL)
            self.goto_btn.config(state=tk.NORMAL)
            self.tilt_btn.config(state=tk.NORMAL)
        else:
            self.register_btn.config(state=tk.DISABLED)
            self.goto_btn.config(state=tk.DISABLED)
            self.tilt_btn.config(state=tk.DISABLED)

    def record_pose(self):
        if self.enable.get() and self.connected:
            if self.M is not None:
                self.pos_init = self.M.get_stage_position()
                self.log(f"Current stage recorded: {self.pos_init}")
            else:
                self.pos_init = {'x': 0, 'y': 0, 'z': 0, 'a': 0, 'b': 0}
                self.log(f"Simulated stage recorded: {self.pos_init}")

    def goto_pose(self):
        if self.enable.get() and self.connected and self.pos_init:
            if self.M is not None:
                self.M.set_stage_position(self.pos_init)
            self.log("Going to registered stage position.")

    def toggle_tilting(self):
        self.tilt_trigger = not self.tilt_trigger
        state = 'start' if self.tilt_trigger else 'stop'
        self.log(f"Tilting {state}.")
        self.update_tracking_status_label()

    def log(self, message):
        self.log_box.config(state='normal')
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_box.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_box.see(tk.END)
        self.log_box.config(state='disabled')

    def write_log(self):
        logs = np.array(self.logs)
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.datetime.fromtimestamp(self.time).strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}.csv"
        filepath = os.path.join("logs", filename)
        np.savetxt(filepath, logs, delimiter=",", fmt="%.6f")

    def para_sync(self):
        self.tilt_LB = float(self.text_entries["Tilt angle start"].get())
        self.tilt_UB = float(self.text_entries["Tilt angle end"].get())
        self.tilt_int = float(self.text_entries["Tilt interval"].get())
        self.FOV = float(self.text_entries["FOV (nm)"].get())
        self.delay = float(self.text_entries["Delay time (s)"].get())
        self.transThres = float(self.text_entries["Trans threshold (px)"].get())
        self.multiplier = float(self.text_entries["Multiplier"].get())
        self.scale = self.img_size / self.FOV

    def segmentation_classical(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply invert if checkbox is checked
        if self.invert_var.get():
            img_gray = 1 - img_gray
            
        img_resized = cv2.resize(img_gray, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        blur_k = max(1, self.blur_var.get() // 2 * 2 + 1)
        blurred = cv2.GaussianBlur(img_resized, (blur_k, blur_k), 0)

        thresh_val = self.thresh_var.get()
        _, binary = cv2.threshold(blurred, thresh_val, 1, cv2.THRESH_BINARY)

        binary_8u = (binary * 255).astype('uint8')
        h, w = binary_8u.shape
        margin = self.margin_var.get()
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)

        binary_8u[:margin, :] = 255
        binary_8u[-margin:, :] = 255
        binary_8u[:, :margin] = 255
        binary_8u[:, -margin:] = 255

        binary_copy = binary_8u.copy()
        cv2.floodFill(binary_copy, flood_mask, seedPoint=(0, 0), newVal=128)

        clean_binary = np.where(binary_copy == 255, 1, 0).astype('uint8')

        contours, _ = cv2.findContours(clean_binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area_lb = self.area_lb_var.get()
        area_ub = self.area_ub_var.get()

        filtered_contours = []
        centroids = []
        areas = []
        bboxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area_lb <= area <= area_ub:
                filtered_contours.append(cnt)
                areas.append(area)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((cx, cy))
                    x, y, w, h = cv2.boundingRect(cnt)
                    bboxes.append((x, y, x + w, y + h))

        largest_index = int(np.argmax(areas)) if areas else None

        # Return non-inverted for display
        display_img = 1 - img_resized if self.invert_var.get() else img_resized

        return display_img, clean_binary, blurred, filtered_contours, centroids, largest_index, bboxes

    def segmentation_ml(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # Send image to YOLO thread if model is loaded
        if self.yolo_model_loaded and self.yolo_running:
            margin = self.margin_var.get()
            area_lb = self.area_lb_var.get()
            area_ub = self.area_ub_var.get()
            confidence = self.confidence_var.get()

            # Clear old input and add new one
            while not self.yolo_input_queue.empty():
                try:
                    self.yolo_input_queue.get_nowait()
                except queue.Empty:
                    break
            
            try:
                self.yolo_input_queue.put_nowait((img_resized.copy(), margin, area_lb, area_ub, confidence))
            except queue.Full:
                pass

        # Get latest result from YOLO thread
        filtered_contours = []
        centroids = []
        bboxes = []
        largest_index = None

        try:
            result = self.yolo_output_queue.get_nowait()
            filtered_contours, centroids, bboxes, largest_index = result
            self.latest_yolo_result = result
        except queue.Empty:
            # Use cached result if available
            if self.latest_yolo_result is not None:
                filtered_contours, centroids, bboxes, largest_index = self.latest_yolo_result

        # Create placeholder binary and blurred images
        binary = np.zeros((self.img_size, self.img_size), dtype='uint8')
        blurred = img_resized.copy()

        return img_resized, binary, blurred, filtered_contours, centroids, largest_index, bboxes

    def segmentation(self, img):
        if self.tracking_method.get() == 'ml':
            return self.segmentation_ml(img)
        else:
            return self.segmentation_classical(img)

    def get_img_display(self, img_resized):
        return (cv2.resize(img_resized, (self.display_size, self.display_size),
                           interpolation=cv2.INTER_LINEAR) * 255).astype('uint8')

    def get_img_blur_display(self, img_blur):
        return (cv2.resize(img_blur, (self.display_size, self.display_size),
                           interpolation=cv2.INTER_LINEAR) * 255).astype('uint8')

    def get_img_bw_display_rgb(self, img_bw):
        img_bw_display = (cv2.resize(img_bw, (self.display_size, self.display_size),
                                     interpolation=cv2.INTER_LINEAR) * 255).astype('uint8')
        img_bw_display_bgr = cv2.cvtColor(img_bw_display, cv2.COLOR_GRAY2BGR)

        margin = self.margin_var.get()
        scaled_margin = int(margin * self.display_size / self.img_size)
        cv2.rectangle(img_bw_display_bgr, (scaled_margin, scaled_margin),
                      (self.display_size - scaled_margin - 1, self.display_size - scaled_margin - 1), (0, 0, 255), 1)

        area_lb = self.area_lb_var.get()
        area_ub = self.area_ub_var.get()
        scale = self.display_size / self.img_size
        lb_len = max(2, int((area_lb ** 0.5) * scale))
        ub_len = max(2, int((area_ub ** 0.5) * scale))
        cx, cy = self.display_size // 2, self.display_size // 2

        def draw_centered_box(img, cx, cy, size, color):
            half = size // 2
            pt1 = (cx - half, cy - half)
            pt2 = (cx + half, cy + half)
            cv2.rectangle(img, pt1, pt2, color, 1)

        draw_centered_box(img_bw_display_bgr, cx, cy, lb_len, (0, 255, 0))
        draw_centered_box(img_bw_display_bgr, cx, cy, ub_len, (255, 0, 0))

        return cv2.cvtColor(img_bw_display_bgr, cv2.COLOR_BGR2RGB)

    def get_img_overlay_rgb(self, img_display, contours, centroids, largest_index, bboxes):
        img_overlay = cv2.cvtColor(img_display.copy(), cv2.COLOR_GRAY2BGR)
        scale_x = self.display_size / self.img_size
        scale_y = self.display_size / self.img_size

        use_bbox = self.tracking_method.get() == 'ml'

        for i, (centroid, bbox) in enumerate(zip(centroids, bboxes)):
            cx, cy = centroid
            cx_disp = int(cx * scale_x)
            cy_disp = int(cy * scale_y)

            if i == largest_index:
                contour_color = (0, 127, 255)  # Orange
                centroid_color = (0, 0, 255)  # Red
            else:
                contour_color = (0, 255, 0)  # Green
                centroid_color = (255, 255, 0)  # Cyan

            if use_bbox:
                # Draw bounding box
                x1, y1, x2, y2 = bbox
                x1_disp = int(x1 * scale_x)
                y1_disp = int(y1 * scale_y)
                x2_disp = int(x2 * scale_x)
                y2_disp = int(y2 * scale_y)
                cv2.rectangle(img_overlay, (x1_disp, y1_disp), (x2_disp, y2_disp), contour_color, 2)
            else:
                # Draw contour
                if i < len(contours):
                    cnt = contours[i]
                    cnt_scaled = np.int32(cnt * [scale_x, scale_y])
                    cv2.drawContours(img_overlay, [cnt_scaled], -1, contour_color, 1)

            cv2.circle(img_overlay, (cx_disp, cy_disp), 3, centroid_color, -1)

        return cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)

    def update_image(self):
        try:
            # Get frame from selected capture source
            img_np = self.get_capture_frame()
            
            # Convert to float32 for processing
            img_np = img_np.astype('float32') / 255

            img_resized, img_bw, img_blur, contours, centroids, largest_index, bboxes = self.segmentation(img_np)

            img_display = self.get_img_display(img_resized)
            img_blur_display = self.get_img_blur_display(img_blur)
            img_bw_display_rgb = self.get_img_bw_display_rgb(img_bw)
            img_overlay_rgb = self.get_img_overlay_rgb(img_display, contours, centroids, largest_index, bboxes)

            self.im1.set_data(img_display)
            self.im2.set_data(img_blur_display)
            self.im3.set_data(img_bw_display_rgb)
            self.im4.set_data(img_overlay_rgb)

            if largest_index is not None:
                cx, cy = centroids[largest_index]
                center = np.array([self.img_size / 2, self.img_size / 2])
                transVec = center - np.array([cx, cy])
                self.obj_pos_label.config(text=f"Object @ (x, y): ({cx}, {cy}) of {self.img_size}")
            else:
                transVec = np.array([0, 0])
                self.obj_pos_label.config(text=f"Object @ (x, y): (—, —) of {self.img_size}")

            # Update YOLO FPS label
            if self.tracking_method.get() == 'ml':
                self.yolo_fps_label.config(text=f"YOLO FPS: {self.yolo_fps:.1f}")

            # Microscopy control logic
            if (not self.tilt_status) and self.tilt_trigger:
                self.para_sync()
                self.tilt_status = True
                self.tlist = np.arange(self.tilt_LB, self.tilt_UB + 0.01, self.tilt_int)
                self.log(f"Tilt angle list initialized: {self.tlist}")
                self.tlist = np.concatenate((self.tlist, self.tlist[1:-1][::-1]), axis=0)
                self.tlist = np.tile(self.tlist, 999) / 180 * np.pi
                self.t_counter = 0
                self.time = time.time()
                self.last_tilt_time = time.time()
                self.logs = []
                self.update_tracking_status_label()
            elif self.tilt_status and self.tilt_trigger:
                self.para_sync()
                dt = time.time() - self.last_tilt_time
                if ((dt) > self.delay) and self.tracking.get() and (
                        np.sum(np.abs(transVec)) > self.transThres) and (not self.corrected):
                    self.last_tilt_time = time.time()
                    self.log(
                        f"[{(self.last_tilt_time - self.time):.2f}] Translating the stage by : ({transVec[0]:.1f},{transVec[1]:.1f})")
                    self.para_sync()
                    if self.enable.get() and self.connected and self.M is not None:
                        pos = self.M.get_stage_position()
                        x = pos['x']
                        y = pos['y']
                        self.M.set_stage_position({
                            'y': y - transVec[0] / self.scale * self.multiplier * 1e-9,
                            'x': x - transVec[1] / self.scale * self.multiplier * 1e-9
                        })
                    self.corrected = True
                    self.update_tracking_status_label()

                dt = time.time() - self.last_tilt_time
                if (dt) > self.delay:
                    self.last_tilt_time = time.time()
                    self.log(
                        f"[{(self.last_tilt_time - self.time):.2f}] Going to the next tilt angle: {np.round((self.tlist[self.t_counter] * 180 / np.pi * 100)) / 100}")
                    if self.enable.get() and self.connected and self.M is not None:
                        self.M.set_stage_position({'a': self.tlist[self.t_counter]})
                    self.logs.append(np.array(
                        [self.last_tilt_time - self.time,
                         np.round((self.tlist[self.t_counter] * 180 / np.pi * 100)) / 100]))
                    self.t_counter += 1
                    self.corrected = False
                    self.write_log()
                    self.update_tracking_status_label()

            elif self.tilt_status and (not self.tilt_trigger):
                self.tilt_status = False
                self.tlist = None
                self.t_counter = None
                self.update_tracking_status_label()
            elif (not self.tilt_status) and (not self.tilt_trigger):
                pass

        except Exception as e:
            self.log(f"Error: {e}")

        self.canvas.draw()

        self.frame_count += 1
        now = time.time()
        if now - self.last_time >= 1.0:
            fps = self.frame_count / (now - self.last_time)
            self.fps_label.config(text=f"FPS: {fps:.1f}")
            self.last_time = now
            self.frame_count = 0

        self.root.after(10, self.update_image)


# Launch the app
if __name__ == "__main__":
    root = tk.Tk()
    app = ScreenGrabberApp(root)
    
    # Clean up on close
    def on_closing():
        app.stop_capture_card()
        app.stop_yolo_thread()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()