import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import mss
import time
import cv2
import os
import datetime

class ScreenGrabberApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fast Tomo v0.0.1")
        self.root.geometry("1000x1400")  # Width x Height in pixels
        self.obj_pos_label = tk.Label(self.root, text="Object @ (x, y): (—, —)")
        self.obj_pos_label.pack()
        
        self.enable = False
        self.M = None
        self.tracking = True         #whether we track the particle and control the M
        self.tilt_trigger = False     #a trigger of the tilt process
        self.tilt_status = False      #whether we are in a tilt process
        self.tlist = None               #the tilt angle list
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
        self.scale = self.img_size/self.FOV
        self.tilt_LB = -50
        self.tilt_UB = 50
        self.tilt_int = 5
        self.delay = 2  #after this time allow to perfome next move
        self.transThres = 25
        
        self.x_var = tk.IntVar(value=0)
        self.y_var = tk.IntVar(value=0)
        self.w_var = tk.IntVar(value=512)
        self.h_var = tk.IntVar(value=512)
        self.blur_var = tk.IntVar(value=15)
        self.thresh_var = tk.DoubleVar(value=0.5)
        self.square_var = tk.BooleanVar(value=True)
        self.margin_var = tk.IntVar(value=5)
        self.area_lb_var = tk.IntVar(value=400)
        self.area_ub_var = tk.IntVar(value=90000)

        ## Slider
        self.slider_frame = tk.Frame(self.root)
        self.slider_frame.pack(fill=tk.X)
        self.make_sliders()

        ## Text box
        self.text_inputs_frame = tk.Frame(self.root)
        self.text_inputs_frame.pack(fill=tk.X)
        self.text_entries = {}
        self.make_text_inputs()

        # Button
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=5)
        self.make_buttons()
        
        self.checkbox = tk.Checkbutton(self.root, text="W == H?", variable=self.square_var,
                                       command=self.toggle_height_slider)
        self.checkbox.pack()

        self.fps_label = tk.Label(self.root, text="FPS: 0")
        self.fps_label.pack()
        
        self.fig, self.ax = plt.subplots(2, 2)
        self.ims = []

        for i in range(2):
            for j in range(2):
                ax_ij = self.ax[i][j]
                im = ax_ij.imshow(np.zeros((self.display_size, self.display_size), dtype='uint8'), vmin=0, vmax=255, cmap='gray')
                ax_ij.axis('off')
                self.ims.append(im)

        # Optional: unpack for easier reference
        self.im1, self.im2, self.im3, self.im4 = self.ims

        # This is the matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # This is the circle at botton
        self.log_box = tk.Text(self.root, height=10, width=80, state='disabled')
        self.log_box.pack(pady=5)
        
        # This is the circle at botton
        self.status_canvas = tk.Canvas(self.root, width=20, height=20, highlightthickness=0)
        self.status_canvas.pack(pady=5)
        self.status_circle = self.status_canvas.create_oval(2, 2, 18, 18, fill= 'green' if self.enable else 'red')

        self.sct = mss.mss()
        self.last_time = time.time()
        self.frame_count = 0

        self.update_image()
        
    def write_log(self):
        logs = np.array(self.logs)
    
        # Ensure the 'logs' directory exists
        os.makedirs("logs", exist_ok=True)

        # Generate filename with today's date and self.time
        timestamp = datetime.datetime.fromtimestamp(self.time).strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp}.csv"
        filepath = os.path.join("logs", filename)

        # Save to CSV
        np.savetxt(filepath, logs, delimiter=",", fmt="%.6f")
        
    
    def log(self, message):
        self.log_box.config(state='normal')
        self.log_box.insert(tk.END, str(message) + '\n')
        self.log_box.see(tk.END)
        self.log_box.config(state='disabled')
    
    def make_buttons(self):
        
        def enable():
            self.enable = not self.enable
            color = 'green' if self.enable else 'red'
            self.status_canvas.itemconfig(self.status_circle, fill=color)
            state = 'ENABLED' if self.enable else 'DISABLED'
            self.log(f"MICROSCOPY CONTROL {state}!")

            if self.enable:
                button2.config(state=tk.DISABLED)
                button3.config(state=tk.DISABLED)
                button5.config(state=tk.DISABLED)
            else:
                button2.config(state=tk.NORMAL)
                button3.config(state=tk.NORMAL)
                button5.config(state=tk.NORMAL)
                self.M = None
        
        def connect():
            IP = self.text_entries["Microscopy IP"].get()
            port = self.text_entries["Port"].get()
            port = int(port)
            
            try:
                if self.enable:
                    import temscript
                    self.M = temscript.RemoteMicroscope((IP, port))
                    self.log(f"Connected to microscopy {IP} via {str(port)}.")
                    button2.config(state=tk.NORMAL)
                    button5.config(state=tk.NORMAL)
                else:
                    self.log(f"Not connecting. You must first enable the control.")
            except Exception as e:
                self.log(f"Error: {e}")
                self.log(f"Microscopy not connected.")
                
        def record_pose():
            if self.enable:
                self.pos_init = self.M.get_stage_position()
            self.log(f"Current stage recorded.")
            self.log(self.pos_init)
            button3.config(state=tk.NORMAL)
            
        def goto_pose():
            if self.enable:
                self.M.set_stage_position(self.pos_init)
            self.log(f"Going to registered stage position.")

        def tracking():
            self.tracking = not self.tracking
            state = 'on' if self.tracking else 'off'
            self.log(f"Tracking {state}.")
            
        def tilting():
            self.tilt_trigger = not self.tilt_trigger
            state = 'start' if self.tilt_trigger else 'stop'
            self.log(f"Tilting {state}.")    

        button0 = tk.Button(self.button_frame, text="ENABLE MICROSCOPY CONTROL (DANGEROUS)", command=lambda: enable())
        button1 = tk.Button(self.button_frame, text="Connect", command=lambda: connect())
        button2 = tk.Button(self.button_frame, text="Register starting pose", command=lambda: record_pose())
        button3 = tk.Button(self.button_frame, text="Go to starting pose", command=lambda: goto_pose())
        button4 = tk.Button(self.button_frame, text="Track on/off", command=lambda: tracking())
        button5 = tk.Button(self.button_frame, text="Tilt start/stop", command=lambda: tilting())

        button0.pack(side=tk.LEFT, padx=5)
        button1.pack(side=tk.LEFT, padx=5)
        button2.pack(side=tk.LEFT, padx=5)
        button3.pack(side=tk.LEFT, padx=5)
        button4.pack(side=tk.LEFT, padx=5)
        button5.pack(side=tk.LEFT, padx=5)
        
    def make_text_inputs(self):
        def add_text_input(name, default=""):
            row = tk.Frame(self.text_inputs_frame)
            row.pack(fill=tk.X)
            label = tk.Label(row, text=name, width=20)
            label.pack(side=tk.LEFT)
            entry = tk.Entry(row, width = 50)
            entry.insert(0, default)
            entry.pack(side=tk.LEFT, fill=tk.X)
            self.text_entries[name] = entry

        add_text_input("Microscopy IP", "192.168.0.1")
        add_text_input("Port", "8080")
        add_text_input("Tilt angle start", str(self.tilt_LB))
        add_text_input("Tilt angle end", str(self.tilt_UB))
        add_text_input("Tilt interval", str(self.tilt_int))
        add_text_input("FOV (nm)", "512")
        add_text_input("Delay time (s)", str(self.delay))
        add_text_input("Trans threshold (px)", str(self.transThres))
        add_text_input("Multiplier", str(self.multiplier))
        
    def para_sync(self):
        self.tilt_LB = float(self.text_entries["Tilt angle start"].get())
        self.tilt_UB = float(self.text_entries["Tilt angle end"].get())
        self.tilt_int = float(self.text_entries["Tilt interval"].get())
        self.FOV = float(self.text_entries["FOV (nm)"].get())
        self.delay = float(self.text_entries["Delay time (s)"].get())
        self.transThres = float(self.text_entries["Trans threshold (px)"].get())
        self.multiplier = float(self.text_entries["Multiplier"].get())
        self.scale = self.img_size/self.FOV
        
    def make_sliders(self):
        self.slider_widgets = {}

        def add_slider(name, var, from_, to):
            row = tk.Frame(self.slider_frame)
            row.pack(fill=tk.X, pady=2)

            # Label for name
            tk.Label(row, text=name, width=8).pack(side=tk.LEFT)

            # Label for min
            tk.Label(row, text=f"{from_:.1f}" if isinstance(from_, float) else str(from_), width=5).pack(side=tk.LEFT)

            # The actual slider
            slider = ttk.Scale(row, from_=from_, to=to, variable=var, orient=tk.HORIZONTAL, length=500)
            slider.pack(side=tk.LEFT)

            # Label for max
            tk.Label(row, text=f"{to:.1f}" if isinstance(to, float) else str(to), width=5).pack(side=tk.LEFT)

            # Live value label
            val_label = tk.Label(row, text=str(var.get()), width=6)
            val_label.pack(side=tk.LEFT, padx=5)

            # Update the label on change
            def update_label(val):
                val_label.config(text=f"{float(val):.2f}" if isinstance(var, tk.DoubleVar) else str(int(float(val))))

            slider.config(command=update_label)

            self.slider_widgets[name] = row

        add_slider("X", self.x_var, 0, 2000)
        add_slider("Y", self.y_var, 0, 2000)
        add_slider("W", self.w_var, 10, 1920)
        add_slider("H", self.h_var, 10, 1080)
        add_slider("Blur", self.blur_var, 1, 31)
        add_slider("Thresh", self.thresh_var, 0.0, 1.0)
        add_slider("Margin", self.margin_var, 1, 100)
        add_slider("Area LB", self.area_lb_var, 10, 10000)
        add_slider("Area UB", self.area_ub_var, 10000, 100000)

        self.toggle_height_slider()

    def toggle_height_slider(self):
        if self.square_var.get():
            self.slider_widgets["H"].pack_forget()
        else:
            self.slider_widgets["H"].pack(fill=tk.X)
    
    def segmentation(self, img):
        img_gray = 1-cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # Blur
        blur_k = max(1, self.blur_var.get() // 2 * 2 + 1)
        blurred = cv2.GaussianBlur(img_resized, (blur_k, blur_k), 0)

        # Threshold
        thresh_val = self.thresh_var.get()
        _, binary = cv2.threshold(blurred, thresh_val, 1, cv2.THRESH_BINARY)

        # Scale binary to 8-bit
        binary_8u = (binary * 255).astype('uint8')
        # Create mask for floodFill (must be 2 pixels larger)
        h, w = binary_8u.shape
        margin = self.margin_var.get()
        flood_mask = np.zeros((h+2, w+2), np.uint8)

        binary_8u[:margin, :] = 255               # Top margin
        binary_8u[-margin:, :] = 255              # Bottom margin
        binary_8u[:, :margin] = 255               # Left margin
        binary_8u[:, -margin:] = 255              # Right margin

        # Flood fill from top-left corner (assuming it's background)
        binary_copy = binary_8u.copy()
        cv2.floodFill(binary_copy, flood_mask, seedPoint=(0, 0), newVal=128)

        # Convert floodfilled region to background (0), rest to 1
        clean_binary = np.where(binary_copy == 255, 1, 0).astype('uint8')

        # Find contours and centroids on clean mask
        contours, _ = cv2.findContours(clean_binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area
        area_lb = self.area_lb_var.get()
        area_ub = self.area_ub_var.get()
        
        filtered_contours = []
        centroids = []
        areas = []

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

        # Get index of largest object (if any)
        largest_index = int(np.argmax(areas)) if areas else None

        return 1-img_resized, clean_binary, blurred, filtered_contours, centroids, largest_index

    def get_img_display(self, img_resized):
        return (cv2.resize(img_resized, (self.display_size, self.display_size), interpolation=cv2.INTER_LINEAR) * 255).astype('uint8')

    def get_img_blur_display(self, img_blur):
        return (cv2.resize(img_blur, (self.display_size, self.display_size), interpolation=cv2.INTER_LINEAR) * 255).astype('uint8')

    def get_img_bw_display_rgb(self, img_bw):
        img_bw_display = (cv2.resize(img_bw, (self.display_size, self.display_size), interpolation=cv2.INTER_LINEAR) * 255).astype('uint8')
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

    def get_img_overlay_rgb(self, img_display, contours, centroids, largest_index):
        img_overlay = cv2.cvtColor(img_display.copy(), cv2.COLOR_GRAY2BGR)
        scale_x = self.display_size / self.img_size
        scale_y = self.display_size / self.img_size

        for i, (cnt, (cx, cy)) in enumerate(zip(contours, centroids)):
            cnt_scaled = np.int32(cnt * [scale_x, scale_y])
            cx_disp = int(cx * scale_x)
            cy_disp = int(cy * scale_y)

            if i == largest_index:
                contour_color = (0, 127, 255)  # Orange
                centroid_color = (0, 0, 255)  # red
            else:
                contour_color = (0, 255, 0)  # Green
                centroid_color = (255, 255, 0) # Cyan

            cv2.drawContours(img_overlay, [cnt_scaled], -1, contour_color, 1)
            cv2.circle(img_overlay, (cx_disp, cy_disp), 2, centroid_color, -1)

        return cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)

    def update_image(self):
        x, y = self.x_var.get(), self.y_var.get()
        w = self.w_var.get()
        h = w if self.square_var.get() else self.h_var.get()

        monitor = {"top": y, "left": x, "width": w, "height": h}
        try:
            # Grab
            img = self.sct.grab(monitor)
            img_np = np.array(img)[:, :, :3].astype('float32')/255
            
            # Process 
            img_resized, img_bw, img_blur, contours, centroids, largest_index = self.segmentation(img_np)
            
            # Plot
            img_display = self.get_img_display(img_resized)
            img_blur_display = self.get_img_blur_display(img_blur)
            img_bw_display_rgb = self.get_img_bw_display_rgb(img_bw)
            img_overlay_rgb = self.get_img_overlay_rgb(img_display, contours, centroids, largest_index)

            self.im1.set_data(img_display)
            self.im2.set_data(img_blur_display)
            self.im3.set_data(img_bw_display_rgb)
            self.im4.set_data(img_overlay_rgb)

            # Report the location
            if largest_index is not None:
                cx, cy = centroids[largest_index]
                center = np.array([self.img_size/2,self.img_size/2])
                transVec = center - np.array([cx,cy])
                self.obj_pos_label.config(text=f"Object @ (x, y): ({cx}, {cy}) of {self.img_size}")
            else:
                transVec = np.array([0,0])
                self.obj_pos_label.config(text=f"Object @ (x, y): (—, —) of {self.img_size}")

            # Microscopy control
            if (not self.tilt_status) and self.tilt_trigger:   #now we have to start to do the tilt 
                self.para_sync()
                self.tilt_status = True #switch the tilt_status on
                self.tlist = np.arange(self.tilt_LB, self.tilt_UB+0.01, self.tilt_int) #initialize a tilt angle list
                self.log(f"Tilt angle list initialized: {self.tlist}")
                self.tlist = np.concatenate((self.tlist, self.tlist[1:-1][::-1]), axis=0)
                self.tlist = np.tile(self.tlist, 999)/180*np.pi
                self.t_counter = 0
                self.time = time.time()
                self.last_tilt_time = time.time()
                self.logs = []
            elif self.tilt_status and self.tilt_trigger:  #now we continue to do the tilt 
                self.para_sync()
                dt = time.time()-self.last_tilt_time
                if ((dt) > self.delay) and self.tracking and (np.sum(np.abs(transVec))>self.transThres) and (not self.corrected):  
                    #do translate only if 1. tilt stablized(delay1) 2.tracking enabled 3.large enough translation 4.haven't translated
                    self.last_tilt_time = time.time()
                    self.log(f"[{(self.last_tilt_time-self.time):.2f}] Translating the stage by : ({transVec[0]},{transVec[1]})")
                    self.para_sync()
                    if self.enable:
                        pos = self.M.get_stage_position()
                        x = pos['x'];y = pos['y']
                        self.M.set_stage_position({'y':y-transVec[0]/self.scale*self.multiplier*1e-9,'x':x-transVec[1]/self.scale*self.multiplier*1e-9})
                    self.corrected = True
                    
                dt = time.time()-self.last_tilt_time
                if (dt) > self.delay:  ##go to the next tilt angle if has spend >delay2 time at this tilt angle
                    self.last_tilt_time = time.time()
                    self.log(f"[{(self.last_tilt_time-self.time):.2f}] Going to the next tilt angle: {np.round((self.tlist[self.t_counter]*180/np.pi*100))/100}")
                    if self.enable:
                        self.M.set_stage_position({'a': tlist[self.t_counter]})
                    self.logs.append(np.array([self.last_tilt_time-self.time, np.round((self.tlist[self.t_counter]*180/np.pi*100))/100]))
                    self.t_counter += 1
                    self.corrected = False
                    self.write_log()
            
            elif self.tilt_status and (not self.tilt_trigger): #user clicked stop
                self.tilt_status = False
                self.tlist = None ## for safety
                self.t_counter = None ## for safety
                #I guess no need to do anything else if the tilt is stopped at this point
            elif (not self.tilt_status) and (not self.tilt_trigger): #default status
                pass #do not need to anything
        
                
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

        self.root.after(1, self.update_image)

# Launch the app
root = tk.Tk()
app = ScreenGrabberApp(root)
root.mainloop()