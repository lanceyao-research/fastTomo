# fastTomo

Fast tomography control for FEI electron microscopy with real-time particle tracking GUI.

![Version](https://img.shields.io/badge/version-0.0.7-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)

## Overview

fastTomo enables automated tilt-series acquisition with real-time particle tracking for electron tomography. The application captures the microscope display (via screen capture or capture card) and uses computer vision to track particles, automatically correcting stage position during tilting.

## Features

- **Dual Capture Modes**: Screen capture (MSS) or capture card input
- **Real-time Particle Tracking**: 
  - Classical method (thresholding + contour detection)
  - ML method (YOLO-based object detection)
- **Automated Stage Control**: Position correction during tilt series
- **Live Visualization**: Original, blurred, binary, and overlay views
- **Configurable Parameters**: All settings saved automatically to `configure.json`

## Requirements

### Core Dependencies
...
numpy
opencv-python
matplotlib
mss
...

### Optional Dependencies
...
temscript          # For microscope control (FEI/ThermoFisher)
ultralytics        # For YOLO-based ML tracking
...

### Installation
...bash
pip install numpy opencv-python matplotlib mss
pip install temscript        # If connecting to microscope
pip install ultralytics      # If using ML tracking
...

## Usage

### Quick Start

1. **Launch the application**:
   ...bash
   python fastTomo.py
   ...

2. **Configure capture source**:
   - **Screenshot (MSS)**: Capture a region of your screen
   - **Capture Card**: Use an external capture device (e.g., HDMI capture card)

3. **Adjust tracking parameters**:
   - Use sliders to tune blur, threshold, and area filters
   - Choose between Classical or ML tracking methods

4. **Connect to microscope** (optional):
   - Enable "Microscopy Control"
   - Enter IP address and port
   - Click "Connect"

5. **Start tilt series**:
   - Register starting pose
   - Configure tilt angles and intervals
   - Click "Tilt Start/Stop"

### Capture Modes

#### Screen Capture (MSS)
- Set X, Y coordinates and Width/Height to capture a screen region
- Ideal when microscope software displays on the same computer

#### Capture Card
- Select device from dropdown and choose resolution
- Click "Start" to begin capture
- Use X, Y, W, H sliders to crop the captured region
- Ideal for external display sources or dedicated capture setups

### Tracking Methods

#### Classical (Default)
- Gaussian blur → Thresholding → Contour detection
- Fast and works well with high-contrast particles
- Adjustable parameters: Blur, Threshold, Invert Contrast

#### ML (YOLO)
- Requires a trained YOLO model (.pt file)
- Better for complex scenes or low-contrast particles
- Runs inference in a separate thread for smooth UI

### Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| FOV (nm) | Field of view in nanometers |
| Tilt angle start/end | Tilt range in degrees |
| Tilt interval | Step size in degrees |
| Delay time | Wait time between tilts (seconds) |
| Trans threshold | Minimum displacement to trigger correction (pixels) |
| Multiplier | Stage movement scaling factor |

## How It Works

1. **Image Acquisition**: Captures microscope display via screen grab or capture card
2. **Particle Detection**: Identifies particles using classical CV or ML methods
3. **Centroid Tracking**: Calculates particle position relative to image center
4. **Stage Correction**: Sends position corrections to maintain particle centering during tilt

This approach is faster than using camera APIs directly, enabling real-time tracking during tomography acquisition.

## Controls

| Control | Function |
|---------|----------|
| Enable Microscopy Control | Activates microscope communication |
| Connect | Establishes connection to microscope |
| Track On/Off | Enables/disables position correction |
| Register Starting Pose | Saves current stage position |
| Go to Starting Pose | Returns to saved position |
| Tilt Start/Stop | Begins/ends automated tilt series |

## File Output

- **configure.json**: Automatically saved settings
- **logs/**: Tilt series logs with timestamps and angles

## Troubleshooting

### Capture Card Issues
- Try different resolutions from the dropdown
- Click "Refresh" to rescan for devices
- Ensure no other application is using the capture device

### Connection Failed
- Verify microscope IP and port
- Check network connectivity
- Ensure temscript server is running on microscope PC

### Tracking Issues
- Adjust blur and threshold for classical method
- Ensure particle is within Area LB/UB bounds
- Try inverting contrast if particle is darker than background

## Author

**Lance Yao**  
Pacific Northwest National Laboratory  
📧 lance.yao@pnnl.gov

## License

[Add your license here]
