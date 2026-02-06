# Kotbyou OpenCV Effect

A project to create various visual effects using OpenCV and MediaPipe to detect hand movements and generate special effects.

## Requirements

### Hardware

- **Webcam** - For real-time video capture
- **Computer** with decent CPU and GPU

### Software

- Python 3.7 or higher
- OpenCV
- MediaPipe
- NumPy

## Installation

### Install Dependencies

```bash
pip install opencv-python mediapipe numpy
```

### Prepare Video Files

Place your .mp4 video files in the same folder as the Python scripts.

---

## File Details

### 1. **domain_expansion.py** -  Domain Expansion Gojo

Change background and create murasaki from a sign hand gesture.

#### 📁 Required Files:

- `fireball.mp4` - Fireball video effect
- `bg.mp4` - Background video

#### How to Run:

```bash
python domain_expansion.py
```

#### Customization:

```python
fire_path = 'fireball.mp4'           # Fireball video file name
bg_video_path = 'bg.mp4'             # Background video file name
threshold_value = 40                 # Threshold value for color detection
blur_intensity = (21, 21)            # Blur intensity for smooth edges
```

---

### 2. **kamekame5.py** - 💥 Kamehameha

Create charging energy ball between two hands that scales with distance.

#### 📁 Required Files:

- `blue_ball.mp4` - Energy ball video

#### 🎮 How to Run:

```bash
python kamekame5.py
```

#### Customization:

```python
video_path = 'blue_ball.mp4'         # Energy ball video file name
min_size = 50                         # Minimum ball size
scale_multiplier = 1.2                # Scaling multiplier based on hand distance
threshold_value = 40                  # Threshold value for color detection
```

---

## 🛑 Stopping the Program

Press **Q** on your keyboard to exit the program.

---

*"Byou Togeter"*
