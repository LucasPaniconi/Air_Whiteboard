# ✋ Air Whiteboard (Raspberry Pi)

Real-time gesture-controlled whiteboard built on Raspberry Pi using MediaPipe, OpenCV, and optimized hand-tracking techniques.

---

## 🎥 Demo

<p align="center">
  <img src="demo_preview.gif" width="700">
</p>

<p align="center">
  <a href="demo.mp4">
  </a>
</p>

---

## 🚀 Features

- Point (index finger only) to draw
- Open hand to erase (dynamic erase circle)
- Pinky hover over palette to change colors
- Two-hand support (Left / Right tracked independently)
- One-Euro filtering for jitter reduction
- Smooth curve rendering (Catmull-Rom interpolation)
- Frame prediction between MediaPipe updates for higher perceived FPS

---

## 🎮 Controls

- **Index finger only** → Draw  
- **Open hand** → Erase  
- **Pinky hover (extended)** → Change color  
- `c` → Clear canvas  
- `q` → Quit  

---

## 🛠 Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- Raspberry Pi (optimized for low-latency tracking)

---

## 📦 Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python air_whiteboard.py
