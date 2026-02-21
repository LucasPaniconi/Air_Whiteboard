import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
import math

# =============================
# PERFORMANCE
# =============================
CAM_INDEX = 0
CAM_W, CAM_H = 640, 480
TARGET_FPS = 30
PROCESS_EVERY_N_FRAMES = 2  # run MediaPipe every N frames

# =============================
# DRAW / LOOK
# =============================
BRUSH_THICKNESS = 6
ERASE_RADIUS = 55

# Curve settings
STROKE_POINTS = 6            # points kept for curve
CURVE_SAMPLES = 18           # more = smoother, more CPU (12-24 is good)

# Prediction between MediaPipe frames
PREDICT_ALPHA = 0.6          # 0.0..1.0; lower reduces “snappy” prediction
HAND_TIMEOUT_FRAMES = 10

# =============================
# COLOR BAR (top-right)
# =============================
PALETTE = [
    ("Blue",  (255, 0, 0)),
    ("Red",   (0, 0, 255)),
    ("Brown", (19, 69, 139)),
    ("Black", (0, 0, 0)),
    ("Green", (0, 255, 0)),
    ("Pink",  (203, 192, 255)),
]
PALETTE_BOX_W = 90
PALETTE_BOX_H = 46
PALETTE_MARGIN = 10
HOVER_FRAMES_TO_SELECT = 8
REQUIRE_PINKY_UP = True

# =============================
# One-Euro Filter (stabilizer)
# =============================
def _alpha(cutoff, dt):
    # smoothing factor based on cutoff frequency
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)

class OneEuro1D:
    def __init__(self, min_cutoff=1.5, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def __call__(self, x, t):
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = float(x)
            self.dx_prev = 0.0
            return float(x)

        dt = max(1e-4, t - self.t_prev)
        self.t_prev = t

        # derivative
        dx = (float(x) - self.x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        # adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)

        x_hat = a * float(x) + (1 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat

class OneEuro2D:
    def __init__(self, min_cutoff=1.5, beta=0.02, d_cutoff=1.0):
        self.fx = OneEuro1D(min_cutoff, beta, d_cutoff)
        self.fy = OneEuro1D(min_cutoff, beta, d_cutoff)

    def reset(self):
        self.fx.reset()
        self.fy.reset()

    def __call__(self, pt, t):
        x, y = pt
        return (int(self.fx(x, t)), int(self.fy(y, t)))

# =============================
# Curve drawing (Catmull-Rom-ish)
# =============================
def catmull_rom(p0, p1, p2, p3, t):
    # t in [0,1]
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * ((2*p1[0]) + (-p0[0] + p2[0])*t + (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0])*t2 + (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0])*t3)
    y = 0.5 * ((2*p1[1]) + (-p0[1] + p2[1])*t + (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1])*t2 + (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1])*t3)
    return (int(x), int(y))

def draw_smooth_stroke(canvas, pts, color, thickness, samples=18):
    # pts: deque/list of points (>=4 recommended)
    if len(pts) < 2:
        return
    if len(pts) < 4:
        # fallback simple polyline
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i-1], pts[i], color, thickness)
        return

    # Use last 4 points for a smooth segment
    p0, p1, p2, p3 = pts[-4], pts[-3], pts[-2], pts[-1]
    prev = p1
    for i in range(1, samples + 1):
        t = i / samples
        cur = catmull_rom(p0, p1, p2, p3, t)
        cv2.line(canvas, prev, cur, color, thickness)
        prev = cur

# =============================
# MediaPipe
# =============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

TIP = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
MCP = {"thumb": 2}
WRIST = 0

def lm_to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def finger_extended(pts, finger_name, handed_label):
    if finger_name in ["index", "middle", "ring", "pinky"]:
        return pts[TIP[finger_name]][1] < pts[PIP[finger_name]][1]
    if finger_name == "thumb":
        tip = pts[TIP["thumb"]]
        mcp = pts[MCP["thumb"]]
        if handed_label == "Right":
            return tip[0] < mcp[0]
        else:
            return tip[0] > mcp[0]
    return False

def palette_rects(frame_w):
    rects = []
    x2 = frame_w - PALETTE_MARGIN
    x1 = x2 - PALETTE_BOX_W
    y = PALETTE_MARGIN
    for _ in range(len(PALETTE)):
        rects.append((x1, y, x2, y + PALETTE_BOX_H))
        y += PALETTE_BOX_H + 7
    return rects

def point_in_rect(pt, r):
    x, y = pt
    x1, y1, x2, y2 = r
    return x1 <= x <= x2 and y1 <= y <= y2

def clamp_pt(pt, w, h):
    return (max(0, min(w - 1, pt[0])), max(0, min(h - 1, pt[1])))

# =============================
# Hand state
# =============================
def new_hand_state():
    return {
        "seen_frame": -9999,
        "index_det": None,
        "pinky_det": None,
        "wrist_det": None,
        "index_prev_det": None,
        "pinky_prev_det": None,
        "wrist_prev_det": None,
        "vel_index": (0, 0),
        "vel_pinky": (0, 0),
        "vel_wrist": (0, 0),

        "gesture_point": False,
        "gesture_open": False,
        "pinky_up": False,

        "filter_index": OneEuro2D(min_cutoff=1.4, beta=0.02, d_cutoff=1.0),
        "filter_pinky": OneEuro2D(min_cutoff=1.4, beta=0.02, d_cutoff=1.0),
        "filter_wrist": OneEuro2D(min_cutoff=1.2, beta=0.015, d_cutoff=1.0),

        "stroke": deque(maxlen=STROKE_POINTS),  # curve points
    }

hand_state = {"Left": new_hand_state(), "Right": new_hand_state()}
hover_counts = {"Left": (None, 0), "Right": (None, 0)}

# =============================
# Camera
# =============================
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try CAM_INDEX=1")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
try:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
except:
    pass
try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except:
    pass

canvas = None
selected_color_index = 0
current_color = PALETTE[selected_color_index][1]

frame_count = 0
last_result = None
last_handedness = None

last_time = time.time()
fps = 0.0

print("v8:")
print(" - Index-only pointing = draw")
print(" - Open hand = erase")
print(" - Pinky hover over palette (pinky extended) = change color")
print(" - c clear, q quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    if canvas is None:
        canvas = np.zeros_like(frame)

    t_now = time.time()
    frame_count += 1

    # Run mediapipe every N frames
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_result = hands.process(rgb)
        last_handedness = last_result.multi_handedness if last_result else None

        if last_result and last_result.multi_hand_landmarks and last_handedness:
            for hand_lms, hand_info in zip(last_result.multi_hand_landmarks, last_handedness):
                label = hand_info.classification[0].label  # Left / Right
                if label not in hand_state:
                    continue

                pts = [lm_to_px(lm, w, h) for lm in hand_lms.landmark]
                index_tip = pts[TIP["index"]]
                pinky_tip = pts[TIP["pinky"]]
                wrist = pts[WRIST]

                idx_up = finger_extended(pts, "index", label)
                mid_up = finger_extended(pts, "middle", label)
                ring_up = finger_extended(pts, "ring", label)
                pky_up = finger_extended(pts, "pinky", label)
                thb_up = finger_extended(pts, "thumb", label)

                pointing = idx_up and (not mid_up) and (not ring_up) and (not pky_up)
                open_hand = idx_up and mid_up and ring_up and pky_up and thb_up

                st = hand_state[label]
                st["seen_frame"] = frame_count

                st["index_prev_det"] = st["index_det"]
                st["pinky_prev_det"] = st["pinky_det"]
                st["wrist_prev_det"] = st["wrist_det"]

                st["index_det"] = index_tip
                st["pinky_det"] = pinky_tip
                st["wrist_det"] = wrist

                if st["index_prev_det"] is not None:
                    st["vel_index"] = (st["index_det"][0] - st["index_prev_det"][0],
                                      st["index_det"][1] - st["index_prev_det"][1])
                if st["pinky_prev_det"] is not None:
                    st["vel_pinky"] = (st["pinky_det"][0] - st["pinky_prev_det"][0],
                                      st["pinky_det"][1] - st["pinky_prev_det"][1])
                if st["wrist_prev_det"] is not None:
                    st["vel_wrist"] = (st["wrist_det"][0] - st["wrist_prev_det"][0],
                                      st["wrist_det"][1] - st["wrist_prev_det"][1])

                st["gesture_point"] = pointing
                st["gesture_open"] = open_hand
                st["pinky_up"] = pky_up

    # Palette UI
    rects = palette_rects(w)
    for i, (name, bgr) in enumerate(PALETTE):
        x1, y1, x2, y2 = rects[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, -1)
        thick = 3 if i == selected_color_index else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), thick)
        cv2.putText(frame, name, (x1 + 6, y2 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    mode_texts = []

    for label in ["Left", "Right"]:
        st = hand_state[label]

        # If hand not seen recently, disable it
        if frame_count - st["seen_frame"] > HAND_TIMEOUT_FRAMES:
            st["stroke"].clear()
            hover_counts[label] = (None, 0)
            continue

        if st["index_det"] is None or st["pinky_det"] is None or st["wrist_det"] is None:
            continue

        # Predict between mediapipe frames
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            ix = st["index_det"][0] + int(PREDICT_ALPHA * st["vel_index"][0])
            iy = st["index_det"][1] + int(PREDICT_ALPHA * st["vel_index"][1])
            px = st["pinky_det"][0] + int(PREDICT_ALPHA * st["vel_pinky"][0])
            py = st["pinky_det"][1] + int(PREDICT_ALPHA * st["vel_pinky"][1])
            wx = st["wrist_det"][0] + int(PREDICT_ALPHA * st["vel_wrist"][0])
            wy = st["wrist_det"][1] + int(PREDICT_ALPHA * st["vel_wrist"][1])

            index_raw = clamp_pt((ix, iy), w, h)
            pinky_raw = clamp_pt((px, py), w, h)
            wrist_raw = clamp_pt((wx, wy), w, h)
        else:
            index_raw = st["index_det"]
            pinky_raw = st["pinky_det"]
            wrist_raw = st["wrist_det"]

        # One-Euro filter (best smoothing)
        index_pt = st["filter_index"](index_raw, t_now)
        pinky_pt = st["filter_pinky"](pinky_raw, t_now)
        wrist_pt = st["filter_wrist"](wrist_raw, t_now)

        # Pinky hover color selection
        can_select = st["pinky_up"] if REQUIRE_PINKY_UP else True
        hovered = None
        if can_select:
            for ci, r in enumerate(rects):
                if point_in_rect(pinky_pt, r):
                    hovered = ci
                    break

        if hovered is not None:
            prev_ci, prev_ct = hover_counts[label]
            if prev_ci == hovered:
                hover_counts[label] = (hovered, prev_ct + 1)
            else:
                hover_counts[label] = (hovered, 1)

            cv2.circle(frame, pinky_pt, 7, (255, 255, 255), -1)
            x1, y1, x2, y2 = rects[hovered]
            prog = int((hover_counts[label][1] / HOVER_FRAMES_TO_SELECT) * (x2 - x1))
            prog = max(0, min(prog, x2 - x1))
            cv2.rectangle(frame, (x1, y2 + 2), (x1 + prog, y2 + 6), (255, 255, 255), -1)

            if hover_counts[label][1] >= HOVER_FRAMES_TO_SELECT:
                selected_color_index = hovered
                current_color = PALETTE[selected_color_index][1]
                hover_counts[label] = (hovered, 0)
        else:
            hover_counts[label] = (None, 0)

        # Erase / Draw
        if st["gesture_open"]:
            mode_texts.append(f"{label}: ERASE")
            cx = int((wrist_pt[0] + index_pt[0]) / 2)
            cy = int((wrist_pt[1] + index_pt[1]) / 2)

            cv2.circle(frame, (cx, cy), ERASE_RADIUS, (0, 0, 255), 2)
            cv2.circle(canvas, (cx, cy), ERASE_RADIUS, (0, 0, 0), -1)

            st["stroke"].clear()

        elif st["gesture_point"]:
            mode_texts.append(f"{label}: DRAW")
            st["stroke"].append(index_pt)

            # draw smooth curve for last points
            draw_smooth_stroke(canvas, st["stroke"], current_color, BRUSH_THICKNESS, samples=CURVE_SAMPLES)

            cv2.circle(frame, index_pt, 7, (0, 255, 0), -1)
        else:
            mode_texts.append(f"{label}: MOVE")
            st["stroke"].clear()
            cv2.circle(frame, index_pt, 6, (255, 255, 0), -1)

    # Composite canvas
    grayc = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(grayc, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    ink_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    out = cv2.add(frame_bg, ink_fg)

    # FPS
    now = time.time()
    dt = now - last_time
    last_time = now
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

    mode_line = " | ".join(mode_texts) if mode_texts else "No hands"
    cv2.putText(out, mode_line, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(out, f"FPS: {fps:.1f}  Color: {PALETTE[selected_color_index][0]}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Air Whiteboard v8 (OneEuro + Curves)", out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c"):
        canvas[:] = 0
        for k in ["Left", "Right"]:
            hand_state[k] = new_hand_state()
            hover_counts[k] = (None, 0)

cap.release()
cv2.destroyAllWindows()
hands.close()