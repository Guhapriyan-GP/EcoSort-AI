"""
AI Battery Detection & Segregation — Live Hackathon Presentation Script
Interactive full-screen HUD demonstration for judges and live evaluation.
Controls:
  [SPACE / RIGHT ARROW / ENTER] : Next Demo Image
  [LEFT ARROW]                  : Previous Image
  [A]                           : Toggle Auto-Slideshow Mode (3s per slide)
  [Q / ESC]                     : Exit Presentation
"""

import os
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = r'D:\Antigravity\Battery_Detection_AI'
MODEL_PATH = os.path.join(BASE_DIR, 'CHAMPION_BATTERY_MODEL.pt')
DEMO_DIR = os.path.join(BASE_DIR, r'live_demo_suite\images_clean')
OUTPUT_DIR = os.path.join(BASE_DIR, r'live_demo_suite\annotated_slides')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("  AI BATTERY HAZARD SEGREGATION — LIVE DEMO PRESENTATION")
print("  Champion Architecture: YOLOv8n Edge Nano (95.5% Precision, 98.7% mAP50)")
print("  Target: Industrial Conveyor Stream at ~50 FPS")
print("=" * 70)

print("\nLoading Champion Model Weights...")
model = YOLO(MODEL_PATH)

# Collect all demo images in sorted order
image_files = sorted([f for f in os.listdir(DEMO_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

if not image_files:
    print(f"Error: No demo images found in {DEMO_DIR}")
    sys.exit(1)

print(f"Loaded {len(image_files)} curated demonstration images.\n")

def render_industrial_hud(orig_img, boxes, latency_ms, filename, index, total):
    target_w, target_h = 1100, 750
    h, w = orig_img.shape[:2]

    # Special handling for Slide 5 Leaderboard (already full-screen graphic)
    if 'Leaderboard' in filename or 'Slide_5' in filename:
        slide = cv2.resize(orig_img, (target_w, target_h))
        nav_txt = "[SPACE/ENTER]: Next Slide  |  [LEFT]: Prev  |  [A]: Auto-Play  |  [Q/ESC]: Exit"
        cv2.putText(slide, nav_txt, (25, target_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 115, 135), 1)
        return slide

    canvas = orig_img.copy()
    
    # 1. Draw detection boxes and industrial corner brackets
    battery_count = 0
    top_conf = 0.0

    for b in boxes:
        battery_count += 1
        conf = float(b.conf[0])
        top_conf = max(top_conf, conf)
        xyxy = b.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        color = (0, 245, 120) # Bright cyber emerald green
        d = min(20, int((x2 - x1) * 0.2), int((y2 - y1) * 0.2))

        # Box fill overlay
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.16, canvas, 0.84, 0, canvas)

        # Thin border
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)

        # Corner brackets
        th = 2
        cv2.line(canvas, (x1, y1), (x1 + d, y1), color, th)
        cv2.line(canvas, (x1, y1), (x1, y1 + d), color, th)
        cv2.line(canvas, (x2, y1), (x2 - d, y1), color, th)
        cv2.line(canvas, (x2, y1), (x2, y1 + d), color, th)
        cv2.line(canvas, (x1, y2), (x1 + d, y2), color, th)
        cv2.line(canvas, (x1, y2), (x1, y2 - d), color, th)
        cv2.line(canvas, (x2, y2), (x2 - d, y2), color, th)
        cv2.line(canvas, (x2, y2), (x2, y2 - d), color, th)

        # Label badge
        lbl = f"BATTERY HAZARD [{conf*100:.1f}%]"
        (tw, th_txt), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(canvas, (x1, max(0, y1 - 26)), (x1 + tw + 10, max(26, y1)), color, -1)
        cv2.putText(canvas, lbl, (x1 + 5, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 20, 10), 2)

    # 2. Scale image to clean presentation size (e.g. 1024x768 canvas with letterboxing)
    target_w, target_h = 1100, 750
    hud = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    hud[:] = (10, 13, 20) # Deep industrial dark blue background

    # Resize image preserving aspect ratio
    scale = min((target_w - 40) / w, (target_h - 180) / h)
    nw, nh = int(w * scale), int(h * scale)
    resized_canvas = cv2.resize(canvas, (nw, nh), interpolation=cv2.INTER_AREA)

    x_offset = (target_w - nw) // 2
    y_offset = 80
    hud[y_offset:y_offset + nh, x_offset:x_offset + nw] = resized_canvas

    # Border around image
    cv2.rectangle(hud, (x_offset - 1, y_offset - 1), (x_offset + nw + 1, y_offset + nh + 1), (30, 40, 60), 2)

    # 3. Header HUD
    cv2.rectangle(hud, (0, 0), (target_w, 65), (14, 18, 28), -1)
    cv2.line(hud, (0, 65), (target_w, 65), (35, 45, 68), 1)

    title_txt = "AI E-WASTE HAZARD INSPECTION — BATTERY SEGREGATION"
    cv2.putText(hud, title_txt, (25, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 245, 140), 2)

    subtitle_txt = f"SLIDE {index+1}/{total} | {filename.replace('_', ' ')}"
    cv2.putText(hud, subtitle_txt, (25, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 200), 1)

    fps_val = int(1000.0 / latency_ms) if latency_ms > 0 else 50
    perf_txt = f"Latency: {latency_ms:.1f}ms  |  Throughput: {fps_val} FPS"
    (pw, _), _ = cv2.getTextSize(perf_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(hud, perf_txt, (target_w - pw - 25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 255), 1)

    # 4. Bottom Actuator Bar
    y_bot = target_h - 85
    cv2.rectangle(hud, (0, y_bot), (target_w, target_h), (14, 18, 28), -1)
    cv2.line(hud, (0, y_bot), (target_w, y_bot), (35, 45, 68), 1)

    if battery_count > 0:
        bar_color = (0, 245, 120)
        status_txt = f"PNEUMATIC DIVERTOR ARMED: {battery_count} BATTERIES ISOLATED (CONF: {top_conf*100:.1f}%)"
        cv2.rectangle(hud, (20, y_bot + 12), (target_w - 20, target_h - 38), (0, 60, 30), -1)
        cv2.rectangle(hud, (20, y_bot + 12), (target_w - 20, target_h - 38), bar_color, 2)
        cv2.putText(hud, status_txt, (35, y_bot + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, bar_color, 2)
    else:
        bar_color = (130, 145, 165)
        status_txt = "ACTUATOR IDLE — ZERO HAZARDS DETECTED (NO FALSE ALARM)"
        cv2.rectangle(hud, (20, y_bot + 12), (target_w - 20, target_h - 38), (20, 25, 35), -1)
        cv2.rectangle(hud, (20, y_bot + 12), (target_w - 20, target_h - 38), (50, 60, 80), 1)
        cv2.putText(hud, status_txt, (35, y_bot + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)

    # Controls hint
    nav_txt = "[SPACE/ENTER]: Next Slide  |  [LEFT]: Prev  |  [A]: Auto-Play  |  [Q/ESC]: Exit"
    cv2.putText(hud, nav_txt, (25, target_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 115, 135), 1)

    return hud

def main():
    window_name = "AI Battery Segregation — Hackathon Live Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 750)

    current_idx = 0
    auto_play = False
    last_advance_time = time.time()

    while True:
        fn = image_files[current_idx]
        img_path = os.path.join(DEMO_DIR, fn)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping unreadable: {fn}")
            current_idx = (current_idx + 1) % len(image_files)
            continue

        # Run model inference on CPU
        t0 = time.time()
        res = model.predict(img, conf=0.25, device='cpu', verbose=False)
        latency_ms = (time.time() - t0) * 1000.0

        boxes = res[0].boxes
        hud_frame = render_industrial_hud(img, boxes, latency_ms, fn, current_idx, len(image_files))

        # Save slide for offline presentation deck
        slide_out_path = os.path.join(OUTPUT_DIR, f"slide_{current_idx+1:02d}.jpg")
        cv2.imwrite(slide_out_path, hud_frame)

        cv2.imshow(window_name, hud_frame)

        # Handle user keyboard input
        wait_ms = 50 if not auto_play else 100
        key = cv2.waitKey(wait_ms) & 0xFF

        if key in [ord('q'), ord('Q'), 27]: # Q or ESC
            print("\nDemo concluded by user.")
            break
        elif key in [ord(' '), 13, 83]: # Space, Enter, Right Arrow
            current_idx = (current_idx + 1) % len(image_files)
            last_advance_time = time.time()
        elif key in [81, ord('b'), ord('B')]: # Left Arrow or B
            current_idx = (current_idx - 1) % len(image_files)
            last_advance_time = time.time()
        elif key in [ord('a'), ord('A')]:
            auto_play = not auto_play
            print(f"Auto-Play: {'ENABLED (3s interval)' if auto_play else 'PAUSED'}")

        # Auto-play advance logic
        if auto_play and (time.time() - last_advance_time >= 3.0):
            current_idx = (current_idx + 1) % len(image_files)
            last_advance_time = time.time()

    cv2.destroyAllWindows()
    print(f"All annotated slides preserved at: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
