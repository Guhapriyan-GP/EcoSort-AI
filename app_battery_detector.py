"""
AI Battery Detection & Segregation System — Interactive Web App
Author: Guhapriyan KA & Project Team
Hardware: Intel(R) Arc(TM) 140T GPU (16GB) via torch.xpu
"""

import os
import io
import time
import json
import base64
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import webbrowser
import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, 'training_runs')
TEST_DIR = os.path.join(BASE_DIR, 'test_eval')
CHAMPION_MODEL_PATH = os.path.join(BASE_DIR, 'CHAMPION_BATTERY_MODEL.pt')
CAND3_V8S_PATH = os.path.join(PROJECT_DIR, 'cand_3_yolov8s', 'weights', 'best.pt')
CAND4_11N_PATH = os.path.join(PROJECT_DIR, 'cand_4_yolo11n', 'weights', 'best.pt')
CAND2_11S_PATH = os.path.join(PROJECT_DIR, 'cand_2_yolo11s', 'weights', 'best.pt')

print("=" * 70)
print("  LAUNCHING INTERACTIVE BATTERY DETECTION DESKTOP APP")
print("  Mode: 100% Safe CPU Inference (Zero GPU display flicker)")
print("=" * 70)

# Pre-load strictly custom trained models into memory
MODELS = {}

if os.path.exists(CHAMPION_MODEL_PATH):
    try:
        print(f"Loading Champion: YOLOv8n Edge Nano ({CHAMPION_MODEL_PATH})...")
        MODELS['champion_yolov8n'] = {
            'name': 'Rank 1: YOLOv8n Champion (95.5% Precision, 98.7% mAP50)',
            'model': YOLO(CHAMPION_MODEL_PATH)
        }
    except Exception as e:
        print(f"Could not load Champion: {e}")

if os.path.exists(CAND3_V8S_PATH):
    try:
        print(f"Loading Rank 2: YOLOv8s Deep Capacity ({CAND3_V8S_PATH})...")
        MODELS['yolov8s_deep'] = {
            'name': 'Rank 2: YOLOv8s Deep Capacity (88.7% Precision, 97.1% mAP50)',
            'model': YOLO(CAND3_V8S_PATH)
        }
    except Exception as e:
        print(f"Could not load YOLOv8s: {e}")

if os.path.exists(CAND4_11N_PATH):
    try:
        print(f"Loading Rank 3: YOLO11n Next-Gen Nano ({CAND4_11N_PATH})...")
        MODELS['yolo11n_nano'] = {
            'name': 'Rank 3: YOLO11n Next-Gen Nano (99.6% Precision, 98.8% mAP50)',
            'model': YOLO(CAND4_11N_PATH)
        }
    except Exception as e:
        print(f"Could not load YOLO11n: {e}")

if os.path.exists(CAND2_11S_PATH):
    try:
        print(f"Loading Rank 4: YOLO11s Next-Gen Small ({CAND2_11S_PATH})...")
        MODELS['yolo11s_small'] = {
            'name': 'Rank 4: YOLO11s Next-Gen Small (80.5% Precision, 78.7% mAP50)',
            'model': YOLO(CAND2_11S_PATH)
        }
    except Exception as e:
        print(f"Could not load YOLO11s: {e}")

# Fallback default
default_model_key = list(MODELS.keys())[0] if MODELS else None

def draw_industrial_boxes(img, boxes, model):
    h, w = img.shape[:2]
    annotated = img.copy()
    detections_info = []

    for b in boxes:
        cls_id = int(b.cls[0])
        raw_name = model.names[cls_id]
        conf = float(b.conf[0])
        xyxy = b.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy

        # Enforce bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        color = (0, 245, 120) # Bright emerald cyber green
        d = min(18, int((x2 - x1) * 0.2), int((y2 - y1) * 0.2))

        # Box fill overlay
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.16, annotated, 0.84, 0, annotated)

        # Thin border
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

        # Corner brackets
        th = 2
        cv2.line(annotated, (x1, y1), (x1 + d, y1), color, th)
        cv2.line(annotated, (x1, y1), (x1, y1 + d), color, th)
        cv2.line(annotated, (x2, y1), (x2 - d, y1), color, th)
        cv2.line(annotated, (x2, y1), (x2, y1 + d), color, th)
        cv2.line(annotated, (x1, y2), (x1 + d, y2), color, th)
        cv2.line(annotated, (x1, y2), (x1, y2 - d), color, th)
        cv2.line(annotated, (x2, y2), (x2 - d, y2), color, th)
        cv2.line(annotated, (x2, y2), (x2, y2 - d), color, th)

        # Label badge
        lbl = f"BATTERY HAZARD [{conf*100:.1f}%]"
        (tw, th_txt), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - 28)), (x1 + tw + 12, max(28, y1)), color, -1)
        cv2.putText(annotated, lbl, (x1 + 6, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 20, 10), 2)

        detections_info.append({
            'label': 'BATTERY',
            'confidence': f"{conf*100:.1f}%",
            'bbox': [int(x1), int(y1), int(x2), int(y2)]
        })

    return annotated, detections_info

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Battery Detection & Segregation — Model Testing Dashboard</title>
    <style>
        :root {
            --bg: #090c10;
            --card: #131822;
            --border: #1e2638;
            --accent: #00f28f;
            --accent-glow: rgba(0, 242, 143, 0.25);
            --cyan: #00d2ff;
            --text: #f0f4fc;
            --text-dim: #8b97a8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 30px; }
        .container { max-width: 1300px; margin: 0 auto; }
        header { border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 25px; display: flex; justify-content: space-between; align-items: flex-end; }
        .tag { display: inline-block; background: rgba(0,242,143,0.12); color: var(--accent); padding: 4px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; border: 1px solid var(--accent); margin-bottom: 8px; }
        h1 { font-size: 2rem; color: #fff; }
        p.sub { color: var(--text-dim); font-size: 0.95rem; margin-top: 4px; }

        .layout { display: grid; grid-template-columns: 360px 1fr; gap: 25px; }
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 22px; }
        
        .control-group { margin-bottom: 20px; }
        label { display: block; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; color: var(--text-dim); letter-spacing: 0.5px; margin-bottom: 8px; }
        select, input[type=range] { width: 100%; padding: 10px; background: #0c1017; border: 1px solid var(--border); border-radius: 8px; color: #fff; font-size: 0.9rem; }
        .slider-val { display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--cyan); margin-top: 6px; font-weight: 700; }

        .dropzone { border: 2px dashed var(--border); border-radius: 10px; padding: 24px 16px; text-align: center; cursor: pointer; transition: all 0.2s; background: rgba(255,255,255,0.01); user-select: none; }
        .dropzone:hover { border-color: var(--accent); background: rgba(0,242,143,0.04); }
        .dropzone.dragover { border-color: var(--accent) !important; background: rgba(0,242,143,0.15) !important; box-shadow: 0 0 25px rgba(0,242,143,0.4) !important; }
        .dropzone p { color: var(--text-dim); font-size: 0.88rem; margin-top: 8px; }

        .samples-title { font-size: 0.82rem; font-weight: 600; text-transform: uppercase; color: var(--text-dim); margin: 20px 0 10px 0; }
        .sample-btns { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .btn-sample { background: #161c28; border: 1px solid var(--border); color: #c4d0e3; padding: 9px; border-radius: 6px; font-size: 0.8rem; cursor: pointer; transition: all 0.15s; font-weight: 600; }
        .btn-sample:hover { border-color: var(--cyan); color: #fff; background: rgba(0,210,255,0.08); }

        .viewer-box { display: flex; flex-direction: column; min-height: 580px; }
        .image-display { flex-grow: 1; display: flex; justify-content: center; align-items: center; background: #000; border-radius: 8px; overflow: hidden; border: 1px solid var(--border); min-height: 440px; position: relative; transition: all 0.2s; }
        .image-display.dragover { border-color: var(--cyan) !important; box-shadow: 0 0 30px rgba(0,210,255,0.45) !important; background: rgba(0,210,255,0.06) !important; }
        .image-display img { max-width: 100%; max-height: 600px; object-fit: contain; }
        .empty-placeholder { color: var(--text-dim); font-size: 0.95rem; text-align: center; pointer-events: none; }

        .telemetry-bar { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 20px; }
        .stat { background: #0d111a; border: 1px solid var(--border); border-radius: 8px; padding: 12px; text-align: center; }
        .stat .val { font-size: 1.3rem; font-weight: 700; color: var(--accent); }
        .stat .lbl { font-size: 0.72rem; color: var(--text-dim); text-transform: uppercase; margin-top: 2px; }

        .actuator-alert { margin-top: 15px; padding: 14px 18px; border-radius: 8px; font-weight: 700; font-size: 0.92rem; display: flex; justify-content: space-between; align-items: center; }
        .alert-armed { background: rgba(0,242,143,0.12); color: var(--accent); border: 1px solid var(--accent); }
        .alert-idle { background: rgba(139,151,168,0.1); color: var(--text-dim); border: 1px solid var(--border); }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <span class="tag">Production AI Inspector</span>
                <h1>Hazardous Battery Computer Vision Inspection App</h1>
                <p class="sub">Test your actual custom models on raw test images with live bounding boxes, confidence telemetry, and pneumatic sorting trigger simulation.</p>
            </div>
        </header>

        <div class="layout">
            <div class="card">
                <div class="control-group">
                    <label>Select Active Model</label>
                    <select id="modelSelect" onchange="runInference()">
                        <!-- Populated by JS -->
                    </select>
                </div>

                <div class="control-group">
                    <label>Confidence Threshold</label>
                    <input type="range" id="confSlider" min="0.05" max="0.90" step="0.05" value="0.25" oninput="updateConf(this.value)" onchange="runInference()">
                    <div class="slider-val">
                        <span>Sensitivity</span>
                        <span id="confDisplay">0.25 (25%)</span>
                    </div>
                </div>

                <div class="control-group">
                    <label>Feed Input Image</label>
                    <div class="dropzone" id="dropArea" onclick="document.getElementById('fileInput').click()">
                        <svg width="32" height="32" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" style="color:var(--accent);"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg>
                        <p><strong>Click to browse</strong> or drag & drop photo here</p>
                        <p style="font-size:0.75rem;color:#6b788e;margin-top:4px;">Supports JPG, PNG, WEBP, or Paste (Ctrl+V)</p>
                    </div>
                    <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="handleFileInput(event)">
                </div>

                <div class="samples-title">Ordered Slides (Matches Video Script 1-to-1)</div>
                <div class="sample-btns" style="grid-template-columns: 1fr 1fr; gap: 8px;">
                    <button class="btn-sample" onclick="loadSample('Slide_1_Single_Battery_98pct')">Slide 1: Battery (98%)</button>
                    <button class="btn-sample" onclick="loadSample('Slide_2_Multi_Battery_Cluster')">Slide 2: Multi-Cluster</button>
                    <button class="btn-sample" onclick="loadSample('Slide_3_Negative_Metal_Can_Zero_Alarm')">Slide 3: Can (0 Alarm)</button>
                    <button class="btn-sample" onclick="loadSample('Slide_4_Negative_Plastic_Cup_Zero_Alarm')">Slide 4: Plastic (0 Alarm)</button>
                </div>
            </div>

            <div class="card viewer-box">
                <div class="image-display" id="displayArea">
                    <div class="empty-placeholder" id="placeholder">Select a test image from the left or upload your own to view live detections.</div>
                    <img id="resultImage" style="display:none;" src="" alt="Detections View">
                </div>

                <div class="telemetry-bar">
                    <div class="stat">
                        <div class="val" id="statCount">0</div>
                        <div class="lbl">Batteries Isolated</div>
                    </div>
                    <div class="stat">
                        <div class="val" id="statTopConf">0.0%</div>
                        <div class="lbl">Top Confidence</div>
                    </div>
                    <div class="stat">
                        <div class="val" id="statLatency">0.0 ms</div>
                        <div class="lbl">Inference Latency</div>
                    </div>
                    <div class="stat">
                        <div class="val" id="statFPS">0 FPS</div>
                        <div class="lbl">Real-Time Throughput</div>
                    </div>
                </div>

                <div class="actuator-alert alert-idle" id="actuatorAlert">
                    <span>SORTING ACTUATOR STATUS</span>
                    <span id="actuatorStatus">IDLE — NO HAZARD DETECTED</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentBase64Image = null;

        function processFile(file) {
            if (!file) return;
            if (!file.type || !file.type.startsWith('image/')) {
                alert('Please upload a valid image file (JPG, PNG, WEBP).');
                return;
            }
            const reader = new FileReader();
            reader.onload = function(evt) {
                currentBase64Image = evt.target.result;
                document.getElementById('actuatorStatus').innerText = 'ANALYZING UPLOADED IMAGE...';
                runInference();
            };
            reader.readAsDataURL(file);
        }

        function handleFileInput(e) {
            const file = e.target.files[0];
            if (file) {
                processFile(file);
                e.target.value = ''; // Reset input to allow re-uploading same file
            }
        }

        function setupDragAndDrop() {
            const dropArea = document.getElementById('dropArea');
            const displayArea = document.getElementById('displayArea');

            // Prevent default drag behaviors across the entire window
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                window.addEventListener(eventName, preventDefaults, false);
                document.body.addEventListener(eventName, preventDefaults, false);
            });

            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }

            // Highlight drag targets
            ['dragenter', 'dragover'].forEach(eventName => {
                dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
                displayArea.addEventListener(eventName, () => displayArea.classList.add('dragover'), false);
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
                displayArea.addEventListener(eventName, () => displayArea.classList.remove('dragover'), false);
            });

            // Handle drop on dropArea
            dropArea.addEventListener('drop', (e) => {
                const dt = e.dataTransfer;
                if (dt && dt.files && dt.files.length > 0) {
                    processFile(dt.files[0]);
                }
            }, false);

            // Handle drop anywhere on the image viewer
            displayArea.addEventListener('drop', (e) => {
                const dt = e.dataTransfer;
                if (dt && dt.files && dt.files.length > 0) {
                    processFile(dt.files[0]);
                }
            }, false);

            // Global drop on entire document window as a convenient fallback
            window.addEventListener('drop', (e) => {
                const dt = e.dataTransfer;
                if (dt && dt.files && dt.files.length > 0) {
                    processFile(dt.files[0]);
                }
            }, false);

            // Handle clipboard paste (Ctrl+V) from screenshots or clipboard
            window.addEventListener('paste', (e) => {
                const items = (e.clipboardData || e.originalEvent.clipboardData).items;
                for (let item of items) {
                    if (item.kind === 'file' && item.type.startsWith('image/')) {
                        const file = item.getAsFile();
                        if (file) processFile(file);
                        break;
                    }
                }
            });
        }

        window.onload = function() {
            setupDragAndDrop();

            fetch('/api/models')
                .then(r => r.json())
                .then(models => {
                    const sel = document.getElementById('modelSelect');
                    sel.innerHTML = '';
                    for (const [key, val] of Object.entries(models)) {
                        const opt = document.createElement('option');
                        opt.value = key;
                        opt.innerText = val.name;
                        sel.appendChild(opt);
                    }
                    // Load default sample
                    loadSample('01_Primary_Battery_Hazard_98pct');
                });
        };

        function updateConf(val) {
            document.getElementById('confDisplay').innerText = `${parseFloat(val).toFixed(2)} (${Math.round(val*100)}%)`;
        }

        function loadSample(sampleName) {
            fetch(`/api/sample?name=${sampleName}`)
                .then(r => r.json())
                .then(data => {
                    if (data.image) {
                        currentBase64Image = data.image;
                        runInference();
                    }
                });
        }

        function runInference() {
            if (!currentBase64Image) return;

            const modelKey = document.getElementById('modelSelect').value;
            const conf = parseFloat(document.getElementById('confSlider').value);

            fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: currentBase64Image,
                    model: modelKey,
                    conf: conf
                })
            })
            .then(r => r.json())
            .then(res => {
                document.getElementById('placeholder').style.display = 'none';
                const img = document.getElementById('resultImage');
                img.src = res.annotated_image;
                img.style.display = 'block';

                document.getElementById('statCount').innerText = res.count;
                document.getElementById('statTopConf').innerText = res.top_conf;
                document.getElementById('statLatency').innerText = `${res.latency_ms.toFixed(1)} ms`;
                
                const fps = res.latency_ms > 0 ? (1000.0 / res.latency_ms).toFixed(0) : '100+';
                document.getElementById('statFPS').innerText = `${fps} FPS`;

                const alertBox = document.getElementById('actuatorAlert');
                const alertTxt = document.getElementById('actuatorStatus');

                if (res.count > 0) {
                    alertBox.className = 'actuator-alert alert-armed';
                    alertTxt.innerText = `PNEUMATIC DIVERTOR ARMED: ${res.count} BATTERIES SEGREGATED`;
                } else {
                    alertBox.className = 'actuator-alert alert-idle';
                    alertTxt.innerText = 'IDLE — NO HAZARD DETECTED';
                }
            });
        }
    </script>
</body>
</html>
"""

class BatteryAppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/' or parsed.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        elif parsed.path == '/api/models':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            data = {k: {'name': v['name']} for k, v in MODELS.items()}
            self.wfile.write(json.dumps(data).encode('utf-8'))
        elif parsed.path == '/api/sample':
            qs = urllib.parse.parse_qs(parsed.query)
            sample_name = qs.get('name', ['Slide_1_Single_Battery_98pct'])[0]
            
            # Map sample name to file (.jpg or .png)
            sample_path = os.path.join(TEST_DIR, f"{sample_name}.jpg")
            if not os.path.exists(sample_path):
                sample_path = os.path.join(TEST_DIR, f"{sample_name}.png")
            if not os.path.exists(sample_path):
                sample_path = os.path.join(TEST_DIR, sample_name)
            if not os.path.exists(sample_path):
                sample_path = os.path.join(TEST_DIR, "Slide_1_Single_Battery_98pct.jpg")

            if os.path.exists(sample_path):
                ext = 'png' if sample_path.endswith('.png') else 'jpeg'
                with open(sample_path, 'rb') as f:
                    b64 = f"data:image/{ext};base64," + base64.b64encode(f.read()).decode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'image': b64}).encode('utf-8'))
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/predict':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            req_data = json.loads(body.decode('utf-8'))

            b64_str = req_data.get('image', '')
            if ',' in b64_str:
                b64_str = b64_str.split(',', 1)[1]

            img_bytes = base64.b64decode(b64_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            model_key = req_data.get('model', default_model_key)
            conf = float(req_data.get('conf', 0.25))

            model_entry = MODELS.get(model_key, MODELS[default_model_key])
            yolo_model = model_entry['model']

            start_t = time.time()
            results = yolo_model.predict(img, conf=conf, verbose=False, device='cpu')
            latency_ms = (time.time() - start_t) * 1000

            annotated_img, detections = draw_industrial_boxes(img, results[0].boxes, yolo_model)

            _, buf = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            out_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode('utf-8')

            top_conf = detections[0]['confidence'] if detections else "0.0%"

            resp = {
                'annotated_image': out_b64,
                'count': len(detections),
                'top_conf': top_conf,
                'latency_ms': latency_ms,
                'detections': detections
            }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def main():
    port = 8501
    server = HTTPServer(('127.0.0.1', port), BatteryAppHandler)
    url = f"http://localhost:{port}"
    print(f"\n>> BATTERY DETECTOR APP RUNNING at: {url}")
    print(">> Opening browser automatically...")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nApp stopped.")

if __name__ == '__main__':
    main()
