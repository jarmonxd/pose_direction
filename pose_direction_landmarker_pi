# pose_direction_landmarker_pi.py  — Raspberry Pi fast path (MediaPipe Tasks)
import os, sys, json, math, time, platform, cv2
import numpy as np
from collections import deque
from PIL import ImageFont, ImageDraw, Image
import mediapipe as mp

# ========= Configs =========
MODEL_PATH = os.path.join("models", "face_landmarker.task")  # ใส่ไฟล์โมเดลไว้ที่นี่
FRAME_W, FRAME_H = 640, 480
USE_MIRROR = True
DRAW_LANDMARKS = False    # ปิดไว้ให้ลื่น
USE_THAI_TEXT = True      # เปิด/ปิดการวาดฟอนต์ไทย (ปิดแล้วลื่นขึ้น)
FONT_PATH = os.path.join(os.path.dirname(__file__), "THSarabunNew.ttf")

# hysteresis (deg)
YAW_ENTER_DEG = 20;   YAW_EXIT_DEG = 12
PITCH_UP_ENTER_DEG = 12;   PITCH_UP_EXIT_DEG = 7
PITCH_DOWN_ENTER_DEG = -12; PITCH_DOWN_EXIT_DEG = -7

EMA_A = 0.25
DEPTH_DELTA = 0.15
SPEED_STEP = 10
MAX_SPEED = 100

# 3D face model (approx.)
MODEL_3D = np.array([
    (0.0,   0.0,   0.0),    # nose tip
    (0.0,  -63.6, -12.5),   # chin
    (-43.3, 32.7, -26.0),   # left eye outer
    (43.3,  32.7, -26.0),   # right eye outer
    (-28.9,-28.9, -24.1),   # left mouth corner
    (28.9, -28.9, -24.1),   # right mouth corner
], dtype=np.float64)

# Indices (FaceMesh topology — ใช้กับ Landmarker ได้)
IDX_NOSE=4; IDX_CHIN=152; IDX_LEFT_EYE_OUT=263; IDX_RIGHT_EYE_OUT=33; IDX_LEFT_MOUTH=291; IDX_RIGHT_MOUTH=61

# ========= Utils =========
def ema(prev, new, a=EMA_A):
    return new if prev is None else prev*(1-a) + new*a

def rot_to_euler_deg(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.degrees(math.atan2(R[2,1], R[2,2]))  # + = เงย
        yaw   = math.degrees(math.atan2(-R[2,0], sy))     # + = หันขวา
        roll  = math.degrees(math.atan2(R[1,0], R[0,0]))
    else:
        pitch = math.degrees(math.atan2(-R[1,2], R[1,1])); yaw = math.degrees(math.atan2(-R[2,0], sy)); roll = 0
    return pitch, yaw, roll

def draw_text(img_bgr, text, xy, font_size=26, color=(255,255,255), stroke=2, stroke_color=(0,0,0)):
    if not USE_THAI_TEXT:
        cv2.putText(img_bgr, text, (int(xy[0]), int(xy[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return img_bgr
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except:
        font = ImageFont.load_default()
    draw.text(xy, text, font=font, fill=tuple(int(c) for c in color),
              stroke_width=int(stroke), stroke_fill=tuple(int(c) for c in stroke_color))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def has_display():
    if os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY"))

def open_camera():
    is_windows = (os.name == "nt") or ("Windows" in platform.system())
    backend = cv2.CAP_DSHOW if is_windows else cv2.CAP_V4L2
    for idx in [0,1,2,3]:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            cap.set(cv2.CAP_PROP_FPS, 30)
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except:
                pass
            return cap
        cap.release()
    raise SystemExit("Cannot open camera")

# ========= Main =========
def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"ไม่พบโมเดล: {MODEL_PATH}\nดาวน์โหลด face_landmarker.task (short-range, int8) แล้ววางตามพาธนี้ด้วยน้า")

    display_ok = has_display()
    cap = open_camera()

    # intrinsic จากเฟรมแรก
    ok, f0 = cap.read()
    if not ok: raise SystemExit("อ่านเฟรมแรกจากกล้องไม่สำเร็จ")
    if USE_MIRROR: f0 = cv2.flip(f0, 1)
    h0, w0 = f0.shape[:2]
    f = w0 * 1.2
    camM = np.array([[f,0,w0/2],[0,f,h0/2],[0,0,1]], dtype=np.float64)
    dist = np.zeros((4,1))

    # MediaPipe Tasks - Face Landmarker (VIDEO mode)
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    mp_image_cls = mp.Image
    mp_imgfmt = mp.ImageFormat

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrices=False
    )

    yaw_deg_ema = pitch_deg_ema = roll_deg_ema = None
    eye_w_ema = None
    baseline_eye_w = None
    yaw_state, pitch_state = "center", "neutral"
    yaw_bias = 0.0; pitch_bias = 0.0
    cmd_hist = deque(maxlen=5)
    stream_json = False; save_json=False; last_export=None

    current_left_speed = current_right_speed = 0.0
    target_left_speed = target_right_speed = 0.0

    ts_ms = 0

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if USE_MIRROR: frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            if (w != w0) or (h != h0):
                w0, h0 = w, h
                f = w0 * 1.2
                camM = np.array([[f,0,w0/2],[0,f,h0/2],[0,0,1]], dtype=np.float64)

            # convert BGR->RGB (MediaPipe ใช้ SRGB)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp_image_cls(image_format=mp_imgfmt.SRGB, data=rgb)

            ts_ms += 33  # ประมาณ 30 FPS
            result = landmarker.detect_for_video(mp_image, ts_ms)

            out = frame.copy()
            command_key="idle"; turn_key="center"; pitch_key="neutral"; depth_key=None
            command_th="คำสั่ง: -"; turn_th="หัน: -"; pitch_th="ก้ม/เงย: -"; depth_th="ระยะ: -"
            color=(0,255,255)

            if result and result.face_landmarks:
                lms = result.face_landmarks[0]  # NormalizedLandmarkList
                # แปลงจุดที่ต้องใช้เป็นพิกเซล
                idxs = [IDX_NOSE, IDX_CHIN, IDX_LEFT_EYE_OUT, IDX_RIGHT_EYE_OUT, IDX_LEFT_MOUTH, IDX_RIGHT_MOUTH]
                pts2d = np.array([(lms[i].x*w, lms[i].y*h) for i in idxs], dtype=np.float64)

                # PnP → หาท่าศีรษะ
                okp, rvec, tvec = cv2.solvePnP(MODEL_3D, pts2d, camM, dist, flags=cv2.SOLVEPNP_ITERATIVE)
                if okp:
                    R,_ = cv2.Rodrigues(rvec)
                    pitch_deg, yaw_deg, roll_deg = rot_to_euler_deg(R)

                    yaw_deg_ema   = ema(yaw_deg_ema,   yaw_deg)
                    pitch_deg_ema = ema(pitch_deg_ema, pitch_deg)
                    roll_deg_ema  = ema(roll_deg_ema,  roll_deg)

                    yaw_c   = yaw_deg_ema   - yaw_bias if yaw_deg_ema   is not None else None
                    pitch_c = pitch_deg_ema - pitch_bias if pitch_deg_ema is not None else None

                    # hysteresis yaw
                    if yaw_c is not None:
                        if yaw_state == "center":
                            if yaw_c <= -YAW_ENTER_DEG: yaw_state = "right"
                            elif yaw_c >=  YAW_ENTER_DEG: yaw_state = "left"
                        elif yaw_state == "left":
                            if yaw_c < YAW_EXIT_DEG: yaw_state = "center"
                        elif yaw_state == "right":
                            if yaw_c > -YAW_EXIT_DEG: yaw_state = "center"

                    # hysteresis pitch
                    if pitch_c is not None:
                        if pitch_state == "neutral":
                            if   pitch_c >=  PITCH_UP_ENTER_DEG:    pitch_state="up"
                            elif pitch_c <= PITCH_DOWN_ENTER_DEG:    pitch_state="down"
                        elif pitch_state == "up":
                            if pitch_c < PITCH_UP_EXIT_DEG:          pitch_state="neutral"
                        elif pitch_state == "down":
                            if pitch_c > PITCH_DOWN_EXIT_DEG:        pitch_state="neutral"

                    # depth จากความกว้างตา
                    eye_w = np.hypot(pts2d[2,0]-pts2d[3,0], pts2d[2,1]-pts2d[3,1])
                    eye_w_ema = ema(eye_w_ema, eye_w)
                    if baseline_eye_w and eye_w_ema:
                        ratio = eye_w_ema / baseline_eye_w
                        if   ratio >= 1.0+DEPTH_DELTA: depth_th, depth_key = "ระยะ: ใกล้ขึ้น", "near"
                        elif ratio <= 1.0-DEPTH_DELTA: depth_th, depth_key = "ระยะ: ไกลออก", "far"
                        else:                           depth_th, depth_key = "ระยะ: คงที่",  "neutral"

                    # summarize
                    turn_key = "left" if yaw_state=="left" else "right" if yaw_state=="right" else "center"
                    turn_th  = "หัน: ซ้าย" if turn_key=="left" else "หัน: ขวา" if turn_key=="right" else "หัน: ตรง"
                    if   pitch_state=="up":   pitch_key, pitch_th = "up",   "ก้ม/เงย: เงยหน้า"
                    elif pitch_state=="down": pitch_key, pitch_th = "down", "ก้ม/เงย: ก้มหน้า"
                    else:                     pitch_key, pitch_th = "neutral","ก้ม/เงย: ปกติ"

                    # policy
                    if   pitch_key=="up":    command_key, command_th, color = "stop",    "คำสั่ง: หยุด",     (0,200,255)
                    elif pitch_key=="down":  command_key, command_th, color = "forward", "คำสั่ง: เดินหน้า", (0,220,0)
                    else:
                        if   turn_key=="left":  command_key, command_th, color = "left",  "คำสั่ง: ไปซ้าย",(0,200,255)
                        elif turn_key=="right": command_key, command_th, color = "right", "คำสั่ง: ไปขวา",(0,200,255)
                        else:                   command_key, command_th, color = "idle",  "คำสั่ง: คงที่", (0,255,255)

                    # speed ramp
                    if   command_key=="forward": target_left_speed, target_right_speed = MAX_SPEED, MAX_SPEED
                    elif command_key=="left":    target_left_speed, target_right_speed = 0,         MAX_SPEED
                    elif command_key=="right":   target_left_speed, target_right_speed = MAX_SPEED, 0
                    elif command_key=="stop":    target_left_speed, target_right_speed = 0,         0
                    else:                         target_left_speed, target_right_speed = 0,         0

                    for side in ("L","R"):
                        if side=="L":
                            if current_left_speed < target_left_speed:   current_left_speed = min(current_left_speed + SPEED_STEP, target_left_speed)
                            elif current_left_speed > target_left_speed: current_left_speed = max(current_left_speed - SPEED_STEP, target_left_speed)
                        else:
                            if current_right_speed < target_right_speed:   current_right_speed = min(current_right_speed + SPEED_STEP, target_right_speed)
                            elif current_right_speed > target_right_speed: current_right_speed = max(current_right_speed - SPEED_STEP, target_right_speed)

                    cmd_hist.append(command_key)
                    command_key = max(set(cmd_hist), key=cmd_hist.count)

                    # (วาด landmarks ถ้าต้องการ)
                    if DRAW_LANDMARKS:
                        connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
                        for a,b in connections:
                            pa = (int(lms[a].x*w), int(lms[a].y*h))
                            pb = (int(lms[b].x*w), int(lms[b].y*h))
                            cv2.line(out, pa, pb, (80,80,80), 1, cv2.LINE_AA)

                    # overlay ตัวเลข
                    if yaw_deg_ema is not None and pitch_deg_ema is not None and roll_deg_ema is not None:
                        out = draw_text(out, f"yaw:{(yaw_deg_ema-yaw_bias):.1f}°  pitch:{(pitch_deg_ema-pitch_bias):.1f}°  roll:{roll_deg_ema:.1f}°",
                                        (10,24), 22, (0,255,0), 2)

            # UI strings
            out = draw_text(out, command_th, (16,100), 36, (0,255,255), 2)
            out = draw_text(out, turn_th,    (16,160), 26, (255,255,0), 2)
            out = draw_text(out, pitch_th,   (16,188), 26, (255,255,0), 2)
            out = draw_text(out, depth_th,   (16,216), 26, (255,255,0), 2)
            out = draw_text(out, f"ความเร็ว: ซ้าย {int(current_left_speed):3d} ขวา {int(current_right_speed):3d}",
                            (16,244), 26, (0,255,255), 2)
            out = draw_text(out, "q:ออก | c:คาลิเบรต | r:รีเซ็ต | f:กระจก | j:JSON | k:Save JSON",
                            (16, out.shape[0]-32), 22, (200,200,200), 1)

            # show / headless
            if display_ok:
                cv2.imshow("Face Landmarker (Pi) - q to quit", out)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = -1
                time.sleep(0.005)

            if key == ord('q'): break
            elif key == ord('c'):
                if eye_w_ema:    baseline_eye_w = float(eye_w_ema)
                if yaw_deg_ema:  yaw_bias   = float(yaw_deg_ema)
                if pitch_deg_ema:pitch_bias = float(pitch_deg_ema)
            elif key == ord('r'):
                eye_w_ema=None; baseline_eye_w=None
                yaw_deg_ema=pitch_deg_ema=roll_deg_ema=None
                yaw_state="center"; pitch_state="neutral"
                yaw_bias=pitch_bias=0.0; cmd_hist.clear()
                current_left_speed=current_right_speed=0.0
                target_left_speed=target_right_speed=0.0
            elif key == ord('f'):
                global USE_MIRROR
                USE_MIRROR = not USE_MIRROR
            elif key == ord('j'):
                stream_json = not stream_json
            elif key == ord('k'):
                save_json = not save_json

            status = {
                "command": command_key, "turn": turn_key, "pitch": pitch_key, "depth": depth_key,
                "yaw_deg":   round(float(yaw_deg_ema - yaw_bias),1)   if yaw_deg_ema   is not None else None,
                "pitch_deg": round(float(pitch_deg_ema - pitch_bias),1) if pitch_deg_ema is not None else None,
                "eye_ratio": round(float(eye_w_ema)/float(baseline_eye_w),3) if (baseline_eye_w and eye_w_ema) else None,
                "calibrated": baseline_eye_w is not None,
                "left_speed": int(current_left_speed),
                "right_speed": int(current_right_speed),
                "target_left_speed": int(target_left_speed),
                "target_right_speed": int(target_right_speed)
            }
            if stream_json:
                try: print(json.dumps(status, ensure_ascii=False), flush=True)
                except: pass
            if save_json and status != last_export:
                try:
                    with open("pose_status.json","w",encoding="utf-8") as f: json.dump(status,f,ensure_ascii=False)
                    last_export = status
                except: pass

    cap.release()
    if display_ok: cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
