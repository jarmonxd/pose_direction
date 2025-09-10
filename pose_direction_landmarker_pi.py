#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pose_direction_landmarker_pi.py  (Tilt-compensated version)

import os, cv2, time, math, json
import numpy as np
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# ---------- Paths ----------
HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "models", "face_landmarker.task")
FONT_PATH  = os.path.join(HERE, "THSarabunNew.ttf")

# ---------- Camera ----------
CAM_INDEX = 0
FRAME_W, FRAME_H, TARGET_FPS = 640, 480, 30
# ถ้ากล้องติดตั้งเอียงแบบถาวร ใส่ค่าตรงนี้ (องศา, + = หมุนตามเข็มนาฬิกา)
CAMERA_ROLL_DEG = 0.0

# ---------- Thresholds / Filters ----------
YAW_ENTER_DEG = 20;  YAW_EXIT_DEG = 12
PITCH_UP_ENTER_DEG = 12;  PITCH_UP_EXIT_DEG = 7
PITCH_DOWN_ENTER_DEG = -12; PITCH_DOWN_EXIT_DEG = -7
EMA_A = 0.25
DEPTH_DELTA = 0.15

# ---------- Draw helpers ----------
def draw_text_thai(img_bgr, text, xy, font_size=28, color=(255,255,255), stroke=2, stroke_color=(0,0,0)):
    try: font = ImageFont.truetype(FONT_PATH, font_size)
    except:  font = ImageFont.load_default()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    drw = ImageDraw.Draw(pil)
    drw.text(xy, text, font=font, fill=tuple(int(c) for c in color),
             stroke_width=int(stroke), stroke_fill=tuple(int(c) for c in stroke_color))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def draw_panel(img_bgr, p1, p2, color=(0,0,0), alpha=0.55):
    x1,y1 = p1; x2,y2 = p2
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), color, -1)
    return cv2.addWeighted(overlay, alpha, img_bgr, 1-alpha, 0)

def ema(prev, new, a=EMA_A): return new if prev is None else prev*(1-a)+new*a

# ---------- Pose math ----------
def rot_to_euler_deg(R):
    """Extract pitch(+up), yaw(+right), roll from rotation matrix."""
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.degrees(math.atan2(R[2,1], R[2,2]))
        yaw   = math.degrees(math.atan2(-R[2,0], sy))
        roll  = math.degrees(math.atan2(R[1,0], R[0,0]))
    else:
        pitch = math.degrees(math.atan2(-R[1,2], R[1,1]))
        yaw   = math.degrees(math.atan2(-R[2,0], sy))
        roll  = 0.0
    return pitch, yaw, roll

def Rz_deg(deg):
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

# ---------- 3D model & landmark indices ----------
MODEL_3D = np.array([
    (  0.0,   0.0,   0.0),   # nose tip
    (  0.0, -63.6, -12.5),   # chin
    (-43.3,  32.7, -26.0),   # left eye outer
    ( 43.3,  32.7, -26.0),   # right eye outer
    (-28.9, -28.9, -24.1),   # left mouth corner
    ( 28.9, -28.9, -24.1),   # right mouth corner
], dtype=np.float64)
IDX_NOSE=4; IDX_CHIN=152; IDX_LEFT_EYE_OUT=263; IDX_RIGHT_EYE_OUT=33; IDX_LEFT_MOUTH=291; IDX_RIGHT_MOUTH=61

# ---------- MediaPipe Tasks ----------
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import Image as MPImage
from mediapipe import ImageFormat

def build_landmarker(model_path: str):
    base = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base, running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1, min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5, min_tracking_confidence=0.5,
        output_face_blendshapes=False, output_facial_transformation_matrixes=False
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)

def main():
    use_mirror   = True
    show_points  = True
    stream_json  = False
    save_json_on_change = False
    last_export  = None

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS,         TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,  1)
    if not cap.isOpened(): raise SystemExit("Cannot open camera")

    landmarker = build_landmarker(MODEL_PATH)
    t0 = time.perf_counter()

    # --- states ---
    yaw_deg_ema = pitch_deg_ema = roll_deg_ema = None
    eye_w_ema = None; baseline_eye_w = None
    yaw_state, pitch_state = "center", "neutral"
    yaw_bias = 0.0; pitch_bias = 0.0
    # bias ชดเชยมุมกล้อง (เริ่มจากค่าคงที่ด้านบน)
    cam_roll_bias_deg = float(CAMERA_ROLL_DEG)
    cmd_hist = deque(maxlen=5)

    # smooth speed
    current_left_speed = current_right_speed = 0.0
    target_left_speed  = target_right_speed  = 0.0
    SPEED_STEP = 10

    while True:
        cap.grab()
        ok, frame = cap.read()
        if not ok: break
        if use_mirror: frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = MPImage(image_format=ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.perf_counter() - t0)*1000)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        out = frame.copy()

        # defaults
        command_key="idle"; turn_key="center"; pitch_key="neutral"; depth_key=None
        command_th="คำสั่ง: -"; turn_th="หัน: -"; pitch_th="ก้ม/เงย: -"; depth_th="ระยะ: -"
        color=(0,255,255)

        if result and result.face_landmarks:
            lm = result.face_landmarks[0]
            pts2d = np.array([
                (lm[IDX_NOSE].x*w,         lm[IDX_NOSE].y*h),
                (lm[IDX_CHIN].x*w,         lm[IDX_CHIN].y*h),
                (lm[IDX_LEFT_EYE_OUT].x*w, lm[IDX_LEFT_EYE_OUT].y*h),
                (lm[IDX_RIGHT_EYE_OUT].x*w,lm[IDX_RIGHT_EYE_OUT].y*h),
                (lm[IDX_LEFT_MOUTH].x*w,   lm[IDX_LEFT_MOUTH].y*h),
                (lm[IDX_RIGHT_MOUTH].x*w,  lm[IDX_RIGHT_MOUTH].y*h),
            ], dtype=np.float64)

            # PnP
            f = w*1.2
            camM = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float64)
            dist = np.zeros((4,1))
            okp, rvec, tvec = cv2.solvePnP(MODEL_3D, pts2d, camM, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if okp:
                R,_ = cv2.Rodrigues(rvec)

                # --- ชดเชยมุมกล้องเอียง: หมุนแกนกล้องกลับด้วย -cam_roll_bias ---
                R_corr = Rz_deg(-cam_roll_bias_deg) @ R

                # ดึง Euler หลังชดเชยแล้ว
                pitch_deg, yaw_deg, roll_deg = rot_to_euler_deg(R_corr)

                # smoothing
                yaw_deg_ema   = ema(yaw_deg_ema,   yaw_deg)
                pitch_deg_ema = ema(pitch_deg_ema, pitch_deg)
                roll_deg_ema  = ema(roll_deg_ema,  roll_deg)

                # ลบ bias ส่วนบุคคล (ศีรษะปกติ)
                yaw_c   = yaw_deg_ema   - yaw_bias
                pitch_c = pitch_deg_ema - pitch_bias

                # hysteresis: yaw
                if yaw_state=="center":
                    if yaw_c <= -YAW_ENTER_DEG: yaw_state="right"
                    elif yaw_c >=  YAW_ENTER_DEG: yaw_state="left"
                elif yaw_state=="left":
                    if yaw_c <  YAW_EXIT_DEG: yaw_state="center"
                elif yaw_state=="right":
                    if yaw_c > -YAW_EXIT_DEG: yaw_state="center"

                # hysteresis: pitch
                if pitch_state=="neutral":
                    if   pitch_c >=  PITCH_UP_ENTER_DEG:     pitch_state="up"
                    elif pitch_c <= PITCH_DOWN_ENTER_DEG:    pitch_state="down"
                elif pitch_state=="up":
                    if   pitch_c <   PITCH_UP_EXIT_DEG:      pitch_state="neutral"
                elif pitch_state=="down":
                    if   pitch_c >   PITCH_DOWN_EXIT_DEG:    pitch_state="neutral"

                # depth (eye width ratio)
                eye_w = np.hypot(pts2d[2,0]-pts2d[3,0], pts2d[2,1]-pts2d[3,1])
                eye_w_ema = ema(eye_w_ema, eye_w)
                if baseline_eye_w and eye_w_ema:
                    ratio = eye_w_ema / baseline_eye_w
                    if   ratio >= 1.0+DEPTH_DELTA: depth_th, depth_key = "ระยะ: ใกล้ขึ้น", "near"
                    elif ratio <= 1.0-DEPTH_DELTA: depth_th, depth_key = "ระยะ: ไกลออก", "far"
                    else:                           depth_th, depth_key = "ระยะ: คงที่", "neutral"

                # สรุปคำสั่ง
                turn_key = "left" if yaw_state=="left" else "right" if yaw_state=="right" else "center"
                turn_th  = "หัน: ซ้าย" if turn_key=="left" else "หัน: ขวา" if turn_key=="right" else "หัน: ตรง"
                if   pitch_state=="up":   command_key, command_th, color = "stop",    "คำสั่ง: หยุด",(0,200,255)
                elif pitch_state=="down": command_key, command_th, color = "forward", "คำสั่ง: เดินหน้า",(0,220,0)
                else:
                    if   turn_key=="left":  command_key, command_th, color = "left",  "คำสั่ง: ไปซ้าย",(0,200,255)
                    elif turn_key=="right": command_key, command_th, color = "right", "คำสั่ง: ไปขวา",(0,200,255)
                    else:                    command_key, command_th, color = "idle",  "คำสั่ง: คงที่",(0,255,255)

                # เป้าหมายความเร็ว
                if   command_key=="forward": target_left_speed, target_right_speed = 100, 100
                elif command_key=="left":    target_left_speed, target_right_speed =   0, 100
                elif command_key=="right":   target_left_speed, target_right_speed = 100,   0
                else:                        target_left_speed, target_right_speed =   0,   0

                # ไล่ความเร็วเนียน ๆ
                def chase(cur, tgt):
                    if cur < tgt:  return min(cur+SPEED_STEP, tgt)
                    if cur > tgt:  return max(cur-SPEED_STEP, tgt)
                    return cur
                current_left_speed  = chase(current_left_speed,  target_left_speed)
                current_right_speed = chase(current_right_speed, target_right_speed)

                cmd_hist.append(command_key)
                command_key = max(set(cmd_hist), key=cmd_hist.count)

                # จุดอ้างอิงเล็ก ๆ
                if show_points:
                    for p in (IDX_NOSE, IDX_CHIN, IDX_LEFT_EYE_OUT, IDX_RIGHT_EYE_OUT, IDX_LEFT_MOUTH, IDX_RIGHT_MOUTH):
                        cx, cy = int(lm[p].x*w), int(lm[p].y*h)
                        cv2.circle(out, (cx,cy), 2, (0,255,0), -1, cv2.LINE_AA)

                # debug txt
                txt = f"yaw:{yaw_c:+.1f}°  pitch:{pitch_c:+.1f}°  roll_corr:{(roll_deg_ema or 0):+.1f}°  cam_roll_bias:{cam_roll_bias_deg:+.1f}°"
                out = draw_text_thai(out, txt, (10,24), 22, (0,255,0), 2)

        # ---------- UI ----------
        out = draw_panel(out, (8, 90), (w-8, 150), (0,0,0), 0.6)
        out = draw_text_thai(out, command_th, (16, 100), 36, color, 2, (0,0,0))

        out = draw_panel(out, (8, 154), (w-8, 270), (0,0,0), 0.45)
        out = draw_text_thai(out, turn_th,  (16,160), 26, (255,255,0), 2, (0,0,0))
        out = draw_text_thai(out, pitch_th, (16,188), 26, (255,255,0), 2, (0,0,0))
        out = draw_text_thai(out, depth_th, (16,216), 26, (255,255,0), 2, (0,0,0))

        speed_text = f"ความเร็ว: ซ้าย {int(current_left_speed):3d} ขวา {int(current_right_speed):3d}"
        out = draw_text_thai(out, speed_text, (16,244), 26, (0,255,255), 2, (0,0,0))

        bias_text  = f"Bias: yaw {yaw_bias:+.1f}° | pitch {pitch_bias:+.1f}° | cam_roll {cam_roll_bias_deg:+.1f}°"
        out = draw_text_thai(out, bias_text, (16,268), 22, (180,220,255), 2, (0,0,0))

        help_text = "q:ออก | c:คาลิเบรตทั้งหมด | h:คาลิเบรตกล้องเอียง | a/d,yaw  w/s,pitch  z/x,roll | f:กระจก | m:จุด | j:JSON | k:Save JSON | r:รีเซ็ต"
        out = draw_text_thai(out, help_text, (16, h-32), 22, (200,200,200), 1, (0,0,0))

        # เส้นไกด์
        cv2.line(out, (w//2,0), (w//2,h), (90,90,90), 1, cv2.LINE_AA)
        cv2.line(out, (0,h//2), (w,h//2), (90,90,90), 1, cv2.LINE_AA)

        cv2.imshow("Face Landmarker (Pi) - q to quit", out)

        # ---------- Keys ----------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('f'): use_mirror = not use_mirror
        elif key == ord('m'): show_points = not show_points
        elif key == ord('j'): stream_json  = not stream_json
        elif key == ord('k'): save_json_on_change = not save_json_on_change
        elif key == ord('r'):
            eye_w_ema = None; baseline_eye_w = None
            yaw_deg_ema = pitch_deg_ema = roll_deg_ema = None
            yaw_state="center"; pitch_state="neutral"
            yaw_bias=pitch_bias=0.0
            cam_roll_bias_deg = float(CAMERA_ROLL_DEG)
            cmd_hist.clear()
            current_left_speed=current_right_speed=0.0
            target_left_speed=target_right_speed=0.0
        elif key == ord('c'):
            # คาลิเบรตทั้งหมด ณ ท่าปกติ (ชดเชยกล้องเอียงด้วย)
            if eye_w_ema:        baseline_eye_w = float(eye_w_ema)
            if yaw_deg_ema:      yaw_bias   = float(yaw_deg_ema)
            if pitch_deg_ema:    pitch_bias = float(pitch_deg_ema)
            if roll_deg_ema is not None:
                cam_roll_bias_deg = float(cam_roll_bias_deg + (roll_deg_ema))  # auto-align horizon
        elif key == ord('h'):
            # คาลิเบรตเฉพาะ camera roll
            if roll_deg_ema is not None:
                cam_roll_bias_deg = float(cam_roll_bias_deg + (roll_deg_ema))
        # จูนละเอียด bias
        elif key == ord('a'): yaw_bias   -= 1.0
        elif key == ord('d'): yaw_bias   += 1.0
        elif key == ord('w'): pitch_bias -= 1.0
        elif key == ord('s'): pitch_bias += 1.0
        elif key == ord('z'): cam_roll_bias_deg -= 1.0
        elif key == ord('x'): cam_roll_bias_deg += 1.0

        # ---------- JSON ----------
        status = {
            "command": command_key, "turn": turn_key, "pitch": pitch_key, "depth": depth_key,
            "yaw_deg":   round(float(yaw_deg_ema - yaw_bias),1) if yaw_deg_ema   is not None else None,
            "pitch_deg": round(float(pitch_deg_ema - pitch_bias),1) if pitch_deg_ema is not None else None,
            "eye_ratio": round(float(eye_w_ema)/float(baseline_eye_w),3) if (baseline_eye_w and eye_w_ema) else None,
            "calibrated": baseline_eye_w is not None,
            "left_speed": int(current_left_speed), "right_speed": int(current_right_speed),
            "target_left_speed": int(target_left_speed), "target_right_speed": int(target_right_speed),
            "cam_roll_bias_deg": round(float(cam_roll_bias_deg),1)
        }
        if stream_json:
            try: print(json.dumps(status, ensure_ascii=False), flush=True)
            except: pass
        if save_json_on_change and status != last_export:
            try:
                with open("face_status.json", "w", encoding="utf-8") as f:
                    json.dump(status, f, ensure_ascii=False)
                last_export = status
            except: pass

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
