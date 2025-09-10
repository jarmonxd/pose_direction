#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pose_direction_landmarker_pi.py — Pi-optimized + manual, step-by-step calibration (SPACE capture, ENTER next)

import os, cv2, time, math, json, sys
import numpy as np
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# ---------------- Paths & Files ----------------
HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "models", "face_landmarker.task")
FONT_PATH  = os.path.join(HERE, "THSarabunNew.ttf")
CFG_PATH   = os.path.join(HERE, "face_config.json")

# ---------------- Camera ----------------
CAM_INDEX = 0
FRAME_W, FRAME_H, TARGET_FPS = 640, 480, 30
DEFAULT_CAMERA_ROLL_DEG = 0.0   # (+ ตามเข็มถ้ากล้องติดตั้งเอียง)

# ---------------- Defaults (จะถูก override ด้วย config) ----------------
CFG = {
    # hysteresis (องศา) — จะถูกคำนวณใหม่อัตโนมัติจากการคาลิเบรตแบบกดทีละจุด
    "YAW_ENTER_DEG": 20.0, "YAW_EXIT_DEG": 12.0,
    "PITCH_UP_ENTER_DEG": 12.0, "PITCH_UP_EXIT_DEG": 7.0,
    "PITCH_DOWN_ENTER_DEG": -12.0, "PITCH_DOWN_EXIT_DEG": -7.0,

    # แปลงค่าที่หันสุดในการคาลิเบรต → เกณฑ์ใช้งานจริง
    "ENTER_RATIO": 0.60,
    "EXIT_RATIO":  0.60,

    # smoothing & depth
    "EMA_A": 0.25,
    "DEPTH_DELTA": 0.15,

    # camera / bias
    "camera_roll_deg": DEFAULT_CAMERA_ROLL_DEG,
    "yaw_bias": 0.0, "pitch_bias": 0.0,

    # speed profile
    "SPEED_STEP": 10,

    # UI / perf
    "show_points": True,
    "perf_overlay": True,
    "mirror": True
}

def load_cfg():
    try:
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            CFG.update(json.load(f))
    except Exception:
        pass

def save_cfg():
    try:
        with open(CFG_PATH, "w", encoding="utf-8") as f:
            json.dump(CFG, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ---------------- Draw helpers ----------------
def draw_text_thai(img_bgr, text, xy, font_size=28, color=(255,255,255), stroke=2, stroke_color=(0,0,0)):
    try: font = ImageFont.truetype(FONT_PATH, font_size)
    except:  font = ImageFont.load_default()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb); drw = ImageDraw.Draw(pil)
    drw.text(xy, text, font=font, fill=tuple(int(c) for c in color),
             stroke_width=int(stroke), stroke_fill=tuple(int(c) for c in stroke_color))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def draw_panel(img_bgr, p1, p2, color=(0,0,0), alpha=0.55):
    x1,y1 = p1; x2,y2 = p2
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), color, -1)
    return cv2.addWeighted(overlay, alpha, img_bgr, 1-alpha, 0)

# ---------------- Math ----------------
def ema(prev, new, a): return new if prev is None else prev*(1-a)+new*a

def rot_to_euler_deg(R):
    # pitch(+up), yaw(+right), roll
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.degrees(math.atan2(R[2,1], R[2,2]))
        yaw   = math.degrees(math.atan2(-R[2,0], sy))
        roll  = math.degrees(math.atan2(R[1,0], R[0,0]))
    else:
        pitch = math.degrees(math.atan2(-R[1,2], R[1,1])); yaw = math.degrees(math.atan2(-R[2,0], sy)); roll = 0.0
    return pitch, yaw, roll

def Rz_deg(deg):
    rad = math.radians(deg); c,s = math.cos(rad), math.sin(rad)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

# ---------------- 3D model & indices ----------------
MODEL_3D = np.array([
    (  0.0,   0.0,   0.0),   # nose tip
    (  0.0, -63.6, -12.5),   # chin
    (-43.3,  32.7, -26.0),   # left eye outer
    ( 43.3,  32.7, -26.0),   # right eye outer
    (-28.9, -28.9, -24.1),   # left mouth
    ( 28.9, -28.9, -24.1),   # right mouth
], dtype=np.float64)
IDX_NOSE=4; IDX_CHIN=152; IDX_LEFT_EYE_OUT=263; IDX_RIGHT_EYE_OUT=33; IDX_LEFT_MOUTH=291; IDX_RIGHT_MOUTH=61

# ---------------- MediaPipe Tasks ----------------
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

# ---------------- Manual step-by-step Calibration ----------------
class ManualCalib:
    """
    ขั้นตอน: center -> left -> right -> up -> down
    ผู้ใช้กด SPACE เพื่อ 'เก็บตัวอย่าง' ของขั้นนั้น, และกด ENTER เพื่อ 'ยืนยันและไปขั้นถัดไป'
    BACKSPACE ย้อนกลับ, ESC ยกเลิก
    """
    STEPS = ["center","left","right","up","down"]
    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.step_i = 0
        self.samples = {k: [] for k in self.STEPS}

    def start(self):
        self.reset()
        self.active = True

    def is_active(self): return self.active
    def step_name(self): return self.STEPS[self.step_i] if self.active else None
    def can_prev(self): return self.active and self.step_i > 0
    def can_next(self): return self.active and self.step_i < len(self.STEPS)-1

    def add_sample(self, yaw_c, pitch_c, roll_corr, eye_w, base_eye_w):
        if not self.active: return False
        if yaw_c is None or pitch_c is None: return False
        self.samples[self.STEPS[self.step_i]].append(
            (float(yaw_c), float(pitch_c), float(roll_corr or 0.0), float(eye_w or 0.0), float(base_eye_w or 0.0))
        )
        return True

    def next_step(self):
        if not self.active: return
        if self.step_i < len(self.STEPS)-1:
            self.step_i += 1
        else:
            self.finish()

    def prev_step(self):
        if self.can_prev(): self.step_i -= 1

    def finish(self):
        # สรุปค่าเฉลี่ยของแต่ละขั้น
        avg = {}
        for k, arr in self.samples.items():
            if len(arr) == 0: continue
            arr = np.array(arr, dtype=np.float64)
            avg[k] = {
                "yaw":   float(np.mean(arr[:,0])),
                "pitch": float(np.mean(arr[:,1])),
                "roll":  float(np.mean(arr[:,2])),
                "eye_w": float(np.mean(arr[:,3])),
                "base" : float(np.mean(arr[:,4])),
            }

        # ใช้ค่า center ตั้ง bias และกล้องเอียง
        if "center" in avg:
            CFG["yaw_bias"]   = avg["center"]["yaw"]
            CFG["pitch_bias"] = avg["center"]["pitch"]
            CFG["camera_roll_deg"] = float(CFG.get("camera_roll_deg",0.0) + avg["center"]["roll"])

        # แปลงเป็นเกณฑ์ hysteresis จาก max/สมมาตรด้วย ratio
        ENTER_RATIO = float(CFG.get("ENTER_RATIO",0.60))
        EXIT_RATIO  = float(CFG.get("EXIT_RATIO", 0.60))

        def diff(name, axis):
            if name not in avg or "center" not in avg: return None
            return avg[name][axis] - avg["center"][axis]

        yaw_L = diff("left","yaw")    # ลบ → ซ้ายมักเป็นลบ
        yaw_R = diff("right","yaw")   # บวก → ขวามักเป็นบวก
        pit_U = diff("up","pitch")    # บวก
        pit_D = diff("down","pitch")  # ลบ

        if yaw_L is not None and yaw_R is not None:
            yaw_enter = min(abs(yaw_L), abs(yaw_R)) * ENTER_RATIO
            CFG["YAW_ENTER_DEG"] = float(max(5.0, yaw_enter))
            CFG["YAW_EXIT_DEG"]  = float(max(3.0, CFG["YAW_ENTER_DEG"] * EXIT_RATIO))
        if pit_U is not None:
            CFG["PITCH_UP_ENTER_DEG"] = float(max(5.0, pit_U * ENTER_RATIO))
            CFG["PITCH_UP_EXIT_DEG"]  = float(max(3.0, CFG["PITCH_UP_ENTER_DEG"] * EXIT_RATIO))
        if pit_D is not None:
            CFG["PITCH_DOWN_ENTER_DEG"] = float(min(-5.0, pit_D * ENTER_RATIO))  # เป็นลบ
            CFG["PITCH_DOWN_EXIT_DEG"]  = float(CFG["PITCH_DOWN_ENTER_DEG"] * EXIT_RATIO)

        save_cfg()
        self.active = False

    def hint(self):
        if not self.active: return None
        s = self.step_name()
        mapping = {
            "center": "คาลิเบรต: มอง 'ตรง' ให้ศีรษะปกติ",
            "left":   "คาลิเบรต: หัน 'ซ้าย'",
            "right":  "คาลิเบรต: หัน 'ขวา'",
            "up":     "คาลิเบรต: 'เงยหน้า'",
            "down":   "คาลิเบรต: 'ก้มหน้า'",
        }
        n = len(self.samples[s])
        return f"{mapping[s]} | เก็บตัวอย่าง: {n} ครั้ง  (SPACE=เก็บ, ENTER=ถัดไป, BACKSPACE=ย้อน, ESC=ยกเลิก)"

# ---------------- Main ----------------
def main():
    load_cfg()

    # OpenCV perf on Pi
    cv2.setUseOptimized(True)
    try: cv2.setNumThreads(2)
    except: pass

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS,         TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,  1)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except: pass
    if not cap.isOpened(): raise SystemExit("Cannot open camera")

    landmarker = build_landmarker(MODEL_PATH)
    t0 = time.perf_counter(); last_t = t0; fps = 0.0

    # states
    yaw_deg_ema = pitch_deg_ema = roll_deg_ema = None
    eye_w_ema = None; baseline_eye_w = None
    yaw_state, pitch_state = "center", "neutral"
    cmd_hist = deque(maxlen=5)
    current_left_speed = current_right_speed = 0.0
    target_left_speed  = target_right_speed  = 0.0
    stream_json = False; save_json_on_change = False; last_export = None

    calib = ManualCalib()

    last_meas = {"yaw_c":None,"pitch_c":None,"roll_corr":None,"eye_w":None}

    while True:
        cap.grab()
        ok, frame = cap.read()
        if not ok: break
        if CFG["mirror"]: frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = MPImage(image_format=ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.perf_counter() - t0)*1000)
        result = landmarker.detect_for_video(mp_img, ts_ms)

        out = frame.copy()

        command_key="idle"; turn_key="center"; pitch_key="neutral"; depth_key=None
        command_th="คำสั่ง: -"; turn_th="หัน: -"; pitch_th="ก้ม/เงย: -"; depth_th="ระยะ: -"
        color=(0,255,255)

        yaw_c = pitch_c = roll_corr = None

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

            f = w*1.2
            camM = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float64)
            dist = np.zeros((4,1))
            okp, rvec, tvec = cv2.solvePnP(MODEL_3D, pts2d, camM, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if okp:
                R,_ = cv2.Rodrigues(rvec)
                R_corr = Rz_deg(-CFG["camera_roll_deg"]) @ R
                pitch_deg, yaw_deg, roll_deg = rot_to_euler_deg(R_corr)

                yaw_deg_ema   = ema(yaw_deg_ema,   yaw_deg, CFG["EMA_A"])
                pitch_deg_ema = ema(pitch_deg_ema, pitch_deg, CFG["EMA_A"])
                roll_deg_ema  = ema(roll_deg_ema,  roll_deg, CFG["EMA_A"])

                yaw_c   = yaw_deg_ema   - CFG["yaw_bias"]
                pitch_c = pitch_deg_ema - CFG["pitch_bias"]
                roll_corr = roll_deg_ema

                eye_w = np.hypot(pts2d[2,0]-pts2d[3,0], pts2d[2,1]-pts2d[3,1])
                eye_w_ema = ema(eye_w_ema, eye_w, CFG["EMA_A"])

                # update last measurement (ให้ SPACE ดึงไปเก็บได้)
                last_meas.update({"yaw_c":yaw_c, "pitch_c":pitch_c, "roll_corr":roll_corr, "eye_w":eye_w_ema})

                # hysteresis yaw
                if yaw_state=="center":
                    if yaw_c <= -CFG["YAW_ENTER_DEG"]: yaw_state="right"
                    elif yaw_c >=  CFG["YAW_ENTER_DEG"]: yaw_state="left"
                elif yaw_state=="left":
                    if yaw_c <  CFG["YAW_EXIT_DEG"]: yaw_state="center"
                elif yaw_state=="right":
                    if yaw_c > -CFG["YAW_EXIT_DEG"]: yaw_state="center"

                # hysteresis pitch
                if pitch_state=="neutral":
                    if   pitch_c >=  CFG["PITCH_UP_ENTER_DEG"]:   pitch_state="up"
                    elif pitch_c <= CFG["PITCH_DOWN_ENTER_DEG"]:  pitch_state="down"
                elif pitch_state=="up":
                    if   pitch_c <   CFG["PITCH_UP_EXIT_DEG"]:    pitch_state="neutral"
                elif pitch_state=="down":
                    if   pitch_c >   CFG["PITCH_DOWN_EXIT_DEG"]:  pitch_state="neutral"

                # depth
                if baseline_eye_w and eye_w_ema:
                    ratio = eye_w_ema / baseline_eye_w
                    if   ratio >= 1.0+CFG["DEPTH_DELTA"]: depth_th, depth_key = "ระยะ: ใกล้ขึ้น", "near"
                    elif ratio <= 1.0-CFG["DEPTH_DELTA"]: depth_th, depth_key = "ระยะ: ไกลออก", "far"
                    else:                                   depth_th, depth_key = "ระยะ: คงที่", "neutral"

                # command
                turn_key = "left" if yaw_state=="left" else "right" if yaw_state=="right" else "center"
                turn_th  = "หัน: ซ้าย" if turn_key=="left" else "หัน: ขวา" if turn_key=="right" else "หัน: ตรง"
                if   pitch_state=="up":   command_key, command_th, color = "stop",    "คำสั่ง: หยุด",(0,200,255)
                elif pitch_state=="down": command_key, command_th, color = "forward", "คำสั่ง: เดินหน้า",(0,220,0)
                else:
                    if   turn_key=="left":  command_key, command_th, color = "left",  "คำสั่ง: ไปซ้าย",(0,200,255)
                    elif turn_key=="right": command_key, command_th, color = "right", "คำสั่ง: ไปขวา",(0,200,255)
                    else:                    command_key, command_th, color = "idle",  "คำสั่ง: คงที่",(0,255,255)

                # speed chase
                def chase(cur, tgt, step):
                    if cur < tgt:  return min(cur+step, tgt)
                    if cur > tgt:  return max(cur-step, tgt)
                    return cur
                if   command_key=="forward": target_left_speed, target_right_speed = 100, 100
                elif command_key=="left":    target_left_speed, target_right_speed =   0, 100
                elif command_key=="right":   target_left_speed, target_right_speed = 100,   0
                else:                        target_left_speed, target_right_speed =   0,   0
                current_left_speed  = chase(current_left_speed,  target_left_speed,  CFG["SPEED_STEP"])
                current_right_speed = chase(current_right_speed, target_right_speed, CFG["SPEED_STEP"])

                cmd_hist.append(command_key)
                command_key = max(set(cmd_hist), key=cmd_hist.count)

                if CFG["show_points"]:
                    for p in (IDX_NOSE, IDX_CHIN, IDX_LEFT_EYE_OUT, IDX_RIGHT_EYE_OUT, IDX_LEFT_MOUTH, IDX_RIGHT_MOUTH):
                        cx, cy = int(lm[p].x*w), int(lm[p].y*h)
                        cv2.circle(out, (cx,cy), 2, (0,255,0), -1, cv2.LINE_AA)

        # ---------- UI ----------
        now = time.perf_counter()
        dt  = now - last_t
        if dt > 0: fps = 0.9*fps + 0.1*(1.0/dt)
        last_t = now

        if CFG["perf_overlay"]:
            cv2.line(out, (w//2,0), (w//2,h), (90,90,90), 1, cv2.LINE_AA)
            cv2.line(out, (0,h//2), (w,h//2), (90,90,90), 1, cv2.LINE_AA)

        panel_h = 60 if not CFG["perf_overlay"] else 150
        out = draw_panel(out, (8, 90), (w-8, 90+panel_h), (0,0,0), 0.55)
        out = draw_text_thai(out, command_th, (16, 100), 36, color, 2, (0,0,0))

        if CFG["perf_overlay"]:
            info1 = f"yaw:{(yaw_c if yaw_c is not None else 0):+.1f}°  pitch:{(pitch_c if pitch_c is not None else 0):+.1f}°  rollBias:{CFG['camera_roll_deg']:+.1f}°  fps:{fps:.1f}"
            out = draw_text_thai(out, info1, (16, 140), 22, (0,255,0), 2)
            out = draw_panel(out, (8, 154), (w-8, 270), (0,0,0), 0.45)
            out = draw_text_thai(out, ("หัน: ซ้าย" if turn_key=="left" else "หัน: ขวา" if turn_key=="right" else "หัน: ตรง"),  (16,160), 26, (255,255,0), 2, (0,0,0))
            out = draw_text_thai(out, ("ก้ม/เงย: เงยหน้า" if pitch_state=="up" else "ก้ม/เงย: ก้มหน้า" if pitch_state=="down" else "ก้ม/เงย: ปกติ"), (16,188), 26, (255,255,0), 2, (0,0,0))
            out = draw_text_thai(out, depth_th, (16,216), 26, (255,255,0), 2, (0,0,0))
            speed_text = f"ความเร็ว: ซ้าย {int(current_left_speed):3d} ขวา {int(current_right_speed):3d}"
            out = draw_text_thai(out, speed_text, (16,244), 26, (0,255,255), 2, (0,0,0))
            bias_text  = f"Bias yaw {CFG['yaw_bias']:+.1f}° | pitch {CFG['pitch_bias']:+.1f}° | cam_roll {CFG['camera_roll_deg']:+.1f}°"
            out = draw_text_thai(out, bias_text, (16,268), 22, (180,220,255), 2, (0,0,0))

        # คำแนะนำคาลิเบรตแบบกดทีละจุด
        if calib.is_active():
            hint = calib.hint()
            out = draw_panel(out, (8, 8), (w-8, 68), (20,20,20), 0.75)
            out = draw_text_thai(out, hint, (16, 16), 26, (0,255,255), 2)

        help_text = "q:ออก | g:เริ่มคาลิเบรต(ทีละจุด) | SPACE:เก็บ | ENTER:ถัดไป | BACKSPACE:ย้อน | a/d,yaw  w/s,pitch  z/x,roll | f:กระจก | m:จุด | p:UIเบา"
        out = draw_text_thai(out, help_text, (16, h-32), 20, (200,200,200), 1, (0,0,0))
        cv2.imshow("Face Landmarker (Pi) - q to quit", out)

        # ---------- Keys ----------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('f'): CFG["mirror"] = not CFG["mirror"]; save_cfg()
        elif key == ord('m'): CFG["show_points"] = not CFG["show_points"]; save_cfg()
        elif key == ord('p'): CFG["perf_overlay"] = not CFG["perf_overlay"]; save_cfg()
        elif key == ord('a'): CFG["yaw_bias"]   -= 1.0; save_cfg()
        elif key == ord('d'): CFG["yaw_bias"]   += 1.0; save_cfg()
        elif key == ord('w'): CFG["pitch_bias"] -= 1.0; save_cfg()
        elif key == ord('s'): CFG["pitch_bias"] += 1.0; save_cfg()
        elif key == ord('z'): CFG["camera_roll_deg"] -= 1.0; save_cfg()
        elif key == ord('x'): CFG["camera_roll_deg"] += 1.0; save_cfg()
        elif key == ord('r'):
            yaw_deg_ema = pitch_deg_ema = roll_deg_ema = None
            eye_w_ema = None; baseline_eye_w = None
            yaw_state, pitch_state = "center", "neutral"
            cmd_hist.clear()
            current_left_speed = current_right_speed = 0.0
            target_left_speed  = target_right_speed  = 0.0

        # ---- Manual calibration keys ----
        elif key == ord('g'):                 # เริ่มใหม่
            if eye_w_ema is not None: baseline_eye_w = float(eye_w_ema)
            calib.start()
        elif key in (32, ord(' ')):           # SPACE: เก็บตัวอย่าง
            if calib.is_active():
                ok_cap = calib.add_sample(last_meas["yaw_c"], last_meas["pitch_c"],
                                          last_meas["roll_corr"], last_meas["eye_w"], baseline_eye_w)
                # เก็บตัวอย่างแรกของ center → ตั้ง baseline ระยะ
                if calib.step_name()=="center" and ok_cap and baseline_eye_w is None and last_meas["eye_w"] is not None:
                    baseline_eye_w = float(last_meas["eye_w"])
        elif key in (13, 10, ord('n')):       # ENTER: ไปขั้นถัดไป (รองรับทั้ง 13/10)
            if calib.is_active(): calib.next_step()
        elif key in (8, ord('b')):            # BACKSPACE: ย้อน
            if calib.is_active(): calib.prev_step()
        elif key == 27:                        # ESC: ยกเลิกคาลิเบรต
            if calib.is_active(): calib.reset()

        # ---------- JSON ----------
        status = {
            "command": command_key, "turn": turn_key, "pitch": pitch_key, "depth": depth_key,
            "yaw_deg":   round(float(yaw_c),1) if yaw_c is not None else None,
            "pitch_deg": round(float(pitch_c),1) if pitch_c is not None else None,
            "eye_ratio": round(float(eye_w_ema)/float(baseline_eye_w),3) if (baseline_eye_w and eye_w_ema) else None,
            "calibrated": baseline_eye_w is not None,
            "left_speed": int(current_left_speed), "right_speed": int(current_right_speed),
            "target_left_speed": int(target_left_speed), "target_right_speed": int(target_right_speed),
            "cam_roll_bias_deg": round(float(CFG["camera_roll_deg"]),1),
            "fps": round(float(fps),1),
            "thresholds": {
                "yaw_enter": CFG["YAW_ENTER_DEG"], "yaw_exit": CFG["YAW_EXIT_DEG"],
                "pitch_up_enter": CFG["PITCH_UP_ENTER_DEG"], "pitch_up_exit": CFG["PITCH_UP_EXIT_DEG"],
                "pitch_down_enter": CFG["PITCH_DOWN_ENTER_DEG"], "pitch_down_exit": CFG["PITCH_DOWN_EXIT_DEG"],
            }
        }
        # (ปิด/เปิดเองถ้าต้องการ)
        # print(json.dumps(status, ensure_ascii=False), flush=True)

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"ไม่พบโมเดล: {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)
    main()
