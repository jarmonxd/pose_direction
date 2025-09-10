# pose_direction_face.py (Raspberry Pi friendly)
import os, sys, json, math, platform, time
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# -------------------- ตั้งค่า --------------------
FONT_PATH = os.path.join(os.path.dirname(__file__), "THSarabunNew.ttf")

# เกณฑ์ hysteresis (หน่วย "องศา")
YAW_ENTER_DEG = 20;   YAW_EXIT_DEG = 12
PITCH_UP_ENTER_DEG = 12;   PITCH_UP_EXIT_DEG = 7
PITCH_DOWN_ENTER_DEG = -12; PITCH_DOWN_EXIT_DEG = -7

EMA_A = 0.25                 # ค่าถ่วงเฉลี่ยเอ็กซ์โปเนนเชียล (0..1)
DEPTH_DELTA = 0.15           # สัดส่วนเปลี่ยนระยะ (จากความกว้างตาเทียบ baseline)
FRAME_W, FRAME_H = 640, 480  # บน Pi ใช้ 640x480 จะลื่นกว่า
DRAW_LANDMARKS = True        # debug วาด mesh

# ความเร็วล้อ (จำลอง/สำหรับส่งออก)
SPEED_STEP = 10  # เพิ่ม/ลดทีละ 10 หน่วย
MAX_SPEED  = 100

# -------------------- Utils --------------------
def draw_text_thai(img_bgr, text, xy, font_size=28, color=(255,255,255), stroke=2, stroke_color=(0,0,0)):
    """วาดข้อความไทย (BGR) ด้วย PIL"""
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

def ema(prev, new, a=EMA_A):
    return new if prev is None else prev*(1-a) + new*a

def rot_to_euler_deg(R):
    # คอนเวิร์ต rotation matrix → Euler ตามแกนที่กำหนด
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.degrees(math.atan2(R[2,1], R[2,2]))   # + เงย
        yaw   = math.degrees(math.atan2(-R[2,0], sy))      # + หันขวา
        roll  = math.degrees(math.atan2(R[1,0], R[0,0]))
    else:
        pitch = math.degrees(math.atan2(-R[1,2], R[1,1])); yaw = math.degrees(math.atan2(-R[2,0], sy)); roll = 0
    return pitch, yaw, roll

# จุด 3D คร่าว ๆ ของโครงหน้า (มม.สมมุติ)
MODEL_3D = np.array([
    (0.0,   0.0,   0.0),    # nose tip
    (0.0,  -63.6, -12.5),   # chin
    (-43.3, 32.7, -26.0),   # left eye outer
    (43.3,  32.7, -26.0),   # right eye outer
    (-28.9,-28.9, -24.1),   # left mouth corner
    (28.9, -28.9, -24.1),   # right mouth corner
], dtype=np.float64)

# ดัชนี FaceMesh
IDX_NOSE=4; IDX_CHIN=152; IDX_LEFT_EYE_OUT=263; IDX_RIGHT_EYE_OUT=33; IDX_LEFT_MOUTH=291; IDX_RIGHT_MOUTH=61

def open_camera():
    """เปิดกล้องให้เหมาะกับ OS (Windows = DSHOW, Linux = V4L2) และตั้งความละเอียด"""
    is_windows = (os.name == "nt") or ("Windows" in platform.system())
    backend = cv2.CAP_DSHOW if is_windows else cv2.CAP_V4L2

    # ลองไล่ device index 0..3 เผื่อกล้องไม่ใช่ /dev/video0
    for idx in [0,1,2,3]:
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            # fallback แบบไม่ระบุ backend
            cap.release()
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            # ตั้งความละเอียด
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            # ลองเซ็ต FOURCC เป็น MJPG จะช้าลง CPU น้อยลงบน Pi (ถ้ากล้องรองรับ)
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except:
                pass
            return cap
        cap.release()
    raise SystemExit("Cannot open camera")

def has_display():
    """มีจอให้เปิดหน้าต่างได้ไหม (กันกรณี ssh/headless)"""
    if os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY"))

# -------------------- Main --------------------
def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_styles   = mp.solutions.drawing_styles
    mp_face     = mp.solutions.face_mesh

    cap = open_camera()
    display_ok = has_display()

    yaw_deg_ema = pitch_deg_ema = roll_deg_ema = None
    eye_w_ema = None
    baseline_eye_w = None
    yaw_state, pitch_state = "center", "neutral"
    cmd_hist = deque(maxlen=5)
    yaw_bias = 0.0; pitch_bias = 0.0
    mirror = True
    stream_json = False
    save_json_on_change = False
    last_export = None

    current_left_speed = 0.0
    current_right_speed = 0.0
    target_left_speed = 0.0
    target_right_speed = 0.0

    # เตรียมกล้องเสมือน (intrinsic) จากขนาดเฟรม
    ret, _frame0 = cap.read()
    if not ret:
        cap.release()
        raise SystemExit("Cannot read first frame from camera")
    h0, w0 = _frame0.shape[:2]
    f = w0 * 1.2
    camM = np.array([[f,0,w0/2],[0,f,h0/2],[0,0,1]], dtype=np.float64)
    dist = np.zeros((4,1))

    with mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as mesh:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # ถ้าขนาดเฟรมเปลี่ยน (บางกล้องจะสลับเอง) อัปเดต intrinsic
            if (w != w0) or (h != h0):
                w0, h0 = w, h
                f = w0 * 1.2
                camM = np.array([[f,0,w0/2],[0,f,h0/2],[0,0,1]], dtype=np.float64)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)
            out = frame.copy()

            command_key="idle"; turn_key="center"; pitch_key="neutral"; depth_key=None
            command_th="คำสั่ง: -"; turn_th="หัน: -"; pitch_th="ก้ม/เงย: -"; depth_th="ระยะ: -"
            color=(0,255,255)

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                pts2d = np.array([
                    (lm[IDX_NOSE].x*w,          lm[IDX_NOSE].y*h),
                    (lm[IDX_CHIN].x*w,          lm[IDX_CHIN].y*h),
                    (lm[IDX_LEFT_EYE_OUT].x*w,  lm[IDX_LEFT_EYE_OUT].y*h),
                    (lm[IDX_RIGHT_EYE_OUT].x*w, lm[IDX_RIGHT_EYE_OUT].y*h),
                    (lm[IDX_LEFT_MOUTH].x*w,    lm[IDX_LEFT_MOUTH].y*h),
                    (lm[IDX_RIGHT_MOUTH].x*w,   lm[IDX_RIGHT_MOUTH].y*h),
                ], dtype=np.float64)

                # PnP หาท่าศีรษะ
                okp, rvec, tvec = cv2.solvePnP(
                    MODEL_3D, pts2d, camM, dist,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if okp:
                    R,_ = cv2.Rodrigues(rvec)
                    pitch_deg, yaw_deg, roll_deg = rot_to_euler_deg(R)

                    # smooth
                    yaw_deg_ema   = ema(yaw_deg_ema,   yaw_deg)
                    pitch_deg_ema = ema(pitch_deg_ema, pitch_deg)
                    roll_deg_ema  = ema(roll_deg_ema,  roll_deg)

                    # หัก bias ส่วนตัว (กด c เพื่อตั้งศูนย์)
                    yaw_c   = yaw_deg_ema   - yaw_bias if yaw_deg_ema   is not None else None
                    pitch_c = pitch_deg_ema - pitch_bias if pitch_deg_ema is not None else None

                    # hysteresis: yaw
                    if yaw_c is not None:
                        if yaw_state == "center":
                            if yaw_c <= -YAW_ENTER_DEG:
                                yaw_state = "right"  # หันซ้าย → ไปขวา
                            elif yaw_c >=  YAW_ENTER_DEG:
                                yaw_state = "left"   # หันขวา → ไปซ้าย
                        elif yaw_state == "left":
                            if yaw_c < YAW_EXIT_DEG:
                                yaw_state = "center"
                        elif yaw_state == "right":
                            if yaw_c > -YAW_EXIT_DEG:
                                yaw_state = "center"

                    # hysteresis: pitch
                    if pitch_c is not None:
                        if pitch_state == "neutral":
                            if   pitch_c >=  PITCH_UP_ENTER_DEG:    pitch_state = "up"    # เงยหน้า
                            elif pitch_c <= PITCH_DOWN_ENTER_DEG:    pitch_state = "down"  # ก้มหน้า
                        elif pitch_state == "up":
                            if pitch_c < PITCH_UP_EXIT_DEG:          pitch_state = "neutral"
                        elif pitch_state == "down":
                            if pitch_c > PITCH_DOWN_EXIT_DEG:        pitch_state = "neutral"

                    # depth จากความกว้างตาซ้าย-ขวา
                    eye_w = np.hypot(pts2d[2,0]-pts2d[3,0], pts2d[2,1]-pts2d[3,1])
                    eye_w_ema = ema(eye_w_ema, eye_w)
                    if baseline_eye_w is not None and eye_w_ema is not None:
                        ratio = eye_w_ema / baseline_eye_w
                        if   ratio >= 1.0 + DEPTH_DELTA: depth_th, depth_key = "ระยะ: ใกล้ขึ้น", "near"
                        elif ratio <= 1.0 - DEPTH_DELTA: depth_th, depth_key = "ระยะ: ไกลออก", "far"
                        else:                            depth_th, depth_key = "ระยะ: คงที่",  "neutral"

                    # สรุปสถานะ yaw/pitch
                    turn_key = "left" if yaw_state=="left" else "right" if yaw_state=="right" else "center"
                    turn_th  = "หัน: ซ้าย" if turn_key=="left" else "หัน: ขวา" if turn_key=="right" else "หัน: ตรง"
                    if   pitch_state=="up":   pitch_key, pitch_th = "up",   "ก้ม/เงย: เงยหน้า"
                    elif pitch_state=="down": pitch_key, pitch_th = "down", "ก้ม/เงย: ก้มหน้า"
                    else:                     pitch_key, pitch_th = "neutral", "ก้ม/เงย: ปกติ"

                    # policy: pitch มาก่อน แล้วค่อย turn
                    if   pitch_key=="up":    command_key, command_th, color = "stop",    "คำสั่ง: หยุด",     (0,200,255)
                    elif pitch_key=="down":  command_key, command_th, color = "forward", "คำสั่ง: เดินหน้า", (0,220,0)
                    else:
                        if   turn_key=="left":   command_key, command_th, color = "left",  "คำสั่ง: ไปซ้าย", (0,200,255)
                        elif turn_key=="right":  command_key, command_th, color = "right", "คำสั่ง: ไปขวา",  (0,200,255)
                        else:                    command_key, command_th, color = "idle",  "คำสั่ง: คงที่",  (0,255,255)

                    # ความเร็วเป้าหมายตามคำสั่ง
                    if   command_key == "forward": target_left_speed, target_right_speed = MAX_SPEED, MAX_SPEED
                    elif command_key == "left":    target_left_speed, target_right_speed = 0,         MAX_SPEED
                    elif command_key == "right":   target_left_speed, target_right_speed = MAX_SPEED, 0
                    elif command_key == "stop":    target_left_speed, target_right_speed = 0,         0
                    else:                           target_left_speed, target_right_speed = 0,         0

                    # speed ramp (smooth)
                    if current_left_speed < target_left_speed:
                        current_left_speed = min(current_left_speed + SPEED_STEP, target_left_speed)
                    elif current_left_speed > target_left_speed:
                        current_left_speed = max(current_left_speed - SPEED_STEP, target_left_speed)
                    if current_right_speed < target_right_speed:
                        current_right_speed = min(current_right_speed + SPEED_STEP, target_right_speed)
                    elif current_right_speed > target_right_speed:
                        current_right_speed = max(current_right_speed - SPEED_STEP, target_right_speed)

                    # filter คำสั่งด้วย majority window
                    cmd_hist.append(command_key)
                    command_key = max(set(cmd_hist), key=cmd_hist.count)

                    # วาด landmark (เลือกเปิด/ปิดได้)
                    if DRAW_LANDMARKS:
                        mp_drawing.draw_landmarks(
                            out, res.multi_face_landmarks[0], mp_face.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                        )

                    # แปะตัวเลข
                    yaw_show   = yaw_c if (yaw_deg_ema is not None) else None
                    pitch_show = pitch_c if (pitch_deg_ema is not None) else None
                    roll_show  = roll_deg_ema if (roll_deg_ema is not None) else None
                    out = draw_text_thai(
                        out,
                        f"yaw:{yaw_show:.1f}°  pitch:{pitch_show:.1f}°  roll:{roll_show:.1f}°",
                        (10,24), 22, (0,255,0), 2
                    )

            # UI
            out = draw_text_thai(out, command_th, (16, 100), 36, color, 2, (0,0,0))
            out = draw_text_thai(out, turn_th,  (16,160), 26, (255,255,0), 2, (0,0,0))
            out = draw_text_thai(out, pitch_th, (16,188), 26, (255,255,0), 2, (0,0,0))
            out = draw_text_thai(out, depth_th, (16,216), 26, (255,255,0), 2, (0,0,0))

            speed_text = f"ความเร็ว: ซ้าย {int(current_left_speed):3d} ขวา {int(current_right_speed):3d}"
            out = draw_text_thai(out, speed_text, (16,244), 26, (0,255,255), 2, (0,0,0))

            help_text = "q:ออก | c:คาลิเบรต | r:รีเซ็ต | f:กระจก | j:JSON | k:Save JSON"
            out = draw_text_thai(out, help_text, (16, out.shape[0]-32), 22, (200,200,200), 1, (0,0,0))

            # แสดงผล (ถ้ามีจอ)
            if display_ok:
                cv2.imshow("Head Pose (FaceMesh) - q to quit", out)
                key = cv2.waitKey(1) & 0xFF
            else:
                # headless: ยังอ่านคีย์ไม่ได้ ให้หน่วงนิดนึง
                key = -1
                time.sleep(0.01)

            # คีย์ลัด
            if key == ord('q'): break
            elif key == ord('c'):
                if eye_w_ema is not None: baseline_eye_w = float(eye_w_ema)
                if yaw_deg_ema is not None: yaw_bias = float(yaw_deg_ema)
                if pitch_deg_ema is not None: pitch_bias = float(pitch_deg_ema)
            elif key == ord('r'):
                baseline_eye_w=None; eye_w_ema=None
                yaw_deg_ema=pitch_deg_ema=roll_deg_ema=None
                yaw_state="center"; pitch_state="neutral"
                yaw_bias=0.0; pitch_bias=0.0; cmd_hist.clear()
                current_left_speed=current_right_speed=0.0
                target_left_speed=target_right_speed=0.0
            elif key == ord('f'):
                mirror = not mirror
            elif key == ord('j'):
                stream_json = not stream_json
            elif key == ord('k'):
                save_json_on_change = not save_json_on_change

            # ส่งออกสถานะ
            status = {
                "command": command_key,
                "turn": turn_key,
                "pitch": pitch_key,
                "depth": depth_key,
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
                try:
                    print(json.dumps(status, ensure_ascii=False), flush=True)
                except:
                    pass
            if save_json_on_change:
                # เซฟเมื่อมีการเปลี่ยนแปลง
                if status != last_export:
                    try:
                        with open("pose_status.json", "w", encoding="utf-8") as f:
                            json.dump(status, f, ensure_ascii=False)
                        last_export = status
                    except:
                        pass

    cap.release()
    if display_ok:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
