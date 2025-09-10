# pose_direction_face.py
import cv2, numpy as np, mediapipe as mp, math, os, json
from collections import deque
from PIL import ImageFont, ImageDraw, Image

FONT_PATH = os.path.join(os.path.dirname(__file__), "THSarabunNew.ttf")
mp_drawing = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles
mp_face     = mp.solutions.face_mesh

# ----- เกณฑ์ hysteresis (หน่วย "องศา") -----
YAW_ENTER_DEG = 20;  YAW_EXIT_DEG = 12
PITCH_UP_ENTER_DEG = 12;  PITCH_UP_EXIT_DEG = 7
PITCH_DOWN_ENTER_DEG = -12; PITCH_DOWN_EXIT_DEG = -7

EMA_A = 0.25
DEPTH_DELTA = 0.15  # ใช้ความกว้างตาซ้าย-ขวาเทียบ baseline

def draw_text_thai(img_bgr, text, xy, font_size=28, color=(255,255,255), stroke=2, stroke_color=(0,0,0)):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb); draw = ImageDraw.Draw(pil_img)
    try: font = ImageFont.truetype(FONT_PATH, font_size)
    except: font = ImageFont.load_default()
    draw.text(xy, text, font=font, fill=tuple(color), stroke_width=int(stroke), stroke_fill=tuple(stroke_color))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def ema(prev, new, a=EMA_A): return new if prev is None else prev*(1-a)+new*a

def rot_to_euler_deg(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.degrees(math.atan2(R[2,1], R[2,2]))  # pitch (+ เงย)
        y = math.degrees(math.atan2(-R[2,0], sy))     # yaw   (+ ขวา)
        z = math.degrees(math.atan2(R[1,0], R[0,0]))  # roll
    else:
        x = math.degrees(math.atan2(-R[1,2], R[1,1])); y = math.degrees(math.atan2(-R[2,0], sy)); z = 0
    return x, y, z

# จุด 3D แบบคร่าว (สัดส่วนใบหน้าโดยประมาณ)
MODEL_3D = np.array([
    (0.0, 0.0, 0.0),        # nose tip
    (0.0, -63.6, -12.5),    # chin
    (-43.3, 32.7, -26.0),   # left eye outer (ของผู้ใช้ = ตาซ้าย)
    (43.3, 32.7, -26.0),    # right eye outer
    (-28.9, -28.9, -24.1),  # left mouth corner
    (28.9, -28.9, -24.1)    # right mouth corner
], dtype=np.float64)

# ดัชนี FaceMesh ที่จะใช้ (ค่อนข้างเป็นมาตรฐาน)
IDX_NOSE=4; IDX_CHIN=152; IDX_LEFT_EYE_OUT=263; IDX_RIGHT_EYE_OUT=33; IDX_LEFT_MOUTH=291; IDX_RIGHT_MOUTH=61

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened(): raise SystemExit("Cannot open camera")

mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

yaw_deg_ema = pitch_deg_ema = roll_deg_ema = None
eye_w_ema = None
baseline_eye_w = None
yaw_state, pitch_state = "center", "neutral"
cmd_hist = deque(maxlen=5)
yaw_bias = 0.0; pitch_bias = 0.0
mirror = True
stream_json = False; save_json_on_change=False; last_export=None

# ตัวแปรสำหรับความเร็วแบบ smooth
current_left_speed = 0.0
current_right_speed = 0.0
target_left_speed = 0.0
target_right_speed = 0.0
SPEED_STEP = 10  # ปรับความเร็วทีละ 10

while True:
    ok, frame = cap.read()
    if not ok: break
    if mirror: frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    out = frame.copy()

    command_key="idle"; turn_key="center"; pitch_key="neutral"; depth_key=None
    command_th="คำสั่ง: -"; turn_th="หัน: -"; pitch_th="ก้ม/เงย: -"; depth_th="ระยะ: -"
    color=(0,255,255)

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        pts2d = np.array([
            (lm[IDX_NOSE].x*w,        lm[IDX_NOSE].y*h),
            (lm[IDX_CHIN].x*w,        lm[IDX_CHIN].y*h),
            (lm[IDX_LEFT_EYE_OUT].x*w, lm[IDX_LEFT_EYE_OUT].y*h),
            (lm[IDX_RIGHT_EYE_OUT].x*w,lm[IDX_RIGHT_EYE_OUT].y*h),
            (lm[IDX_LEFT_MOUTH].x*w,  lm[IDX_LEFT_MOUTH].y*h),
            (lm[IDX_RIGHT_MOUTH].x*w, lm[IDX_RIGHT_MOUTH].y*h),
        ], dtype=np.float64)

        # กล้องสมมุติ
        f = w*1.2
        camM = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float64)
        dist = np.zeros((4,1))

        okp, rvec, tvec = cv2.solvePnP(MODEL_3D, pts2d, camM, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if okp:
            R,_ = cv2.Rodrigues(rvec)
            pitch_deg, yaw_deg, roll_deg = rot_to_euler_deg(R)  # ตามนิยามด้านบน
            # smooth
            yaw_deg_ema   = ema(yaw_deg_ema,   yaw_deg)
            pitch_deg_ema = ema(pitch_deg_ema, pitch_deg)
            roll_deg_ema  = ema(roll_deg_ema,  roll_deg)

            # bias ส่วนตัว (กด c ตั้งศูนย์)
            yaw_c   = yaw_deg_ema   - yaw_bias
            pitch_c = pitch_deg_ema - pitch_bias

            # hysteresis yaw (สลับทิศทาง: ขวาไปซ้าย, ซ้ายไปขวา)
            if yaw_state=="center":
                if yaw_c <= -YAW_ENTER_DEG: yaw_state="right"  # หันซ้าย = ไปขวา
                elif yaw_c >=  YAW_ENTER_DEG: yaw_state="left"  # หันขวา = ไปซ้าย
            elif yaw_state=="left":
                if yaw_c <  YAW_EXIT_DEG: yaw_state="center"
            elif yaw_state=="right":
                if yaw_c > -YAW_EXIT_DEG: yaw_state="center"

            # hysteresis pitch (ก้อมหน้า=เดินหน้า, เงยหน้า=หยุด)
            if pitch_state=="neutral":
                if pitch_c >=  PITCH_UP_ENTER_DEG:   pitch_state="up"    # เงยหน้า = หยุด
                elif pitch_c <= PITCH_DOWN_ENTER_DEG: pitch_state="down"  # ก้อมหน้า = เดินหน้า
            elif pitch_state=="up":
                if pitch_c <   PITCH_UP_EXIT_DEG:    pitch_state="neutral"
            elif pitch_state=="down":
                if pitch_c >   PITCH_DOWN_EXIT_DEG:  pitch_state="neutral"

            # depth จากความกว้างตาซ้าย-ขวา
            eye_w = np.hypot(pts2d[2,0]-pts2d[3,0], pts2d[2,1]-pts2d[3,1])
            eye_w_ema = ema(eye_w_ema, eye_w)
            if baseline_eye_w is not None and eye_w_ema is not None:
                ratio = eye_w_ema / baseline_eye_w
                if ratio >= 1.0+DEPTH_DELTA:  depth_th, depth_key = "ระยะ: ใกล้ขึ้น", "near"
                elif ratio <= 1.0-DEPTH_DELTA:depth_th, depth_key = "ระยะ: ไกลออก", "far"
                else:                          depth_th, depth_key = "ระยะ: คงที่",  "neutral"

            # สรุปสถานะ
            turn_key = "left" if yaw_state=="left" else "right" if yaw_state=="right" else "center"
            turn_th  = "หัน: ซ้าย" if turn_key=="left" else "หัน: ขวา" if turn_key=="right" else "หัน: ตรง"
            if   pitch_state=="up":   pitch_key, pitch_th = "up",   "ก้ม/เงย: เงยหน้า"
            elif pitch_state=="down": pitch_key, pitch_th = "down", "ก้ม/เงย: ก้มหน้า"
            else:                     pitch_key, pitch_th = "neutral","ก้ม/เงย: ปกติ"

            # policy: pitch มาก่อน, ไม่งั้นใช้ turn (สลับ: ก้มหน้า=เดินหน้า, เงยหน้า=หยุด)
            if   pitch_key=="up":   command_key, command_th, color = "stop",   "คำสั่ง: หยุด",(0,200,255)
            elif pitch_key=="down": command_key, command_th, color = "forward","คำสั่ง: เดินหน้า",(0,220,0)
            else:
                if   turn_key=="left":  command_key, command_th, color = "left","คำสั่ง: ไปซ้าย",(0,200,255)
                elif turn_key=="right": command_key, command_th, color = "right","คำสั่ง: ไปขวา",(0,200,255)
                else:                    command_key, command_th, color = "idle","คำสั่ง: คงที่",(0,255,255)

            # กำหนดความเร็วเป้าหมายตามคำสั่ง
            if   command_key == "forward": target_left_speed, target_right_speed = 100, 100  # เดินหน้า
            elif command_key == "left":    target_left_speed, target_right_speed = 0, 100    # ไปซ้าย
            elif command_key == "right":   target_left_speed, target_right_speed = 100, 0    # ไปขวา
            elif command_key == "stop":    target_left_speed, target_right_speed = 0, 0      # หยุด
            else:                          target_left_speed, target_right_speed = 0, 0      # คงที่

            # ปรับความเร็วแบบ smooth ทีละ 10
            if current_left_speed < target_left_speed:
                current_left_speed = min(current_left_speed + SPEED_STEP, target_left_speed)
            elif current_left_speed > target_left_speed:
                current_left_speed = max(current_left_speed - SPEED_STEP, target_left_speed)
                
            if current_right_speed < target_right_speed:
                current_right_speed = min(current_right_speed + SPEED_STEP, target_right_speed)
            elif current_right_speed > target_right_speed:
                current_right_speed = max(current_right_speed - SPEED_STEP, target_right_speed)

            cmd_hist.append(command_key)
            command_key = max(set(cmd_hist), key=cmd_hist.count)

            # วาด landmark ใบหน้า (เพื่อ debug)
            mp_drawing.draw_landmarks(out, res.multi_face_landmarks[0], mp_face.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None, connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())

            # แปะตัวเลขเล็กๆ
            out = draw_text_thai(out, f"yaw:{yaw_c:.1f}°  pitch:{pitch_c:.1f}°  roll:{roll_deg_ema:.1f}°", (10,24), 22, (0,255,0),2)

    # UI
    out = draw_text_thai(out, command_th, (16, 100), 36, color, 2, (0,0,0))
    out = draw_text_thai(out, turn_th,  (16,160), 26, (255,255,0), 2, (0,0,0))
    out = draw_text_thai(out, pitch_th, (16,188), 26, (255,255,0), 2, (0,0,0))
    out = draw_text_thai(out, depth_th, (16,216), 26, (255,255,0), 2, (0,0,0))
    
    # แสดงความเร็วแบบ smooth
    speed_text = f"ความเร็ว: ซ้าย {int(current_left_speed):3d} ขวา {int(current_right_speed):3d}"
    out = draw_text_thai(out, speed_text, (16, 244), 26, (0,255,255), 2, (0,0,0))

    help_text = "q:ออก | c:คาลิเบรต | r:รีเซ็ต | f:กระจก | j:JSON | k:Save JSON"
    out = draw_text_thai(out, help_text, (16, out.shape[0]-32), 22, (200,200,200), 1, (0,0,0))
    cv2.imshow("Head Pose (FaceMesh) - q to quit", out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'):
        if eye_w_ema is not None: baseline_eye_w = float(eye_w_ema)
        if yaw_deg_ema   is not None: yaw_bias   = float(yaw_deg_ema)
        if pitch_deg_ema is not None: pitch_bias = float(pitch_deg_ema)
    elif key == ord('r'):
        baseline_eye_w=None; eye_w_ema=None; yaw_deg_ema=pitch_deg_ema=roll_deg_ema=None
        yaw_state="center"; pitch_state="neutral"; yaw_bias=0.0; pitch_bias=0.0; cmd_hist.clear()
        current_left_speed=0.0; current_right_speed=0.0; target_left_speed=0.0; target_right_speed=0.0
    elif key == ord('f'): mirror = not mirror
    elif key == ord('j'): stream_json = not stream_json
    elif key == ord('k'): save_json_on_change = not save_json_on_change

    status = {
        "command": command_key, "turn": turn_key, "pitch": pitch_key, "depth": depth_key,
        "yaw_deg": round(float(yaw_deg_ema - yaw_bias),1) if yaw_deg_ema is not None else None,
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

cap.release(); cv2.destroyAllWindows()
