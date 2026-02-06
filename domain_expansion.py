import cv2
import mediapipe as mp
import numpy as np
import math

fire_path = 'fireball.mp4'  
bg_video_path = 'bg.mp4'  
threshold_value = 40     
blur_intensity = (21, 21) 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
effect_cap = cv2.VideoCapture(fire_path)
bg_cap = cv2.VideoCapture(bg_video_path)

def is_victory_sign(lm):
    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_down = lm[16].y > lm[14].y
    pinky_down = lm[20].y > lm[18].y
    return index_up and middle_up and ring_down and pinky_down

print("ชู 2 นิ้ว เพื่อเปลี่ยนมิติและปล่อยพลัง! (กด 'q' เพื่อออก)")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    ret_bg, bg_frame = bg_cap.read()
    if not ret_bg:
        bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_bg, bg_frame = bg_cap.read()
    bg_frame = cv2.resize(bg_frame, (w, h))

    ret_eff, effect_frame = effect_cap.read()
    if not ret_eff:
        effect_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_eff, effect_frame = effect_cap.read()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    is_power_active = False 
    final_image = frame.copy()
    fire_positions = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            mp_drawing.draw_landmarks(
                 final_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                 mp_drawing_styles.get_default_hand_landmarks_style(),
                 mp_drawing_styles.get_default_hand_connections_style()
            )

            if is_victory_sign(lm):
                is_power_active = True
                idx_tip = lm[8]
                cx, cy = int(idx_tip.x * w), int(idx_tip.y * h)
                wrist = lm[0]
                middle_mcp = lm[9]
                hand_size_px = math.sqrt(((middle_mcp.x - wrist.x)*w)**2 + ((middle_mcp.y - wrist.y)*h)**2)
                size = int(hand_size_px * 0.8)
                fire_positions.append((cx, cy, size))

    if is_power_active:
        results_seg = segmentation.process(rgb_frame)
        mask = results_seg.segmentation_mask
        mask_blurred = cv2.GaussianBlur(mask, blur_intensity, 0)
        mask_3d = np.stack((mask_blurred,) * 3, axis=-1)
        foreground_part = frame * mask_3d
        background_part = bg_frame * (1.0 - mask_3d)
        final_image = cv2.add(foreground_part, background_part).astype(np.uint8)
        
        for (cx, cy, size) in fire_positions:
            eff_w, eff_h = size, size
            effect_resized = cv2.resize(effect_frame, (eff_w, eff_h))
            circular_mask = np.zeros((eff_h, eff_w), dtype=np.uint8)
            cv2.circle(circular_mask, (eff_w // 2, eff_h // 2), size // 2, 255, -1)
            effect_resized = cv2.bitwise_and(effect_resized, effect_resized, mask=circular_mask)
            pos_x = cx - (eff_w // 2)
            pos_y = cy - (eff_h // 2)
            y1, y2 = pos_y, pos_y + eff_h
            x1, x2 = pos_x, pos_x + eff_w
            if y1 < 0: y1 = 0
            if y2 > h: y2 = h
            if x1 < 0: x1 = 0
            if x2 > w: x2 = w
            overlay_h = y2 - y1
            overlay_w = x2 - x1
            if overlay_h > 0 and overlay_w > 0:
                img_y = 0 if pos_y >= 0 else -pos_y
                img_x = 0 if pos_x >= 0 else -pos_x
                eff_crop = effect_resized[img_y:img_y+overlay_h, img_x:img_x+overlay_w]
                roi = final_image[y1:y2, x1:x2]
                img2gray = cv2.cvtColor(eff_crop, cv2.COLOR_BGR2GRAY)
                ret, mask_fire = cv2.threshold(img2gray, threshold_value, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask_fire)
                bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                fg = cv2.bitwise_and(eff_crop, eff_crop, mask=mask_fire)
                final_image[y1:y2, x1:x2] = cv2.add(bg, fg)

    cv2.imshow('Super Power Mode (Smooth Edge)', final_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
effect_cap.release()
bg_cap.release()
cv2.destroyAllWindows()