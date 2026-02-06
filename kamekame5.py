import cv2
import mediapipe as mp
import math

video_path = 'blue_ball.mp4'  
min_size = 50               
scale_multiplier = 1.2       
threshold_value = 40         

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,        
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

effect_cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    ret_eff, effect_frame = effect_cap.read()
    if not ret_eff:
        effect_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_eff, effect_frame = effect_cap.read()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_centers = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            landmark = hand_landmarks.landmark[9]
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            hand_centers.append((cx, cy))
            
    if len(hand_centers) == 2:
        (x1, y1) = hand_centers[0]
        (x2, y2) = hand_centers[1]

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        current_size = int(distance * scale_multiplier)
        
        if current_size < min_size: current_size = min_size
        if current_size > w: current_size = w 

        eff_w = current_size
        eff_h = current_size 
        
        effect_resized = cv2.resize(effect_frame, (eff_w, eff_h))

        pos_x = center_x - (eff_w // 2)
        pos_y = center_y - (eff_h // 2)

        y1_roi, y2_roi = pos_y, pos_y + eff_h
        x1_roi, x2_roi = pos_x, pos_x + eff_w

        if y1_roi < 0: y1_roi = 0
        if y2_roi > h: y2_roi = h
        if x1_roi < 0: x1_roi = 0
        if x2_roi > w: x2_roi = w

        overlay_h = y2_roi - y1_roi
        overlay_w = x2_roi - x1_roi

        if overlay_h > 0 and overlay_w > 0:
            img_y_start = 0 if pos_y >= 0 else -pos_y
            img_x_start = 0 if pos_x >= 0 else -pos_x
            
            effect_cropped = effect_resized[img_y_start : img_y_start + overlay_h, 
                                            img_x_start : img_x_start + overlay_w]
            
            roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]

            img2gray = cv2.cvtColor(effect_cropped, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, threshold_value, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            img2_fg = cv2.bitwise_and(effect_cropped, effect_cropped, mask=mask)

            dst = cv2.add(img1_bg, img2_fg)
            frame[y1_roi:y2_roi, x1_roi:x2_roi] = dst

    cv2.imshow('Hand Energy Ball', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
effect_cap.release()
cv2.destroyAllWindows()