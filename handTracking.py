import cv2
import mediapipe as mp
import math  # For distance calculation

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Variables to store the coordinates of id 8 for both hands
    left_hand_8 = None
    right_hand_8 = None

    left_hand_landmarks = {}
    right_hand_landmarks = {}

    if results.multi_hand_landmarks and results.multi_handedness:
        

        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Determine if the hand is left or right
            hand_label = results.multi_handedness[hand_index].classification[0].label

            for id, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)

                if hand_label == "Left":
                    left_hand_landmarks[id] = (cx, cy)
                elif hand_label == "Right":
                    right_hand_landmarks[id] = (cx, cy)

            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Example: Store id 8 coordinates for both hands
        left_hand_8 = left_hand_landmarks.get(8)
        right_hand_8 = right_hand_landmarks.get(8)

    if left_hand_landmarks.get(4) and left_hand_landmarks.get(8):
        cv2.line(frame, left_hand_landmarks[4], left_hand_landmarks[8], (255, 0, 0), 2)
        distance = math.sqrt((left_hand_landmarks[4][0] - left_hand_landmarks[8][0]) ** 2 + (left_hand_landmarks[4][1] - left_hand_landmarks[8][1]) ** 2)
        mid_point = ((left_hand_landmarks[4][0] + left_hand_landmarks[8][0]) // 2, (left_hand_landmarks[4][1] + left_hand_landmarks[8][1]) // 2)
        cv2.putText(frame, f'{distance:.2f}', mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if right_hand_landmarks.get(4) and right_hand_landmarks.get(8):
        cv2.line(frame, right_hand_landmarks[4], right_hand_landmarks[8], (255, 0, 0), 2)
        distance = math.sqrt((right_hand_landmarks[4][0] - right_hand_landmarks[8][0]) ** 2 + (right_hand_landmarks[4][1] - right_hand_landmarks[8][1]) ** 2)
        mid_point = ((right_hand_landmarks[4][0] + right_hand_landmarks[8][0]) // 2, (right_hand_landmarks[4][1] + right_hand_landmarks[8][1]) // 2)
        cv2.putText(frame, f'{distance:.2f}', mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()