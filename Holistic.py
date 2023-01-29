import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
holistic = mp_holistic.Holistic(min_detection_confidence=0.7,min_tracking_confidence=0.7)

while True:
    success, image =  cap.read()
    imageRgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    result = holistic.process(imageRgb)
    image.flags.writeable = True
    mp_drawing.draw_landmarks(
        image,
        result.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        result.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break