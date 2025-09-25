# author: 通信2301 肖远志 202331223130
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 从MediaPipe的FACE_OVAL中提取所有脸部轮廓点并去重
FACE_OVAL = set()
for connection in mp_face_mesh.FACEMESH_FACE_OVAL:
    FACE_OVAL.add(connection[0])
    FACE_OVAL.add(connection[1])
FACE_POINTS_IDX = sorted(FACE_OVAL)

# 筛选出有用的关键点
LEFT_EYEBROW_IDX = [70, 63, 105, 66, 107]
RIGHT_EYEBROW_IDX = [336, 296, 334, 293, 300]
NOSE_IDX = [168, 6, 197, 195, 5, 4]
LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDX = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
MOUTH_IDX = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 191, 80, 81, 82, 13, 312, 311, 310]

ALL_IDX = FACE_POINTS_IDX + LEFT_EYEBROW_IDX + RIGHT_EYEBROW_IDX + NOSE_IDX + LEFT_EYE_IDX + RIGHT_EYE_IDX + MOUTH_IDX


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            h, w, _ = frame.shape
            if results.multi_face_landmarks:
                for face in results.multi_face_landmarks:

                    for idx in ALL_IDX:
                        lm = face.landmark[idx]
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                    # 脸部连线
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face,
                        connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
                    )

                    # 左眼连线
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face,
                        connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                    )

                    # 右眼连线
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face,
                        connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                    )

                    # 嘴巴连线
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face,
                        connections=mp_face_mesh.FACEMESH_LIPS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)
                    )
            cv2.imshow("Face", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

