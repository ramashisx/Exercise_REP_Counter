import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


cap = cv2.VideoCapture("KneeBendVideo.mp4")

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), int(fps), (int(width), int(height)))

counter = 0
time = 0
time_add = 1/fps
angle = 180
timer_started = False
relaxed = True
error = False
relax_message = False

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        time += time_add

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z

            if right_shoulder - left_shoulder < 0:

                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            else:
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate angle
            new_angle = calculate_angle(hip, knee, ankle)

            # average out the new angle to remove the noise
            angle = (2/fps)*new_angle + (1-2/fps)*angle

            # Visualize angle
            cv2.putText(image, str(angle),
                        tuple(np.multiply(knee, [width, height]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Setup status box
            cv2.rectangle(image, (0, 0), (250, 120), (245, 117, 16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Counter logic
            if angle < 140 and not timer_started and relaxed:
                timer_started = True
                error = False
                relaxed = False
                relax_message = False
                start_time = time
            elif angle > 140 and timer_started:
                if time-start_time < 8:
                    error = True
                    relaxed = True
                    timer_started = False
                    relax_message = False
                else:
                    relax_message = True
                    counter += 1
                    error = False
                    timer_started = False
                    relaxed = False
            elif angle < 140 and timer_started and time-start_time >= 8:
                relax_message = True
                counter += 1
                timer_started = False
                error = False
                relaxed = False
            elif angle > 140:
                relaxed = True
                relax_message = False
                cv2.putText(image, "Bend Your Knee", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                timer_started = False



            if relax_message:
                cv2.putText(image, "Relax Your Knee", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            if error:
                cv2.putText(image, "Keep Your Knee Bent", (300, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


            if not relaxed:

                cv2.putText(image, str(counter) + "-" + "{:.2f}".format(time-start_time), (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            else:

                cv2.putText(image, str(counter) + "-", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)




        except:
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        out.write(image)

        # UNCOMMENT
        # cv2.imshow("FEED", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
