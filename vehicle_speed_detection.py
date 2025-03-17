import cv2

# Load the video file
video_path = "car_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Video Feed", frame)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
