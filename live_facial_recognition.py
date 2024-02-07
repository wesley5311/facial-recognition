import cv2

face_class = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_class = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

video_capture = cv2.VideoCapture(0)
def detect_bounding_box(vid):
    frame_copy = vid.copy()
    gray_image=cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    faces = face_class.detectMultiScale(gray_image, 1.1, 5, minSize = (40,40))
    eyes = eye_class.detectMultiScale(gray_image, 1.1, 5, minSize = (10,10))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (255, 0, 0), 4)
    for (x, y, w, h) in eyes:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)

while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(video_frame) 
    eyes = detect_bounding_box(video_frame)
    cv2.imshow(
        "My Face Detection Project", video_frame
    )  

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()