import cv2
import os
from  my_face_reqnize import my_face_reqnize

mfr=my_face_reqnize()
mfr.load_encoding_images('faces')
cap=cv2.VideoCapture(2)
while True:
    ret,frame=cap.read()
    # face dectect
    face_location,face_name=mfr.detect_known_faces(frame)

    for (face_loc, name) in zip(face_location, face_name):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (200, 0, 00), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 0, 0), 4)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()



