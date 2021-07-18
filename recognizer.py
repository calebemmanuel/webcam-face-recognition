import face_recognition
import cv2

video_capture = cv2.VideoCapture(0)

jeff_image = face_recognition.load_image_file("Jeff.jpg")
jeff_face_encoding = face_recognition.face_encodings(jeff_image)[0]

elon_image = face_recognition.load_image_file("Elon.jpg")
elon_face_encoding = face_recognition.face_encodings(elon_image)[0]

bill_image = face_recognition.load_image_file("Bill.jpg")
bill_face_encoding = face_recognition.face_encodings(bill_image)[0]

larry_image = face_recognition.load_image_file("Larry.jpg")
larry_face_encoding = face_recognition.face_encodings(larry_image)[0]

mark_image = face_recognition.load_image_file("mark.jpg")
mark_face_encoding = face_recognition.face_encodings(mark_image)[0]

known_face_encodings = [
    
    jeff_face_encoding,
    elon_face_encoding,
    bill_face_encoding,
    larry_face_encoding,
    mark_face_encoding

]
known_face_names = [

    "Jeff bezos",
    "Elon musk",
    "Bill gate",
    "Larry page",
    "Mark zuckerberg"

]

while True:

    ret, frame = video_capture.read()

    rgb_small_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)


    for (top, right, bottom, left), face_encodings in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encodings)

        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]


        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
