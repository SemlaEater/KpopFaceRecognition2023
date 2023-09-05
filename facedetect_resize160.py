import cv2

face_cascade_path = 'haarcascade/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)
image_path = 'test_face.jpg'            # 유저가 원하는 사진 입력

image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

if len(faces) == 0:
    print("얼굴을 찾을 수 없습니다.")
    exit()

(x, y, w, h) = faces[0]


center_x = x + w // 2
center_y = y + h // 2


crop_size = 256
x1 = center_x - crop_size // 2
y1 = center_y - crop_size // 2
x2 = x1 + crop_size
y2 = y1 + crop_size


if x1 < 0:
    x1 = 0
if y1 < 0:
    y1 = 0
if x2 > image.shape[1]:
    x2 = image.shape[1]
if y2 > image.shape[0]:
    y2 = image.shape[0]


face_crop = image[y1:y2, x1:x2]


face_resized = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_AREA)


cv2.imshow('Result', face_resized)
cv2.waitKey(0)


output_image_path = 'output_image.jpg'
cv2.imwrite(output_image_path, face_resized)

print("출력 이미지 경로:", output_image_path)
