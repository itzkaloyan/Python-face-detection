import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image_paths = ['photo1.jpg', 'photo2.jpg']

for image_path in image_paths:
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image {image_path}")
        continue

    height, width = img.shape[:2]
    ratio = width / height
    height = 640
    width = height * ratio
    resized = cv2.resize(img, (int(width), int(height)))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    #detects
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #draws rectangles around the found faces
    for (x, y, w, h) in faces:
        cv2.rectangle(resized, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow(f"Face Detection - {image_path}", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
