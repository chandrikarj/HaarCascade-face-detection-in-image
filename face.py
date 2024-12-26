import cv2

# Load the pre-trained classifier
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Read the image
img = cv2.imread("faceimg.jpeg")

# Convert to grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = faceCascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Show the image
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
