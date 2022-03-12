import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

img = cv2.imread("me.png", -1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("a", img)
# cv2.waitKey(0)
face = faceCascade.detectMultiScale(img_gray)
for (x,y,w,h) in face:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 3)

cv2.imshow("a", img)
cv2.waitKey(0)