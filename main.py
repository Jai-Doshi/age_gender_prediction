from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Images/jai.jpeg")
plt.imshow(img[:,:,::-1])
plt.show()

result = DeepFace.analyze(img, actions=['age', 'gender'])

print("Gender: ", result[0]['gender'])
print("Age: ", result[0]['age'])

# import cv2
# from deepface import DeepFace

# cap = cv2.VideoCapture(0)

# while True:
#     success, img = cap.read()
#     DeepFace.stream("database")
#     cv2.imshow("Image", img)
#     if cv2.waitKey(1) == ord('q'):
#         break
