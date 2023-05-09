import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('path/to/haarcascade_frontalface_default.xml')

# Load the reference image
ref_image = cv2.imread('path/to/image/folder/308191143.jpg')

# Convert reference image to grayscale
ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

# Detect faces in reference image
ref_faces = face_cascade.detectMultiScale(ref_gray, scaleFactor=1.1, minNeighbors=4)

# To capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Set similarity threshold
    threshold = 0.9

    # Compare the faces
    for (x, y, w, h) in faces:
        # Crop the face region
        roi_gray = gray[y:y + h, x:x + w]

        # Resize the face region to match the reference image size
        roi_gray = cv2.resize(roi_gray, ref_gray.shape[::-1])

        # Calculate the absolute difference between the two face regions
        diff = cv2.absdiff(ref_gray, roi_gray)

        # Calculate the mean of the absolute difference as a measure of similarity
        similarity = cv2.mean(diff)[0]

        # Compare the similarity with the threshold
        if similarity > threshold:
            print('True')
        else:
            print('False')

        # Draw the rectangle around each face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the output
    cv2.imshow('Face Detection', img)

    # Stop if escape key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
