from opencv import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # 1. Découper le visage
        roi_gray = gray[y:y+h, x:x+w]
        
        # 2. Redimensionner à la taille du dataset (IMPORTANT)
        roi_resized = cv2.resize(roi_gray, (92, 112)) # Inverse de shape ! (Largeur, Hauteur)
        
        # 3. Prédire
        label, dist = predict_face(roi_resized, mean_face, eigenfaces, train_weights, y_train)
        
        # 4. Dessiner le rectangle et le nom
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"{label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('Reconnaissance Faciale', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()