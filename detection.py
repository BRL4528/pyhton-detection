import cv2
import numpy as np

cap = cv2.VideoCapture("pig.mp4")

whT = 320

confThreshold = 0.5

nmsThreshold  = 0.3

classNames = ["pig"]

modelConfiguration = "yolov3-tiny.cfg"
modelWeights = "yolov3-tiny_2000.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs, img):

    hT, wT, cT = img.shape
    boundingBoxes = []
    classIndexes = []
    confidenceValues = []

    for output in outputs:

      for detection in output:

        probScores = detection[5:]
        classIndex = np.argmax(probScores)
        confidence = probScores[classIndex]

        if confidence >= confThreshold:

          w, h = int(detection[2]*wT), int(detection[3]*hT)
          x, y = int((detection[0]*wT)-w/2), int((detection[1]*hT)-h/2)

          boundingBoxes.append([x, y, w, h])
          classIndexes.append(classIndex)
          confidenceValues.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boundingBoxes, confidenceValues, confThreshold, nmsThreshold)

    for i in indices:
      box = boundingBoxes[i]
      x, y, w, h = box[0], box[1], box[2], box[3]

      cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
      cv2.rectangle(img, (x - 1, y - 25), (x + w+1, y), (0, 255, 0), cv2.FILLED)
      cv2.putText(img, f'{classNames[classIndexes[i]].upper()} {int(confidenceValues[i] * 100)}%', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


while cap.isOpened():
  success, frame = cap.read()

  if success: 
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], swapRB=True, crop=False)

    net.setInput(blob)

    outputLayerNames = net.getUnconnectedOutLayersNames()
    print(f"Outout Layers Name: {outputLayerNames}")

    outputs = net.forward(outputLayerNames)

    print(f"Total Number of output layers: {len(outputs)}")
    print(f"Shape of 1st output layer: {outputs[0].shape}")
    print(f"Shape of 2nd output layer: {outputs[1].shape}")
    # print(f"Shape of 3nd output layer: {outputs[2].shape}")
    print(f"Inside details of 1st output layer: {outputs[0][0]}")

    findObjects(outputs, frame)

    cv2.imshow('Pig Detecition', frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
      break
  
  else: 
    print("Did not read the frame")

