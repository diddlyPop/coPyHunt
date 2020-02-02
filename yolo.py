import numpy as np
import time
import cv2
import os


class ImageClassifier:

    def __init__(self):
        self.CONFIDENCE_LEVEL = 0.5
        self.THRESHOLD = 0.3

        labels_path = os.path.sep.join(["yolo-coco", "coco.names"])  # Pre-trained class labels
        self.LABELS = open(labels_path).read().strip().split("\n")

        self.last_classes = []

        # Initialize a list of colors to represent each possible class label
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")

        weights_path = os.path.sep.join(["yolo-coco", "yolov3.weights"])  # Paths to weights and classes
        config_path = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

        # Load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    def classify(self):
        self.last_classes.clear()
        # Load our input image and grab its spatial dimensions
        image = cv2.imread("image.png")
        (H, W) = image.shape[:2]

        # Determine only the output layer names that we need from YOLO
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # Construct a blob from the input image and then perform a forward pass. Gives us boxes and probabilities.
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layer_outputs = self.net.forward(ln)
        end = time.time()

        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        boxes = []
        confidences = []
        class_IDs = []

        # Loop over each of the layer outputs
        for output in layer_outputs:
            # Loop over each of the detections
            for detection in output:
                # Extract the class ID and confidence
                scores = detection[5:]
                class_ID = np.argmax(scores)
                confidence = scores[class_ID]

                # Test against preset confidence level
                if confidence > self.CONFIDENCE_LEVEL:
                    # Scale the bounding box coordinates back relative to the size of the image, keeping in mind
                    # that YOLO actually returns the center (x, y)-coordinates of the bounding box
                    # followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Use the center (x, y)-coordinates to derive the top and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # Update our list of bounding box coordinates, confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_IDs.append(class_ID)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_LEVEL, self.THRESHOLD)

        # Ensure at least one detection exists
        if len(idxs) > 0:
            # Loop over the indexes we are keeping
            for i in idxs.flatten():
                # Extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # Draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.COLORS[class_IDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.LABELS[class_IDs[i]], confidences[i])
                self.last_classes.append(self.LABELS[class_IDs[i]])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write the image back to disk
        cv2.imwrite('image.png', image)

    def listClassesFromClassify(self):
        return self.last_classes
