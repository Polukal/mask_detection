import cv2
import torch
import numpy as np
from config import Config


class WebcamDemo:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()wq

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open webcam")

    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        # Resize
        frame = cv2.resize(frame, Config.IMAGE_SIZE)

        # Convert to RGB and normalize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame / 255.0

        # Convert to tensor
        frame = torch.FloatTensor(frame).permute(2, 0, 1).unsqueeze(0)
        return frame.to(self.device)

    def draw_predictions(self, frame, boxes, labels):
        """Draw bounding boxes and labels on frame"""
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, box)
            class_name = Config.CLASSES[label]

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            cv2.putText(frame, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def run(self):
        """Run real-time detection"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Preprocess frame
                input_tensor = self.preprocess_frame(frame)

                # Get predictions
                with torch.no_grad():
                    outputs = self.model(input_tensor)

                # Process predictions
                boxes, labels = self.process_predictions(outputs)

                # Draw predictions on frame
                frame = self.draw_predictions(frame, boxes, labels)

                # Show frame
                cv2.imshow('Face Mask Detection', frame)

                # Break on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def process_predictions(self, outputs):
        """Process model outputs for visualization"""
        if isinstance(outputs, tuple):  # SSD output
            classifications, boxes = outputs
            boxes = boxes.cpu().numpy()[0]
            labels = classifications.argmax(dim=1).cpu().numpy()[0]
        else:  # YOLO output
            # Process YOLO outputs
            boxes = []
            labels = []
            for output in outputs:
                b, l = self.process_yolo_output(output)
                boxes.extend(b)
                labels.extend(l)

        # Filter by confidence
        mask = np.array([score > Config.CONFIDENCE_THRESHOLD for score in labels])
        boxes = boxes[mask]
        labels = labels[mask]

        return boxes, labels