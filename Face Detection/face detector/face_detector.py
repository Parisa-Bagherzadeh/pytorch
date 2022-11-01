import cv2
import numpy as np

import torch
from facenet_pytorch import MTCNN


class FaceDetecotr():
    def __init__(self,mtcnn):
        self.mtcnn = mtcnn

    def draw(self, frame, boxes, probs, landmarks):

        try:
            for box, prob,ld in zip(boxes, probs, landmarks):
                cv2.rectangle(frame,
                (box[0], box[1]),
                (box[2], box[3]),
                (0, 0, 255),
                thickness = 2)

                cv2.putText(frame, str(prob),(box[2], box[3], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.Line_AA))        

                cv2.circle(frame, tuple(ld[0]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[1]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 255), -1)
        except:
            pass

        return frame

    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()

            try:
                boxes, probs , landmarks = self.mtcnn.detect(frame, lanedmarks = True)
                self.draw(frame, boxes, probs, landmarks)

            except:
                pass

            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    mtcnn = MTCNN()
    fcd = FaceDetecotr(mtcnn)
    fcd.run()
    