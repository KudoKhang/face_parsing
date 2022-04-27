import os
import cv2
from PIL import Image
import time
import numpy as np
import torch
from argparse import ArgumentParser
from ibug.face_detection import RetinaFacePredictor
from ibug.face_parsing import FaceParser as RTNetPredictor
from ibug.face_parsing.utils import label_colormap
from utils.inference_funcs import *

class Ibug_Parsing():
    def __init__(self, threshold=0.8, encoder='rtnet50', decoder='fcn',
                        num_classes=11, max_num_faces=50,
                        weights='./ibug/face_parsing/rtnet/weights/rtnet50-fcn-11.torch'):
        self.threshold = threshold
        self.encoder = encoder
        self.decoder = decoder
        self.num_classes = num_classes
        self.max_num_faces = max_num_faces
        self.weights = weights
        self.alphas = np.linspace(0.75, 0.25, num=self.max_num_faces)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.face_detector = RetinaFacePredictor(threshold=self.threshold, device=self.device,
                                            model=(RetinaFacePredictor.get_model('mobilenet0.25')))

        self.face_parser = RTNetPredictor(device=self.device,
                                     ckpt=self.weights,
                                     encoder=self.encoder,
                                     decoder=self.decoder,
                                     num_classes=self.num_classes)

        self.colormap = label_colormap(self.num_classes)
        print('Face detector created using RetinaFace.')

    def check_type(self, img_path):
        if type(img_path) == str:
            if img_path.endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(img_path)
            else:
                raise Exception("Please input a image file")
        elif type(img_path) == np.ndarray:
            img = img_path
        return img

    def drawSquare(self, img, x, y):
        YELLOW = (0, 255, 255)
        BLUE = (255, 225, 0)

        cv2.line(img, (x - 150, y - 150), (x - 100, y - 150), YELLOW, 2)
        cv2.line(img, (x - 150, y - 150), (x - 150, y - 100), BLUE, 2)

        cv2.line(img, (x + 150, y - 150), (x + 100, y - 150), YELLOW, 2)
        cv2.line(img, (x + 150, y - 150), (x + 150, y - 100), BLUE, 2)

        cv2.line(img, (x + 150, y + 150), (x + 100, y + 150), YELLOW, 2)
        cv2.line(img, (x + 150, y + 150), (x + 150, y + 100), BLUE, 2)

        cv2.line(img, (x - 150, y + 150), (x - 100, y + 150), YELLOW, 2)
        cv2.line(img, (x - 150, y + 150), (x - 150, y + 100), BLUE, 2)

        cv2.circle(img, (x, y), 5, (255, 255, 153), -1)

    def run(self, frame):
        # Detect faces
        start_time = time.time()
        frame = self.check_type(frame)
        faces = self.face_detector(frame, rgb=False)
        if len(faces) > 0:
            elapsed_time = time.time() - start_time
            print(f'Processed in {elapsed_time * 1000.0:.04f} ms: ' +
                  f'{len(faces)} faces detected.')

            # Parse faces
            start_time = time.time()
            masks = self.face_parser.predict_img(frame, faces, rgb=False)
            elapsed_time = time.time() - start_time

            print(f'Processed in {elapsed_time * 1000.0:.04f} ms: ' +
                  f'{len(masks)} faces parsed.')

            # Rendering
            dst = frame
            for i, (face, mask) in enumerate(zip(faces, masks)):
                bbox = face[:4].astype(int)
                x_center, y_center = int((bbox[2] - bbox[0]) / 2) + bbox[0], int((bbox[3] - bbox[1]) / 2) + bbox[1]
                self.drawSquare(frame, x_center, y_center)
                # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(
                #     0, 0, 255), thickness=2)
                alpha = self.alphas[i]
                index = mask > 0
                res = self.colormap[mask]
                dst[index] = (1 - alpha) * frame[index].astype(float) + \
                             alpha * res[index].astype(float)
            dst = np.clip(dst.round(), 0, 255).astype(np.uint8)
            frame = dst

            # return masks[0] # Mask voi cac gia tri [0,num_classes]
        return frame # Mask da len mau


#--------------------------------------------------------------------------------------------------
def image(path_img='image_test/g2.jpg'):
    mask = Face_parsing_predictor.run(path_img)
    cv2.imshow('Result', mask)
    cv2.waitKey(0)

def process_with_folder(input_dir, output_dir):
    if not os.path.exists(input_dir):
        raise Exception("Input_dir not exist. Please check again!")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path_dir = [name for name in os.listdir(input_dir) if name.endswith(('png', 'jpg', 'jpeg'))]
    for name in path_dir:
        mask = Face_parsing_predictor.run(os.path.join(input_dir ,name))
        cv2.imwrite(os.path.join(output_dir, name.split('.')[0] + '.png'), mask)
    print('\n Done!')

def video(path_video):
    print('Processing video... \nPlease wait...')
    cap = cv2.VideoCapture(path_video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    fps = 30
    out = cv2.VideoWriter('results_' + path_video.split('/')[-1], cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)

    while True:
        _, frame = cap.read()
        try:
            frame = Face_parsing_predictor.run(frame)
            out.write(frame)
        except:
            out.release()
            break
    out.release()
    print('Done!')

def webcam():
    print("Using webcam, press q to exit, press s to save")
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        start = time.time()
        frame = Face_parsing_predictor.run(frame)
        # FPS
        fps = round(1 / (time.time() - start), 2)
        cv2.putText(frame, "FPS : " + str(fps), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.imshow('Prediction', frame)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('s'):
            cv2.imwrite('image_out/' + str(time.time()) + '.jpg', frame)
        if k == ord('q'):
            break
#--------------------------------------------------------------------------------------------------
Face_parsing_predictor = Ibug_Parsing()
#--------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # image('image_test/g2.jpg')
    # video('dathao1.mp4')
    # webcam()
    process_with_folder(input_dir='image_test', output_dir='image_out')
