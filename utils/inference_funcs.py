import os
from tqdm import tqdm

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