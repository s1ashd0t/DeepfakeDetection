# extract frames from a video
import cv2
import os

def extract_frames(path):

    if not os.path.exists('frames'): 
        os.makedirs('frames')

    cam = cv2.VideoCapture(path)
    
    count = 0

    sucess = 1

    while sucess:

        sucess, image = cam.read()

        if sucess:
            cv2.imwrite("./frames/frame%d.jpg" % count, image)
            count += 1
        else:
            cam.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':

    extract_frames("testvideo.mp4")