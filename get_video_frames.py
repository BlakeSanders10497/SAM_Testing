import cv2
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path", type=str, default="./videos/lake-tahoe-clip-1080p.mp4", help="path to video file"
    )
    parser.add_argument(
        "--num_frames", type=int, default=5, help="number of frames to generate from the input video"
    )
    return parser.parse_args()


def main(args):
    video = cv2.VideoCapture(args.video_path)

    success, image = video.read()
    frame_count = 1

    while success and frame_count <= args.num_frames:
        cv2.imwrite(f'images/video_frames/frame{frame_count}.jpg', image)
        
        success, image = video.read()
        frame_count += 1

    if(not success):
        print(f'Video read unsuccessful on frame {frame_count}.')

    


if __name__ == "__main__":
    args = parse_args()
    main(args)
