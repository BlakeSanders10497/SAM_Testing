import argparse
from fastsam import FastSAM, FastSAMVideoPrompt 
import ast
import torch
from utils.tools import convert_box_xywh_to_xyxy

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM-x.pt", help="model"
    )
    parser.add_argument(
        "--vid_path", type=str, default="./videos/lake-tahoe-clip-1080p.mp4", help="path to video file"
    )
    parser.add_argument(
        "--show_frames", action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--num_frames", type=int, default=-1, help="number of frames to write to output video"
    )
    parser.set_defaults(show_frames=False)
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--text_prompt", type=str, default="the water", help='use text prompt eg: "a dog"'
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./output/", help="image save path"
    )
    parser.add_argument(
        "--random_color", action=argparse.BooleanOptionalAction
    )
    parser.set_defaults(random_color=False)
    parser.add_argument(
        "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument("--box_prompt", type=str, default="[[0,0,0,0]]", help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes")
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--with_contours", action=argparse.BooleanOptionalAction 
    )
    parser.set_defaults(with_contours=False)
    return parser.parse_args()


def main(args):
    # load model
    model = FastSAM(args.model_path)
    args.point_prompt = ast.literal_eval(args.point_prompt)
    args.box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(args.box_prompt))
    args.point_label = ast.literal_eval(args.point_label)

    # Read in video
    print(f'loading video: {args.vid_path}')
    video = cv2.VideoCapture(args.vid_path)
    success, img = video.read()
    assert success, f'Video read failed! Is the path right? ({args.vid_path})'

    # Set up video writer
    output_file= './output/lake-tahoe.mp4'
    writer = cv2.VideoWriter(filename=output_file,
                                fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                                fps=30,
                                frameSize=(int(video.get(3)), int(video.get(4))))

    assert writer.isOpened(), f'Video writer failed to open! Path: {output_file}'

    # Begin processing frames
    frame_count = 1
    while success:
        # Check for frame count limit
        if(args.num_frames != -1 and frame_count > args.num_frames):
            break
        
        # Get the results for this image
        everything_results = model(
            img,
            device=args.device,
            retina_masks=args.retina,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou    
            )
        bboxes = None
        points = None
        point_label = None

        # Use custom SAM Prompt instead so we pass just the image in
        prompt_process = FastSAMVideoPrompt(img, everything_results, device=args.device)

        assert args.text_prompt != None, "no text prompt!"

        ann = prompt_process.text_prompt(text=args.text_prompt)
        #if args.box_prompt[0][2] != 0 and args.box_prompt[0][3] != 0:
        #        ann = prompt_process.box_prompt(bboxes=args.box_prompt)
        #        bboxes = args.box_prompt
        #elif args.text_prompt != None:
        #    ann = prompt_process.text_prompt(text=args.text_prompt)
        #elif args.point_prompt[0] != [0, 0]:
        #    ann = prompt_process.point_prompt(
        #        points=args.point_prompt, pointlabel=args.point_label
        #    )
        #    points = args.point_prompt
        #    point_label = args.point_label
        #else:
        #    ann = prompt_process.everything_prompt()

        # Store result as a frame for the output video
        result = prompt_process.plot(
            annotations=ann,
            output=args.output,
            bboxes = bboxes,
            points = points,
            point_label = point_label,
            mask_random_color = args.random_color, # actually pass the random color arg
            withContours=args.with_contours,
            better_quality=args.better_quality,
        )

        # Show the image in a window for manual inspection
        if args.show_frames:
            cv2.imshow(f'frame{frame_count}', result)
            while cv2.getWindowProperty(f'frame{frame_count}', cv2.WND_PROP_VISIBLE) >= 1:
                key = cv2.waitKey(100)
                if key != -1:
                    cv2.destroyWindow(f'frame{frame_count}')
                    break

        # Write out a video frame
        writer.write(result)
        
        # Try to read the next frame in
        success, img = video.read()

        # Update frame counter
        frame_count += 1


    video.release()
    writer.release()



if __name__ == "__main__":
    args = parse_args()
    main(args)
