import numpy as np
from yolo import YOLO
from mot import MOT
import cv2
from timeit import default_timer as timer
import warnings
warnings.filterwarnings('ignore')

def draw_line(image, line_y, color, thickness):
    x1 = 0
    y1 = int(line_y)
    x2 = image.shape[1]
    y2 = int(line_y)
    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image

# use to filter vehicle classes when use coco weights (80 classess)
def filter_vehicle_classes(boxes, scores, classes):
    vehicle_classes = {2, 5, 7}
    inds = np.arange(classes.shape[0])
    filtered_inds = [i for i in inds if classes[i] in vehicle_classes]
   
    boxes = boxes[filtered_inds]
    scores = scores[filtered_inds]
    classes = classes[filtered_inds]

    return boxes, scores, classes

def detect_video(yolo, input_path, output_video, h_reltive_pos, coco_weights=False):
    vid = cv2.VideoCapture(input_path)
    if not vid.isOpened():
        raise IOError("Couldn't open video")

    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    class_names = yolo.get_class_names()
    class_ids = np.arange(len(class_names))
    if coco_weights:
        class_names = ['car', 'bus', 'truck']
        class_ids = [2,5,7]

    line_y = video_size[1]*h_reltive_pos

    mot_config = {'min_hits': 3, 'max_age':10, 'line_y': line_y,
                  'max_distance': 90, 'warmup_frames': 10,
                  'class_ids': class_ids}
    
    mot_tracker = MOT(**mot_config)

    print("info: ", output_video, video_FourCC, video_fps, video_size)
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'MJPG'), video_fps, video_size)
   
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    nb_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'frames num = {nb_frames}')
    for i in range(nb_frames):
        return_value, frame = vid.read()
        boxes, scores, classes = yolo.predict(frame)
        
        if coco_weights:
            boxes, scores, classes = filter_vehicle_classes(boxes, scores, classes)

        filtered_inds, boxes, scores, classes, ids = mot_tracker.update_state(boxes, scores, classes, timestamp=int(i//video_fps))

        class_count = mot_tracker.get_class_count()
        image = yolo.draw_boxes(frame, boxes, scores, classes, filtered_inds, ids, class_count, line_y, 5)
        image = draw_line(image, line_y, color=(255,255,255), thickness=4)

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        if i % 50 == 0:
            print(f'fps = {fps}')
        out.write(np.uint8(image))
    
    statistics = mot_tracker.get_statistics()

    vid.release()
    out.release()

    #write statistics to file
    statistics_path = str(output_video)[:-4] + '_statistics.txt'
    with open(statistics_path, 'w') as file:
        file.write(f'input video: {input_path}')
        file.write('\n')
        file.write(f'output video: {output_video}')
        file.write('\n')
        file.write(f'line position: {h_reltive_pos}')
        file.write('\n')
        for stat in statistics:
            timestamp = stat['timestamp']
            class_count = stat['class_count']
            total = int(sum(class_count.values()))
            if coco_weights:
                helper_dict = {2:0, 5:1, 7:2}
                items_cnt = ['{}: {}'.format(class_names[helper_dict[key]], int(val)) for key, val in class_count.items()]
                objects_str = '; '.join(list(map(lambda x: 'class: {0}; pred: {1}'.format(class_names[helper_dict[x[0]]], x[1]),
                                                 stat['objects'])))
            else:
                items_cnt = ['{}: {}'.format(class_names[key], int(val)) for key, val in class_count.items()]
                objects_str = '; '.join(list(map(lambda x: f'class: {int(x[0])}; pred: {x[1]}', stat['objects'])))
            items_cnt_str = ', '.join(items_cnt)
            whole_count_text = f'total: {total}, {items_cnt_str}'
            line = f'timestamp: {timestamp} sec.;    {whole_count_text};    objects: {objects_str}'
            file.write(line)
            file.write('\n')

    yolo.close_session()



if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='analyse traffic')

    # Required parameters
    parser.add_argument('--input', required=True,
                        metavar="/path/to/input/video",
                        help='Path to video')

    # Optional parameters
    parser.add_argument('--line', required=False,
                        metavar="horizontal line position",
                        help='Relative horizontal line position. 0.5 will be line in the middle',
                        default=0.33)

    parser.add_argument('--output', required=False,
                        metavar="/path/to/output/video",
                        help="Path to output video",
                        default="")

    parser.add_argument('--weights', required=False,
                        metavar="path/to/weight/ or put 'coco'",
                        help="Path to weights or put 'coco'",
                        default='coco')

    parser.add_argument('--classes', required=False,
                        metavar="path/to/classes/file",
                        help="Path to classes file in case you do not use coco")

   
    args = parser.parse_args()

    if args.output == '':
    	args.output = str(args.input)[:-4] + '_proc.mp4'

    print("Input video path: ", args.input)
    print("Output video path: ", args.output)
    print("Weights path: ", args.weights)
    print("Classes path: ", args.classes)
    print("Horizontal line position: ", args.line)

    use_coco = False
    if args.weights == 'coco':
    	print('create yolo model with coco weight')
    	use_coco = True
    	yolo = YOLO()
    else:
        config = {'model_path': args.weights, 'classes_path': args.classes }
        yolo = YOLO(**config)

    detect_video(yolo, args.input, args.output, float(args.line), use_coco)

    print('Process has finished')
