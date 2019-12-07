import numpy as np
from sort.sort import Sort


DEFAULT_MAX_DISTANCE_BETWEEN_POINTS = 60 
DEFAULT_WARMUP_FRAMES = 8
DEFAUTL_MIN_HITS = 3
DEFAULT_MAX_AGE = 10
class MOT():

    def __init__(self, **kwargs):
        print(kwargs)
        self._state = {}
        self._statistics = []
        self._class_count = dict(zip(kwargs['class_ids'], np.zeros(len(kwargs['class_ids']))))
        # if distance between centers of two bboxes is less than _max_distance then object is staying
        self._max_distance = kwargs['max_distance'] if 'max_distance' in kwargs else DEFAULT_MAX_DISTANCE_BETWEEN_POINTS

        # after _warmup_frames we start to compare bbox's centers for one tracked object
        self._warmup_frames = kwargs['warmup_frames'] if 'warmup_frames' in kwargs else DEFAULT_WARMUP_FRAMES

        self._line_y = kwargs['line_y'] if 'line_y' in kwargs else 0

        min_hits = kwargs['min_hits'] if 'min_hits' in kwargs else DEFAUTL_MIN_HITS
        max_age = kwargs['max_age'] if 'max_age' in kwargs else DEFAULT_MAX_AGE
        #self.display_config()
        self._mot_tracker = Sort(max_age, min_hits)

    def display_config(self):
        print('line_y')
        print(self._line_y)
        print('warmup_frames')
        print(self._warmup_frames)
        print('max_distance')
        print(self._max_distance)

    def update_state(self, boxes, scores, classes, timestamp):
        dets = np.array(boxes)
        dets = np.hstack((dets, scores.reshape(scores.shape[0],1)))
        trackers, matched, unmatched_dets = self._mot_tracker.update(dets)
        boxes, scores, classes, ids = self.mot_output_postprocess(trackers, boxes, scores, classes, matched, unmatched_dets)
        filtered_inds, object_crossed = self.filter_moving_obj_ids(boxes, scores, classes, ids)

        if len(object_crossed) > 0:
             self._statistics.append({'timestamp': timestamp, 'class_count': self._class_count.copy(), 'objects': object_crossed})
     
        scores = scores.reshape((scores.shape[0],))
        classes = classes.reshape((classes.shape[0],))
        classes = classes.astype(int)

        return filtered_inds, boxes, scores, classes, ids

    def filter_moving_obj_ids(self, boxes, scores, classes, ids):
        filtered_inds = set()
        object_crossed = []
        for i, obj_id in enumerate(ids):
            top, left, bottom, right = boxes[i]
            w = right - left
            h = bottom - top
            x_c = left + w/2
            y_c = top + h/2
            if obj_id in self._state:
                state_obj = self._state[obj_id]
                if state_obj['frame_num'] <  self._warmup_frames:
                    state_obj['frame_num']+=1
                    self._state[obj_id] = state_obj
                else:
                    if not self.is_close([x_c, y_c], state_obj['origin_pos']) and \
                        state_obj['origin_pos'][1] < y_c:
                        filtered_inds.add(i)
                
                        if not state_obj['already_counted']:
                            origin_y = state_obj['origin_pos'][1]
                        
                            if state_obj['origin_pos'][1] < self._line_y and y_c >= self._line_y:
                                self._class_count[classes[i]] += 1
                                state_obj['already_counted'] = True
                                self._state[obj_id] = state_obj
                                object_crossed.append([classes[i], scores[i]])
                    
            else:
                new_obj = {'frame_num': 1, 'origin_pos': [x_c, y_c], 'already_counted': False}
                self._state[obj_id] = new_obj

        return filtered_inds, object_crossed

    def mot_output_postprocess(self, trackers, boxes, scores, classes, matched, unmatched_dets):
        trackers =trackers[::-1]
      
        matched = matched[matched[:,1].argsort()]
        new_ind = matched[:,0]

        boxes_unmathced = np.empty((0,4))
        scores_unmathced = np.empty((0,1))
        classes_unmathced = np.empty((0,1))
        if len(unmatched_dets) > 0:
            boxes_unmathced = boxes.take(unmatched_dets, axis = 0)
            scores_unmathced = scores.take(unmatched_dets, axis = 0)
            classes_unmathced = classes.take(unmatched_dets, axis = 0)

        boxes = trackers[:, 0:4]
        scores = scores.take(new_ind, axis = 0)
        classes = classes.take(new_ind, axis = 0)
       
        ids = trackers[:, 4]

        scores = scores.reshape(-1,1)
        classes = classes.reshape(-1,1)
        scores_unmathced = scores_unmathced.reshape(-1,1)
        classes_unmathced = classes_unmathced.reshape(-1,1)

        boxes = np.vstack((boxes, boxes_unmathced))
        scores = np.vstack((scores, scores_unmathced))
        classes = np.vstack((classes, classes_unmathced))
        
        scores = scores.reshape((-1,))
        classes = classes.reshape((-1,))

        return boxes, scores, classes, ids
        
    def get_class_count(self):
        return self._class_count
    
    def get_statistics(self):
        return self._statistics

    def is_close(self, point_1, point_2):
        dist = np.linalg.norm(np.array(point_1)-np.array(point_2))
        return dist < self._max_distance
        