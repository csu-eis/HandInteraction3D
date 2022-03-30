import pyrealsense2 as rs
import numpy as np
import paddle
import cv2
import time
from net.internet.common.utils.preprocessing import load_skeleton, gen_black_image
from net.internet.common.utils.vis import vis_keypoints_cv2
from net.ppyolo.detect import PredictConfig, Detector, generate_od_input, merge_bbox,generate_od_input_from_bgr
from net.internet.model import get_model
from net.internet.config import Config
from utils.filters import OneEuroFilter


class Timer():
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t = time.time() - self.start
        print(f"{self.name}: {self.t * 1000:.2f}ms")



if __name__ == "__main__":

    filter1= OneEuroFilter()
    filter2 = OneEuroFilter()

    # 加载 ppyolo
    detector_dir = "weights/ppyolo_r18vd_voc"
    detector_cfg = PredictConfig(detector_dir)
    detector = Detector(
        detector_cfg,
        detector_dir,
        use_gpu=True,
        run_mode="trt_fp32",
        threshold=detector_cfg.threshold
    )
    print("Detector loaded.")
    #
    #
    predictor_weight_path = "weights/internet/snapshot_90.pdparams"
    predictor_cfg = Config()
    predictor = get_model("test", predictor_cfg.joint_nums)
    state_dict = paddle.load(predictor_weight_path)
    if "network" in state_dict:
        predictor.load_dict(state_dict["network"])
    else:
        predictor.load_dict(state_dict)
    predictor.eval()
    print("Predictor loaded.")
    #
    # # this dict describes the link between joints
    skeleton_file_path = "skeleton.txt"
    skeleton = load_skeleton(skeleton_file_path, predictor_cfg.joint_nums * 2)

    idx = 0
    isBusy = True
    # init Camera
    cap = cv2.VideoCapture(0)

    prev_coord = []
    while True:
        idx += 1

        ret, frame = cap.read()

        # generate Tensor for detect
        image_src, input = generate_od_input_from_bgr(frame, detector_cfg)

        bboxes = detector.predict(input)
        # check bbox in case cause program dead
        if bboxes.size == 0:
            continue

        for bbox in bboxes:
            if bbox[1] > 0.45:
                image = cv2.rectangle(image_src, (int(bbox[2]), int(bbox[3])), (int(bbox[4]), int(bbox[5])), (0, 0, 255), 2)

        # with Timer('merge bbox and generate black image'):
        merged_box = bboxes[0]
        if bboxes.size == 12:
            merged_box = merge_bbox(bboxes[0], bboxes[1])
            image_black = gen_black_image(image_src, bboxes[0, 2:], bboxes[1, 2:])
        else:
            image_black = gen_black_image(image_src, bboxes[0, 2:], bboxes[0, 2:])

        if merged_box[0] == 0.0 and merged_box[1] > 0.45:
            _coords, root_depth, hand_score, coords,_ = predictor.predict(image_black, merged_box, image_coordinate=False)
            _coords=filter1.filter(_coords)
            coords=filter2.filter(coords)
            right_hand_coords = coords[:21] * (hand_score[0] >0.5)
            left_hand_coords = coords[21:] * (hand_score[1] > 0.5)
            if hand_score[0] < 0.9:
                _coords[:21] = 0
            elif hand_score[1] < 0.9:
                _coords[21:] = 0

            image = vis_keypoints_cv2(image, _coords, skeleton)
        #     print(idx)

        cv2.imshow('Realsense',image)
        cv2.waitKey(10)

