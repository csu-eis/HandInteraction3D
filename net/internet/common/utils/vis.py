# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import os.path as osp
import cv2
import numpy as np
import matplotlib

# matplotlib.use('tkagg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from net.internet.config import cfg
from PIL import Image, ImageDraw


def get_keypoint_rgb(skeleton):
    rgb_dict = {}
    for joint_id in range(len(skeleton)):
        joint_name = skeleton[joint_id]['name']

        # RGB->BGR
        if "thumb" in joint_name:
            rgb_dict[joint_name] = (0, 0, 255)
        elif "index" in joint_name:
            rgb_dict[joint_name] = (0, 255, 0)
        elif "middle" in joint_name:
            rgb_dict[joint_name] = (0, 128, 255)
        elif "ring" in joint_name:
            rgb_dict[joint_name] = (255, 128, 0)
        elif "pinky" in joint_name:
            rgb_dict[joint_name] = (255, 0, 255)
        else:
            rgb_dict[joint_name] = (0, 230, 230)

    return rgb_dict


def vis_keypoints(img, kps, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad=3, save_path=None):
    rgb_dict = get_keypoint_rgb(skeleton)
    _img = Image.fromarray(img.transpose(1, 2, 0).astype('uint8'))
    draw = ImageDraw.Draw(_img)
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
        kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=rgb_dict[parent_joint_name],
                      width=line_width)
        if score[i] > score_thr:
            draw.ellipse(
                (kps[i][0] - circle_rad, kps[i][1] - circle_rad, kps[i][0] + circle_rad, kps[i][1] + circle_rad),
                fill=rgb_dict[joint_name])
        if score[pid] > score_thr and pid != -1:
            draw.ellipse((kps[pid][0] - circle_rad, kps[pid][1] - circle_rad, kps[pid][0] + circle_rad,
                          kps[pid][1] + circle_rad), fill=rgb_dict[parent_joint_name])

    if save_path is None:
        _img.save(osp.join(cfg.vis_dir, filename))
    else:
        _img.save(osp.join(save_path, filename))


def vis_keypoints_cv2(img_src: np.ndarray, kps: np.ndarray, skeleton: dict, thickness=3, circle_rad=3):
    img = img_src.copy()
    rgb_dict = get_keypoint_rgb(skeleton)

    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        x = (int(kps[i][0]), int(kps[i][1]))
        y = (int(kps[pid][0]), int(kps[pid][1]))
        cv2.line(img, x, y, rgb_dict[parent_joint_name], thickness=thickness)
        cv2.circle(img, x, circle_rad, rgb_dict[joint_name], thickness=thickness)
        cv2.circle(img, y, circle_rad, rgb_dict[parent_joint_name], thickness=thickness)

    return img


def vis_3d_keypoints(kps_3d, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad=3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    rgb_dict = get_keypoint_rgb(skeleton)

    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        x = np.array([kps_3d[i, 0], kps_3d[pid, 0]])
        y = np.array([kps_3d[i, 1], kps_3d[pid, 1]])
        z = np.array([kps_3d[i, 2], kps_3d[pid, 2]])

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            ax.plot(x, z, -y, c=np.array(rgb_dict[parent_joint_name]) / 255., linewidth=line_width)
        if score[i] > score_thr:
            ax.scatter(kps_3d[i, 0], kps_3d[i, 2], -kps_3d[i, 1], c=np.array(rgb_dict[joint_name]).reshape(1, 3) / 255.,
                       marker='o')
        if score[pid] > score_thr and pid != -1:
            ax.scatter(kps_3d[pid, 0], kps_3d[pid, 2], -kps_3d[pid, 1],
                       c=np.array(rgb_dict[parent_joint_name]).reshape(1, 3) / 255., marker='o')

    # plt.show()
    # cv2.waitKey(0)

    fig.savefig(osp.join(cfg.vis_dir, filename), dpi=fig.dpi)

