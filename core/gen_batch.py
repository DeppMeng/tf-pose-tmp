import os
import random

import cv2
import numpy as np

# from config import cfg


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def normalize_input(img, pixel_means):
    return img - pixel_means

class BatchGeneration(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def generate_batch(self, data, stage='train'):
        cfg = self.cfg
        img = cv2.imread(os.path.join(cfg.img_path, data['imgpath']), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            print('cannot read ' + os.path.join(cfg.img_path, data['imgpath']))
            assert 0

        bbox = np.array(data['bbox']).astype(np.float32)

        x, y, w, h = bbox
        aspect_ratio = cfg.input_shape[1] / cfg.input_shape[0]
        center = np.array([x + w * 0.5, y + h * 0.5])
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        scale = np.array([w, h]) * 1.25
        rotation = 0

        if stage == 'train':

            joints = np.array(data['joints']).reshape(cfg.num_kps, 3).astype(np.float32)

            # data augmentation
            scale = scale * np.clip(np.random.randn() * cfg.scale_factor + 1, 1 - cfg.scale_factor, 1 + cfg.scale_factor)
            rotation = np.clip(np.random.randn() * cfg.rotation_factor, -cfg.rotation_factor * 2, cfg.rotation_factor * 2) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5:
                img = img[:, ::-1, :]
                center[0] = img.shape[1] - 1 - center[0]
                joints[:, 0] = img.shape[1] - 1 - joints[:, 0]
                for (q, w) in cfg.kps_symmetry:
                    joints_q, joints_w = joints[q, :].copy(), joints[w, :].copy()
                    joints[w, :], joints[q, :] = joints_q, joints_w

            trans = get_affine_transform(center, scale, rotation, (cfg.input_shape[1], cfg.input_shape[0]))
            cropped_img = cv2.warpAffine(img, trans, (cfg.input_shape[1], cfg.input_shape[0]), flags=cv2.INTER_LINEAR)
            # cropped_img = cropped_img[:,:, ::-1]
            cropped_img = normalize_input(cropped_img, cfg.pixel_means)

            for i in range(cfg.num_kps):
                if joints[i, 2] > 0:
                    joints[i, :2] = affine_transform(joints[i, :2], trans)
                    joints[i, 2] *= ((joints[i, 0] >= 0) & (joints[i, 0] < cfg.input_shape[1]) & (joints[i, 1] >= 0) & (
                            joints[i, 1] < cfg.input_shape[0]))
            target_coord = joints[:, :2]
            target_valid = joints[:, 2]

            return [cropped_img, target_coord, (target_valid > 0)]

        else:
            trans = get_affine_transform(center, scale, rotation, (cfg.input_shape[1], cfg.input_shape[0]))
            cropped_img = cv2.warpAffine(img, trans, (cfg.input_shape[1], cfg.input_shape[0]), flags=cv2.INTER_LINEAR)
            # cropped_img = cropped_img[:,:, ::-1]
            cropped_img = normalize_input(cropped_img, cfg.pixel_means)

            crop_info = np.asarray([center[0] - scale[0] * 0.5, center[1] - scale[1] * 0.5, center[0] + scale[0] * 0.5,
                                    center[1] + scale[1] * 0.5])

            return [cropped_img, crop_info]
