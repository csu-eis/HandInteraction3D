import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as _transforms
import numpy as np

from net.internet.common.nets.module import BackboneNet,PoseNet
from net.internet.common.nets.loss import JointHeatmapLoss,HandTypeLoss,RelRootDepthLoss,TwistLoss,Joint2DLoss
from net.internet.common.utils.preprocessing import generate_patch_image, process_bbox
from net.internet.config import cfg


class Model(nn.Layer):

    def __init__(self,backbone_net,pose_net):
        super(Model, self).__init__()

        self.backbone_net=backbone_net
        self.pose_net=pose_net

        self.joint_heatmap_loss=JointHeatmapLoss()
        self.rel_root_depth_loss=RelRootDepthLoss()
        self.hand_type_loss=HandTypeLoss()
        # self.twist_loss=TwistLoss()
        self.joint_2d_loss= Joint2DLoss()

    def render_gaussian_heatmap(self, joint_coord):
        x = paddle.arange(cfg.output_hm_shape[2])
        y = paddle.arange(cfg.output_hm_shape[1])
        z = paddle.arange(cfg.output_hm_shape[0])
        zz, yy, xx = paddle.meshgrid(z, y, x)
        xx = paddle.unsqueeze(xx, axis=[0, 0]).astype(paddle.float32).cuda()
        yy = paddle.unsqueeze(yy, axis=[0, 0]).astype(paddle.float32).cuda()
        zz = paddle.unsqueeze(zz, axis=[0, 0]).astype(paddle.float32).cuda()

        x = paddle.unsqueeze(joint_coord[:, :, 0], axis=[-1, -1, -1])
        y = paddle.unsqueeze(joint_coord[:, :, 1], axis=[-1, -1, -1])
        z = paddle.unsqueeze(joint_coord[:, :, 2], axis=[-1, -1, -1])
        heatmap = paddle.exp(
            -(((xx - x) / cfg.sigma) ** 2) / 2 - (((yy - y) / cfg.sigma) ** 2) / 2 - (((zz - z) / cfg.sigma) ** 2) / 2)
        heatmap = heatmap * 255
        return heatmap

    def forward(self, inputs: dict, targets, meta_info, mode):
        input_img = inputs['img']
        batch_size = input_img.shape[0]
        img_feat = self.backbone_net(input_img)
        joint_heatmap_out, rel_root_depth_out, hand_type = self.pose_net(img_feat)

        if mode == 'train':
            target_joint_heatmap = self.render_gaussian_heatmap(targets['joint_coord'])

            loss = {}

            loss['joint_heatmap'] = self.joint_heatmap_loss(joint_heatmap_out, target_joint_heatmap,
                                                            meta_info['joint_valid']) #+  self.twist_loss(joint_heatmap_out, meta_info['joint_valid'])
            loss['rel_root_depth'] = self.rel_root_depth_loss(rel_root_depth_out, targets['rel_root_depth'],
                                                              meta_info['root_valid'])
            loss['joint_2d_loss'] = self.joint_2d_loss(joint_heatmap_out,targets['joint_coord'],meta_info['joint_valid'])
            loss['hand_type'] = self.hand_type_loss(hand_type, targets['hand_type'], meta_info['hand_type_valid'])
            return loss
        elif mode == 'test':
            out = {}
            out['heatmap']=joint_heatmap_out
            val_z, idx_z = paddle.max(joint_heatmap_out, axis=2), paddle.argmax(joint_heatmap_out, axis=2)
            val_zy, idx_zy = paddle.max(val_z, axis=2), paddle.argmax(val_z, axis=2)
            val_zyx, joint_x = paddle.max(val_zy, axis=2), paddle.argmax(val_zy, axis=2)
            
            batch_size = joint_heatmap_out.shape[0]
            num_joint = joint_heatmap_out.shape[1] // 2
            index_x = paddle.squeeze(joint_x)
            joint_x = paddle.unsqueeze(joint_x, axis=-1)

            shape = (-1, 1) if batch_size > 1 else (-1, )
            idx = paddle.concat((
                paddle.arange(0, batch_size).reshape(shape).expand_as(index_x).reshape((-1, 1)),
                paddle.arange(0, num_joint * 2).expand_as(index_x).reshape((-1, 1)),
                index_x.reshape((-1, 1))
            ),
                axis=1)
            joint_y = paddle.gather_nd(idx_zy, idx).reshape((batch_size, num_joint * 2, 1))
            index_y = paddle.squeeze(joint_y)

            idx = paddle.concat((
                paddle.arange(0, batch_size).reshape(shape).expand_as(index_x).reshape((-1, 1)),
                paddle.arange(0, num_joint * 2).expand_as(index_x).reshape((-1, 1)),
                index_y.reshape((-1, 1)),
                index_x.reshape((-1, 1))
            ),
                axis=1)
            joint_z = paddle.gather_nd(idx_z, idx).reshape((batch_size, num_joint * 2, 1))

            joint_coord_out = paddle.concat((joint_x, joint_y, joint_z), 2).astype(paddle.float32)
            out['joint_coord'] = joint_coord_out
            out['rel_root_depth'] = rel_root_depth_out
            out['hand_type'] = hand_type
            if 'inv_trans' in meta_info:
                out['inv_trans'] = meta_info['inv_trans']
            if 'joint_coord' in targets:
                out['target_joint'] = targets['joint_coord']
            if 'joint_valid' in meta_info:
                out['joint_valid'] = meta_info['joint_valid']
            if 'hand_type_valid' in meta_info:
                out['hand_type_valid'] = meta_info['hand_type_valid']
            return out

    def predict(self, img, bbox,
                image_coordinate=False,
                joint_num=21,
                input_image_shape=(256, 256),
                output_hm_shape=(64, 64, 64),
                bbox_3d_size=400,
                bbox_3d_size_root=400,
                output_root_hm_shape=64):
        # bbox type: (xmin, ymin, xmax, ymax) or (label, score, xmin, ymin, xmax, ymax)
        # with self.Timer('keypoint detection preprocess'):
        bbox = np.array(bbox)
        if bbox.shape[0] == 6:
            bbox = bbox[2:]

        # xyxy to xywh
        bbox[2], bbox[3] = bbox[2] - bbox[0], bbox[3] - bbox[1]
        bbox = process_bbox(bbox, input_image_shape, expand_ratio=1.25)
        img, trans, inv_trans = generate_patch_image(img, bbox, False, 1.0, 0.0, input_image_shape)
        img = _transforms.ToTensor()(img.astype(np.float32)) / 255
        img = paddle.unsqueeze(img, axis=0)

        inputs = {'img': img}
        targets = {}
        meta_info = {}
        with paddle.no_grad():
            out = self(inputs, targets, meta_info, 'test')

        joint_coord = out['joint_coord'][0].cpu().numpy()  # x,y pixel, z root-relative discretized depth
        rel_root_depth = out['rel_root_depth'][0].cpu().numpy()  # discretized depth
        score = out['hand_type'][0].cpu().numpy()

        joint_type = {'right': np.arange(0, joint_num), 'left': np.arange(joint_num, joint_num * 2)}
        # joint_coord[joint_type['left'], 2] += rel_root_depth
        result = joint_coord.copy()
        # if image_coordinate:

        # restore joint coord to original image space and continuous depth space
        joint_coord[:, 0] = joint_coord[:, 0] / output_hm_shape[2] * input_image_shape[1]
        joint_coord[:, 1] = joint_coord[:, 1] / output_hm_shape[1] * input_image_shape[0]
        joint_coord[:, :2] = np.dot(inv_trans, np.concatenate((joint_coord[:, :2], np.ones_like(joint_coord[:, :1])), 1).transpose((1, 0))).transpose((1, 0))
        joint_coord[:, 2] = (joint_coord[:, 2] / output_hm_shape[0] * 2 - 1) * (bbox_3d_size / 2)

        # restore right hand-relative left hand depth to continuous depth space
        rel_root_depth = (rel_root_depth / output_root_hm_shape * 2 - 1) * (bbox_3d_size_root / 2)

        # right hand root depth == 0, left hand root depth == rel_root_depth
        joint_coord[joint_type['left'], 2] += rel_root_depth
        # return joint_coord, rel_root_depth, score ,out['heatmap']#, result / 64
        return joint_coord, rel_root_depth, score, result / 64,out['heatmap']
        # else:
        #     return joint_coord / 64, rel_root_depth, score


def init_weights(m):
    if type(m) == nn.Conv2DTranspose:
        m.weight = m.create_parameter(m.weight.shape,m._param_attr, m.weight.dtype, nn.initializer.Normal(std=0.001))
    elif type(m) == nn.Conv2D:
        m.weight = m.create_parameter(m.weight.shape, m._param_attr,m.weight.dtype, nn.initializer.Normal(std=0.001))
        m.bias = m.create_parameter(attr= m._bias_attr,shape=m.bias.shape, default_initializer=nn.initializer.Constant(value=0.0),is_bias=True)
    elif type(m) == nn.BatchNorm2D:
        m.weight = m.create_parameter(m.weight.shape,m._weight_attr, m.weight.dtype, nn.initializer.Constant(1.0))
        m.bias = m.create_parameter(shape=m.bias.shape, default_initializer=nn.initializer.Constant(value=0.0),is_bias=True)
    elif type(m) == nn.Linear:
        m.weight = m.create_parameter(shape=m.weight.shape,attr=m._weight_attr, dtype=m.weight.dtype,default_initializer= nn.initializer.Normal(std=0.01))
        m.bias = m.create_parameter(shape=m.bias.shape,attr=m._bias_attr, default_initializer=nn.initializer.Constant(value=0.0),is_bias=True)

def get_model(mode, joint_num):
    backbone_net = BackboneNet()
    pose_net = PoseNet(joint_num)

    if mode == 'train':
        backbone_net.init_weights()
        pose_net.apply(init_weights)

    model = Model(backbone_net, pose_net)
    return model



if __name__=="__main__":
    model=get_model('train',21)
