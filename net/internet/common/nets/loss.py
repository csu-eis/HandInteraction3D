import numpy as np

from net.internet import config
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class TwistLoss(nn.Layer):
    def __init__(self):
        super(TwistLoss, self).__init__()
    def getLoss(self,fingers_joint,batch_size):
        V01 = fingers_joint[:, :, 0, :] - fingers_joint[:,:, 1, :]
        V12 = fingers_joint[:,:, 1, :] - fingers_joint[:,:, 2, :]
        V23 = fingers_joint[:,:, 2, :] - fingers_joint[:,:, 3, :]
        tmp1=paddle.cross(V01, V12)
        C1 = paddle.sum(paddle.multiply(tmp1, V23),axis=2)
        # C2 = paddle.sum(paddle.multiply(tmp1, paddle.cross(V12, V23)),axis=2)
        loss=paddle.abs(C1) #- paddle.minimum(C2, paddle.tensor.zeros(shape=[batch_size,5]))
        a=paddle.expand(loss.reshape([batch_size,5,1]),shape=[batch_size,5,4]).reshape([batch_size,20])
        return paddle.concat([a,paddle.tensor.zeros([batch_size,1],dtype=paddle.float32)],axis=1)

        # return paddle.abs(C1) #- paddle.minimum(C2, paddle.tensor.zeros(shape=[batch_size,5]))

    def forward(self, joint_heatmap_out,joint_valid):
        # have the max value in dim z
        val_z, idx_z = paddle.max(joint_heatmap_out, axis=2), paddle.argmax(joint_heatmap_out, axis=2)
       # have the max value in dim y
        val_zy, idx_zy = paddle.max(val_z, axis=2), paddle.argmax(val_z, axis=2)
        # have the max value in dim x
        val_zyx, joint_x = paddle.max(val_zy, axis=2), paddle.argmax(val_zy, axis=2)
        batch_size = joint_heatmap_out.shape[0]
        num_joints = joint_heatmap_out.shape[1]
        index_x = paddle.squeeze(joint_x)
        joint_x = paddle.unsqueeze(joint_x, axis=-1)
        shape = (-1, 1) if batch_size > 1 else (-1,)
        idx = paddle.concat((
            paddle.arange(0, batch_size).reshape(shape).expand_as(index_x).reshape((-1, 1)),
            paddle.arange(0, num_joints).expand_as(index_x).reshape((-1, 1)),
            index_x.reshape((-1, 1))
        ),
            axis=1)
        joint_y = paddle.gather_nd(idx_zy, idx).reshape((batch_size, num_joints, 1))
        index_y = paddle.squeeze(joint_y)

        idx = paddle.concat((
            paddle.arange(0, batch_size).reshape(shape).expand_as(index_x).reshape((-1, 1)),
            paddle.arange(0, num_joints).expand_as(index_x).reshape((-1, 1)),
            index_y.reshape((-1, 1)),
            index_x.reshape((-1, 1))
        ),
            axis=1)
        joint_z = paddle.gather_nd(idx_z, idx).reshape((batch_size, num_joints, 1))
        joint_coord_out = paddle.multiply(paddle.concat((joint_x, joint_y, joint_z), 2).astype(paddle.float32) ,
                                          paddle.expand(joint_valid.reshape([batch_size,42,1]),[batch_size,42,3]))
        hands_joint = joint_coord_out.reshape([batch_size, -1, 21, 3])
        right_finger_joint=hands_joint[:,0,0:-1,:].reshape([batch_size,-1,4,3])
        left_finger_joint = hands_joint[:, 1, 0:-1, :].reshape([batch_size,-1, 4, 3])
        loss1 = self.getLoss(right_finger_joint, batch_size)
        loss2 = self.getLoss(left_finger_joint, batch_size)
        loss =paddle.concat([loss1,loss2],axis=1).reshape([batch_size,42,1])
        loss=paddle.expand(loss,shape=[batch_size,42,64]).reshape([batch_size,42,64,1])
        loss = paddle.expand(loss, shape=[batch_size, 42, 64,64]).reshape([batch_size, 42, 64,64, 1])
        loss = paddle.expand(loss, shape=[batch_size, 42, 64, 64,64])
        return loss


class Joint2DLoss(nn.Layer):
    def __init__(self):
        super(Joint2DLoss, self).__init__()

    def forward(self, joint_heatmap_out, joint_gt, joint_valid):
        # have the max value in dim z
        val_z, idx_z = paddle.max(joint_heatmap_out, axis=2), paddle.argmax(joint_heatmap_out, axis=2)
        # have the max value in dim y
        val_zy, idx_zy = paddle.max(val_z, axis=2), paddle.argmax(val_z, axis=2)
        # have the max value in dim x
        val_zyx, joint_x = paddle.max(val_zy, axis=2), paddle.argmax(val_zy, axis=2)
        batch_size = joint_heatmap_out.shape[0]
        num_joints = joint_heatmap_out.shape[1]
        index_x = paddle.squeeze(joint_x)
        joint_x = paddle.unsqueeze(joint_x, axis=-1)
        shape = (-1, 1) if batch_size > 1 else (-1,)
        idx = paddle.concat((
            paddle.arange(0, batch_size).reshape(shape).expand_as(index_x).reshape((-1, 1)),
            paddle.arange(0, num_joints).expand_as(index_x).reshape((-1, 1)),
            index_x.reshape((-1, 1))
        ),
            axis=1)
        joint_y = paddle.gather_nd(idx_zy, idx).reshape((batch_size, num_joints, 1))
        index_y = paddle.squeeze(joint_y)

        idx = paddle.concat((
            paddle.arange(0, batch_size).reshape(shape).expand_as(index_x).reshape((-1, 1)),
            paddle.arange(0, num_joints).expand_as(index_x).reshape((-1, 1)),
            index_y.reshape((-1, 1)),
            index_x.reshape((-1, 1))
        ),
            axis=1)
        joint_z = paddle.gather_nd(idx_z, idx).reshape((batch_size, num_joints, 1))
        joint_coord_out = paddle.multiply(paddle.concat((joint_x, joint_y, joint_z), 2).astype(paddle.float32),
                                          paddle.expand(joint_valid.reshape([batch_size, 42, 1]), [batch_size, 42, 3]))
        loss = paddle.multiply((joint_coord_out - joint_gt) ** 2 , paddle.expand(paddle.unsqueeze(joint_valid, axis=[-1]),shape=[batch_size,42,3]))
        return loss


class JointHeatmapLoss(nn.Layer):
    def __init__(self):
        super(JointHeatmapLoss, self).__init__()

    def forward(self, joint_out, joint_gt, joint_valid):
        # loss = (joint_out - joint_gt) ** 2 * joint_valid[:, :, None, None, None]
        loss = (joint_out - joint_gt) ** 2 * paddle.unsqueeze(joint_valid, axis=[-1, -1, -1])
        return loss


class HandTypeLoss(nn.Layer):
    def __init__(self):
        super(HandTypeLoss, self).__init__()

    def forward(self, hand_type_out, hand_type_gt, hand_type_valid):
        loss = F.binary_cross_entropy(hand_type_out, hand_type_gt, reduction='none')
        loss = loss.mean(1)
        loss = loss * hand_type_valid

        return loss


class RelRootDepthLoss(nn.Layer):
    def __init__(self):
        super(RelRootDepthLoss, self).__init__()

    def forward(self, root_depth_out, root_depth_gt, root_valid):
        loss = paddle.abs(root_depth_out - root_depth_gt) * root_valid
        return loss
