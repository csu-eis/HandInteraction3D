import paddle
import paddle.vision.transforms as transforms
import os
import os.path as osp
import glob
from net.internet.common.dataset import InterhandsDataset
from net.internet.config import cfg
# from net.internet.model import get_model
from net.dualhand.model import get_model
import abc
from net.internet.common.timer import Timer
from net.internet.common.logger import colorlogger
import math


class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        self.cur_epoch = 0

        # timer.py
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):

    def __init__(self):
        super(Trainer, self).__init__(log_name='train_logs.txt')

    def get_optimizer(self, model):
        optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=cfg.lr)
        return optimizer

    def set_lr(self, epoch):
        if len(cfg.lr_dec_epoch) == 0:
            return cfg.lr

        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            # learning rate decrease index
            idx = cfg.lr_dec_epoch.index(e)
            # for g in self.optimizer._parameter_list:
            self.optimizer.set_lr( cfg.lr / (cfg.lr_dec_factor ** idx))
        else:
            # for g in self.optimizer.param_groups:
            self.optimizer.set_lr(cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch)))

    def get_lr(self):
        # for g in self.optimizer.param_groups:
        #     cur_lr = g['lr']

        return self.optimizer.get_lr()

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating train dataset...")
        trainset_loader =InterhandsDataset(transforms.ToTensor(),'train',0.05)
        batch_generator = paddle.io.DataLoader(dataset=trainset_loader, batch_size=cfg.train_batch_size,
                                     shuffle=True, num_workers=cfg.num_thread)

        self.joint_num = trainset_loader.joint_num
        self.itr_per_epoch = math.ceil(trainset_loader.__len__() / cfg.train_batch_size)
        self.batch_generator = batch_generator

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model('train', self.joint_num, init_weight=True)
        # model = model.cuda()
        # model = paddle.DataParallel(model)
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer

    # TODO
    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir, 'snapshot_{}.pdparams'.format(str(epoch)))

        paddle.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    # TODO
    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir, '*.pdparams'))

        if len(model_file_list) > 0:
            cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9: file_name.find('.pdparams')]) for file_name in
                             model_file_list])
            model_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pdparams')
            self.logger.info('Load checkpoint from {}'.format(model_path))

            ckpt = paddle.load(model_path)
            model.load_dict(ckpt['network'])
            try:
                optimizer.set_state_dict(ckpt['optimizer'])
            except:
                pass
        else:
            cur_epoch = 0
        start_epoch = cur_epoch + 1

        return start_epoch, model, optimizer


class Tester(Base):

    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(log_name='test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        # self.logger.info("Creating " + test_set + " dataset...")


        testset_loader =  InterhandsDataset(transforms.ToTensor(), 'test')
        batch_generator =  paddle.io.DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus * cfg.test_batch_size,
                                               shuffle=True, num_workers=cfg.num_thread)

        self.joint_num = testset_loader.joint_num
        self.batch_generator = batch_generator
        self.testset = testset_loader

    def _make_model(self):
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pdparams' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))

        # prepare network
        self.logger.info("Creating graph...")
        model = get_model('test', self.joint_num)
        ckpt = paddle.load(model_path)

        if 'network' in ckpt:
            model.load_dict(ckpt['network'])
        else:
            model.load_dict(ckpt)
        model.eval()

        self.model = model

    def _evaluate(self, preds):
        self.testset.evaluate(preds)
