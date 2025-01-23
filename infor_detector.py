import torch

from utils import *
from config import get_config

from CenterNet.src.lib.utils.image import get_affine_transform
from CenterNet.src.lib.models.utils import flip_tensor
from CenterNet.src.lib.models.decode import ctdet_decode
from CenterNet.src.lib.utils.post_process import ctdet_post_process
from CenterNet.src.lib.utils.debugger import Debugger
from CenterNet.src.lib.models.networks.msra_resnet import get_pose_net

model_factory = {
  'res': get_pose_net
}

class INFOR_DETECTOR(object):
    def __init__(self, name_config):
        # print('Creating model...')
        self.config = get_config(name_config)
        self.model = self.create_model(self.config['arch'], self.config['heads'], self.config['head_conv'])
        self.model = self.load_model(self.model, self.config['load_model'])
        self.model = self.model.to(self.config['device'])
        self.model.eval()

        self.mean = np.array(self.config['mean'],  dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(self.config['std'], dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = self.config['num_classes']
        self.scales = self.config['test_scales']
        self.pause = True
        self.label  = self.config['label']

    def create_model(self, arch, heads, head_conv):
        num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
        arch = arch[:arch.find('_')] if '_' in arch else arch
        get_model = model_factory[arch]
        model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
        return model

    def load_model(self, model, model_path, optimizer=None, resume=False, lr=None, lr_step=None):
        start_epoch = 0
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        # print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        state_dict = {}

        # convert data_parallal to model
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()

        # check loaded parameters and created model parameters
        msg = 'If you see this, your model does not fully load the ' + \
              'pre-trained weight. Please make sure ' + \
              'you have correctly specified --arch xxx ' + \
              'or set the correct --num_classes for your own dataset.'
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, ' \
                          'loaded shape{}. {}'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape, msg))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k) + msg)
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k) + msg)
                state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)

        # resume optimizer parameters
        if optimizer is not None and resume:
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch']
                start_lr = lr
                for step in lr_step:
                    if start_epoch >= step:
                        start_lr *= 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = start_lr
                print('Resumed optimizer with start lr', start_lr)
            else:
                print('No optimizer parameters in checkpoint.')
        if optimizer is not None:
            return model, optimizer, start_epoch
        else:
            return model

    def pre_process(self, image, scale, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        if self.config['fix_res']:
            inp_height, inp_width = self.config['input_h'], self.config['input_w']
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            inp_height = (new_height | self.config['pad']) + 1
            inp_width = (new_width | self.config['pad']) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        if self.config['flip_test']:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.config['down_ratio'],
                'out_width': inp_width // self.config['down_ratio']}
        return images, meta

    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model(images)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.config['reg_offset'] else None
            if self.config['flip_test']:
                hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
                wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
                reg = reg[0:1] if reg is not None else None
            # torch.cuda.synchronize()
            dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.config['cat_spec_wh'], K=self.config['K'])

        if return_time:
            return output, dets
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.config['num_classes'])
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            # if len(self.scales) > 1 or self.config['nms']:
            #     soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        detection[:, :, :4] *= self.config['down_ratio']
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            for k in range(len(dets[i])):
                if detection[i, k, 4] > self.config['center_thresh']:
                    debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                           detection[i, k, 4],
                                           img_id='out_pred_{:.1f}'.format(scale))

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='infor')
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] > self.config['vis_thresh']:
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='infor')
        # print(debugger)
        debugger.show_all_imgs(pause=self.pause)

    def run(self, image_or_path_or_tensor, meta=None):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        debugger = Debugger(dataset=self.config['dataset'], ipynb=(self.config['debug'] == 3),
                            theme=self.config['debugger_theme'])

        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            image = cv2.imread(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        detections = []
        for scale in self.scales:
            if not pre_processed:
                images, meta = self.pre_process(image, scale, meta)
            else:
                # import pdb; pdb.set_trace()
                images = pre_processed_images['images'][scale][0]
                meta = pre_processed_images['meta'][scale]
                meta = {k: v.numpy()[0] for k, v in meta.items()}
            # images = images.to(self.opt.device)
            # torch.cuda.synchronize()

            output, dets = self.process(images, return_time=True)

            # torch.cuda.synchronize()

            dets = self.post_process(dets, meta, scale)
            # torch.cuda.synchronize()

            detections.append(dets)

        results = self.merge_outputs(detections)

        results = {'results': results}

        # get box infor
        dict_box_all = dict()
        for key in results['results'].keys():
            boxes = results['results'][key]
            list_box = get_box(boxes, 0.3)
            if len(list_box) != 0:
                list_box_sort = sort_box(list_box)
                dict_box_all[self.label[key - 1]] = list_box_sort
            else:
                dict_box_all[self.label[key - 1]] = list_box

        return dict_box_all

if __name__ == "__main__":
    file_image = 'output_b2.jpg'
    image = cv2.imread(file_image)
    dict_box_all  = INFOR_DETECTOR('infor_chip').run(image)
    print(dict_box_all)

