import os

from mmcv.utils import print_log
from torch.utils.data import Dataset

from depth.datasets.builder import DATASETS
from depth.datasets.pipelines import Compose
from depth.utils import get_root_logger


@DATASETS.register_module()
class MD2EDepthDataset(Dataset):
    def __init__(self,
                 pipeline,
                 data_root,
                 img_dir,
                 depth_dir,
                 test_mode=True,
                 min_depth=1e-3,
                 max_depth=10,
                 depth_scale=1):
        self.pipeline = Compose(pipeline)
        self.img_path = os.path.join(data_root, img_dir)
        self.depth_path = os.path.join(data_root, depth_dir)
        self.test_mode = test_mode
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_scale = depth_scale
        self.img_infos = self.load_annotations(self.img_path, self.depth_path)

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, img_dir, depth_dir):
        img_infos = []
        images = sorted(os.listdir(img_dir))

        if not self.test_mode:
            targets = sorted(os.listdir(depth_dir))
            if len(images) != len(targets):
                raise RuntimeError(
                    'MD2E image and target directories contain different '
                    f'file counts: {len(images)} vs {len(targets)}.')
            for image_name, target_name in zip(images, targets):
                img_infos.append(
                    dict(
                        filename=image_name,
                        ann=dict(depth_map=target_name)))
        else:
            for image_name in images:
                img_infos.append(dict(filename=image_name))

        img_infos = sorted(img_infos, key=lambda x: x['filename'])
        print_log(f'Loaded {len(img_infos)} images.', logger=get_root_logger())
        return img_infos

    def pre_pipeline(self, results):
        results['depth_fields'] = []
        results['img_prefix'] = self.img_path
        results['depth_prefix'] = self.depth_path
        results['depth_scale'] = self.depth_scale

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_ann_info(self, idx):
        return self.img_infos[idx]['ann']

    def format_results(self, results, imgfile_prefix=None, indices=None, **kwargs):
        results[0] = results[0] * self.depth_scale
        return results

    def pre_eval(self, preds, indices):
        pass

    def evaluate(self, results, metric='eigen', logger=None, **kwargs):
        pass
