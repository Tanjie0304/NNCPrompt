import os
from os.path import join, exists, realpath
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from typing import Union
from copy import deepcopy
from functools import partial
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config


class ImagePathDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.class_imgpath_dict: dict[int, str] = {}
        self.class_int_str_map: dict[int, str] = {}
        self.split_path = ""

    def __getitem__(self, index):
        return self.class_imgpath_dict[index]

    def __len__(self):
        return len(self.class_imgpath_dict)

    @property
    def class_list(self):
        return sorted(list(self.class_imgpath_dict.keys()))

    @property
    def num_classes(self):
        return len(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: image_dir=\"{self.split_path}\", dataset_classes={len(self)}"


class CIFAR100Path(ImagePathDataset):
    def __init__(self, root_dir: str, train: bool):
        super().__init__()
        cls_dir_list = sorted(os.listdir(split_path := join(root_dir, 'train' if train else 'val')))
        assert exists(split_path), split_path
        self.split_path = split_path

        for cls_dir in cls_dir_list:
            cls_int = int(cls_dir)
            assert not cls_int in self.class_imgpath_dict
            cls_path = realpath(join(split_path, cls_dir))
            self.class_imgpath_dict[cls_int] = [join(cls_path, img_file) for img_file in os.listdir(cls_path)]
            self.class_int_str_map[cls_int] = cls_dir

        assert len(self.class_imgpath_dict) == 100


class DomainNetPath(ImagePathDataset):
    def __init__(self, root_dir: str, train: bool):
        super().__init__()
        cls_dir_list = sorted(os.listdir(split_path := join(root_dir, 'train' if train else 'val')))
        assert exists(split_path), split_path
        self.split_path = split_path

        for cls_dir in cls_dir_list:
            cls_int = int(cls_dir)
            assert not cls_int in self.class_imgpath_dict
            cls_path = realpath(join(split_path, cls_dir))
            self.class_imgpath_dict[cls_int] = [join(cls_path, img_file) for img_file in os.listdir(cls_path)]
            self.class_int_str_map[cls_int] = cls_dir

        assert len(self.class_imgpath_dict) == 200


class ImageNetRPath(ImagePathDataset):
    def __init__(self, root_dir: str, train: bool):
        super().__init__()
        cls_dir_list = sorted(os.listdir(split_path := join(root_dir, 'train' if train else 'val')))
        assert exists(split_path), split_path
        self.split_path = split_path

        for cls_int, cls_dir in enumerate(cls_dir_list):
            assert not cls_int in self.class_imgpath_dict
            cls_path = realpath(join(split_path, cls_dir))
            self.class_imgpath_dict[cls_int] = [join(cls_path, img_file) for img_file in os.listdir(cls_path)]
            self.class_int_str_map[cls_int] = cls_dir

        assert len(self.class_imgpath_dict) == 200


class ImagePathDatasetClassManager():
    def __init__(self, **kwargs):
        self.dataset_dict = {
            'imagenet_r': partial(ImageNetRPath, root_dir="./datasets/data.ImageNet-R" if not (v := kwargs.get('imagenet_r')) else v),
            'cifar100': partial(CIFAR100Path, root_dir="./datasets/data.CIFAR100" if not (v := kwargs.get('cifar100')) else v),
            'domainnet': partial(DomainNetPath, root_dir="./datasets/data.DomainNet" if not (v := kwargs.get('domainnet')) else v),
        }

    def __getitem__(self, dataset: str):
        dataset = dataset.lower()
        if dataset not in (_valid_names := self.dataset_dict.keys()):
            raise NameError(f"{dataset} is not in {_valid_names}")
        return self.dataset_dict[dataset]


class ClassIncremantalDataset(Dataset):
    def __init__(self, path_dataset: ImagePathDataset, task_class_list: list[int], transforms: T.Compose = None, expand_times: int = 1, label_map_g2l: dict[int, int] = None, verbose: bool = False, return_index: bool = False):
        super().__init__()
        self.path_dataset = path_dataset
        self.task_class_list = tuple(deepcopy(task_class_list))
        assert isinstance(expand_times, int) and expand_times >= 1
        self.expand_times = expand_times
        self.transforms = transforms
        self.label_map_g2l = deepcopy(label_map_g2l)
        self.label_map_l2g = self.make_label_map_l2g(self.label_map_g2l)
        self.__sample_type = 'image'

        self.samples, self.labels = self.get_all_samples(sample_type=self.__sample_type)
        self.verbose = verbose
        self.return_index = return_index
        self.num_samples = len(self.labels)

        self.cache_dict = {}

    @staticmethod
    def make_label_map_l2g(label_map_g2l: dict[int, int]) -> dict[int, int]:
        if label_map_g2l:
            return {v: k for k, v in label_map_g2l.items()}

    def get_all_samples(self, sample_type: str = 'path') -> tuple[list[Image.Image | str], list[int]]:
        assert sample_type in ('path', 'image'), f"{sample_type}"
        smp_list = []
        lbl_list = []
        for cls_int in self.task_class_list:
            assert cls_int in self.path_dataset.class_list
            assert len(self.path_dataset[cls_int]) > 0
            for img_path in self.path_dataset[cls_int]:
                if sample_type == 'image':
                    sample: Image.Image = Image.open(img_path).convert('RGB')
                elif sample_type == 'path':
                    sample: str = img_path
                smp_list.append(sample)
                lbl_list.append(self.label_map_g2l[cls_int] if self.label_map_g2l else cls_int)
        return smp_list, lbl_list

    def read_one_image_label(self, index: int) -> tuple[Image.Image, int]:
        if self.__sample_type == 'path':
            if self.expand_times == 1:
                img = Image.open(self.samples[index]).convert('RGB')
            elif self.expand_times > 1:
                img = Image.open(self.samples[index]).convert('RGB')
            else:
                raise ValueError(f"{self.expand_times}")

            lbl = self.labels[index]
            return img, lbl
        elif self.__sample_type == 'image':
            return self.samples[index], self.labels[index]
        else:
            raise NameError(f"{self.__sample_type}")

    def __getitem__(self, index: int) -> tuple[Union[Image.Image, Tensor], int]:
        index %= self.num_samples
        img, lbl = self.read_one_image_label(index)

        if self.transforms is not None:
            if self.return_index:
                return self.transforms(img), lbl, index
            else:
                return self.transforms(img), lbl
        if self.return_index:
            return img, lbl, index
        else:
            return img, lbl

    def __len__(self):
        return self.num_samples * self.expand_times

    def __repr__(self) -> str:
        _repr = f"{self.__class__.__name__} for {self.path_dataset.__repr__()}: task_class_list({len(self.task_class_list)})={self.task_class_list}, num_samples={len(self.samples)}, expand_times={self.expand_times}, use_label_map={bool(self.label_map_g2l)}"
        if self.verbose:
            _repr += f", \nclasses_string=\n{[self.path_dataset.class_int_str_map[cls_int] for cls_int in self.task_class_list]}"
            _repr += f", \ntransform={self.transforms}"
            _repr += f", \nlabel_map_g2l={self.label_map_g2l}"
            _repr += f", \nlabel_map_l2g={self.label_map_l2g}"
        return _repr


def define_dataset(GVM, task_classes: list[int], training: bool, transform_type: str = 'autoaug', use_eval_transform: bool = False, use_label_map: bool = False, expand_times: int = 1, **kwargs) -> ClassIncremantalDataset:
    '''
    transform_type: 'timm', 'autoaug', 'prototype'
    '''
    _current_dataset = GVM.args.dataset
    match transform_type:
        case 'timm':
            if not 'model' in kwargs:
                raise NameError("Argument 'model' is need if the transform is created from timm.")
            transform = create_transform(**resolve_data_config(kwargs['model'].pretrained_cfg, model=kwargs['model']), is_training=training if not use_eval_transform else False)
        case 'autoaug':
            if training and not use_eval_transform:
                match _current_dataset:
                    case 'cifar100':
                        transform = T.Compose([T.AutoAugment(T.AutoAugmentPolicy.CIFAR10, T.InterpolationMode.BILINEAR), T.RandomResizedCrop((224, 224), antialias=True), T.ToTensor(), T.Normalize(0.5, 0.5)])
                    case 'imagenet_r':
                        transform = T.Compose([T.AutoAugment(T.AutoAugmentPolicy.IMAGENET, T.InterpolationMode.BILINEAR), T.RandomResizedCrop((224, 224), antialias=True), T.ToTensor(), T.Normalize(0.5, 0.5)])
                    case 'domainnet':
                        transform = T.Compose([T.AutoAugment(T.AutoAugmentPolicy.IMAGENET, T.InterpolationMode.BILINEAR), T.RandomResizedCrop((224, 224), antialias=True), T.ToTensor(), T.Normalize(0.5, 0.5)])
                    case _:
                        raise NotImplementedError(_current_dataset)
            else:
                match _current_dataset:
                    case 'cifar100':
                        transform = T.Compose([T.Resize((224, 224), antialias=True), T.ToTensor(), T.Normalize(0.5, 0.5)])
                    case 'imagenet_r':
                        transform = T.Compose([T.Resize((256, 256), antialias=True), T.CenterCrop(224), T.ToTensor(), T.Normalize(0.5, 0.5)])
                    case 'domainnet':
                        transform = T.Compose([T.Resize((256, 256), antialias=True), T.CenterCrop(224), T.ToTensor(), T.Normalize(0.5, 0.5)])
        case 'prototype':
            assert not training or use_eval_transform, "Only used for extracting prototypes"
            match _current_dataset:
                case 'cifar100':
                    transform = T.Compose([T.ToTensor(), T.Resize((224, 224), antialias=True)])
                case 'imagenet_r':
                    transform = T.Compose([T.Resize((256, 256), antialias=True), T.CenterCrop((224, 224)), T.ToTensor()])
                case 'domainnet':
                    transform = T.Compose([T.Resize((256, 256), antialias=True), T.CenterCrop((224, 224)), T.ToTensor()])
                case _:
                    raise NotImplementedError(_current_dataset)
        case _:
            raise NotImplementedError(f"{transform_type}")

    _mode = 'train' if training else 'eval'
    dataset = ClassIncremantalDataset(GVM.path_data_dict[_mode], task_classes, transform, expand_times=expand_times, label_map_g2l=GVM.label_map_g2l if use_label_map else None, **kwargs)

    return dataset
