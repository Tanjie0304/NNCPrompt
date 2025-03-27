import os
from time import time as ttime
import argparse
import random
from collections import OrderedDict
import warnings
import tqdm
import itertools
from copy import deepcopy
from functools import partial
os.environ['TIMM_FUSED_ATTN'] = '1'
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
import torchvision
warnings.filterwarnings('ignore')
import timm
from timm import create_model
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from utils.dataset_builder import ImagePathDatasetClassManager, ImagePathDataset, define_dataset
from utils.continual_manager import ClassIncrementalManager
from utils import misc
from utils.vit_builder import VisionTransformer


class GlobalVarsManager:
    args: argparse.Namespace
    path_data_dict: dict[str, ImagePathDataset]
    cl_mngr: ClassIncrementalManager
    acc_mat_dict: OrderedDict[str, np.ndarray]
    forgetting_dict: OrderedDict[str, np.ndarray]
    cache_dict: dict
    param_dict: dict[str, OrderedDict[str, Tensor]]
    label_map_g2l: dict[int, int]
    label_map_l2g: dict[str, int]

    def update_label_maps(self, taskid: int, task_classes: list[int]) -> tuple[dict[int, int], dict[str, int]]:
        _g2l_map, _l2g_map = misc.make_label_maps(taskid, task_classes)
        assert all([_k not in self.label_map_g2l.keys() for _k in _g2l_map.keys()])
        assert all([_k not in self.label_map_l2g.keys() for _k in _l2g_map.keys()])
        self.label_map_g2l.update(_g2l_map)
        self.label_map_l2g.update(_l2g_map)
        return _g2l_map, _l2g_map

    def cache_prototypes(self, name_prototype: tuple[str, Tensor], name_label: tuple[str, Tensor], taskid_as_key: int = None):
        assert len(name_prototype) == len(name_label) == 2
        assert name_prototype[0] != name_label[0]

        for _name, _array in (name_prototype, name_label, ):
            if taskid_as_key is None:
                if _name not in self.cache_dict:
                    self.cache_dict[_name] = _array.clone()
                else:
                    self.cache_dict[_name] = torch.cat([self.cache_dict[_name].cpu(), _array.cpu().clone()])
            elif isinstance(taskid, int):
                if _name not in self.cache_dict:
                    self.cache_dict[_name] = {}
                assert taskid_as_key not in self.cache_dict[_name]
                self.cache_dict[_name][taskid_as_key] = _array.cpu().clone()
            else:
                raise TypeError(f"type ({type(taskid_as_key)}) should be int")

        _info = ""
        return _info


GVM = GlobalVarsManager()


def getargs():
    parser = argparse.ArgumentParser(description='Class-incremental Learning')
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=('cifar100', 'imagenet_r', 'domainnet'), help='use lowercase')
    parser.add_argument('-dr', '--data_root', type=str, default="")
    parser.add_argument('-t', '--num_tasks', type=int, default=10, choices=(1, 2, 5, 10, 20, 25, 50, 100))
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('-dv', '--device', type=int, default=0, help='Training device')
    parser.add_argument('-mo', '--model', type=str, choices=timm.list_models(), default='vit_base_patch16_224.augreg_in21k')
    parser.add_argument('--prompt_len', type=int, default=8, help='0 means not using prompt')
    parser.add_argument('--prompt_init', type=str, choices=('uniform', 'zero'), default='uniform')
    parser.add_argument('--prompt_start_block', type=int, default=0)
    parser.add_argument('--prompt_end_block', type=int, default=8)
    parser.add_argument('--stage1_prompt_len', type=int, default=8)
    parser.add_argument('--stage1_prompt_layer', type=int, default=6)
    parser.add_argument('--seperate_head', type=misc.str2bool, default=True)
    parser.add_argument('-rep', '--representation', type=str, default='cls_token', choices=('cls_token', 'embed_mean'))
    parser.add_argument('--use_feat_norm', type=misc.str2bool, default=True)
    parser.add_argument('--num_nn_idx', type=int, default=1)
    parser.add_argument('--num_cls_limit', type=int, default=1)
    parser.add_argument('-jt', '--workers', type=int, default=4)
    parser.add_argument('-je', '--eval_workers', type=int, default=0)
    parser.add_argument('-et', '--expand_times', type=int, default=10)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('--simu_logits', type=misc.str2bool, default=True)
    parser.add_argument('--alpha', type=int, default=10)
    parser.add_argument('--beta', type=int, default=95)
    parser.add_argument('-m', '--momentum', type=float, default=1.0)
    parser.add_argument('-b', '--batch_size', type=int, default=150)
    parser.add_argument('--temperature', type=float, default=28)
    parser.add_argument('--use_amp', type=misc.str2bool, default=True)
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--lr_sch', type=str, default='multistep', choices=('cosine', 'step', 'multistep'))
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--decay_milestones', type=int, nargs='+', default=[5, 8])
    parser.add_argument('--decay_epochs', type=int, default=1000)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--show_bar', action='store_true')

    args = parser.parse_args()

    assert args.num_cls_limit <= args.num_nn_idx

    return args


def seed_etc_options(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.set_printoptions(precision=4, linewidth=256)
    torch.set_printoptions(linewidth=256)
    torchvision.set_image_backend('accimage')


def set_model_mode(model: VisionTransformer, training: bool, to_gpu: bool = True, verbose: bool = False, basemode: bool = False) -> VisionTransformer:
    for n, p in model.named_parameters():
        if training and (n.startswith(('prompt', 'head')) or 'prompt' in n):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    for n, m in model.named_children():
        if training and (n in ('prompt', 'head')):
            m.train()
        else:
            m.eval()

    if to_gpu:
        model.cuda(device=args.device)

    return model


def train_one_epoch(dataloader: DataLoader, model: VisionTransformer, criterion: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer, use_amp: bool = True, temperature: float = 1., **kwargs) -> str:
    assert temperature > 0.
    amp_scalar = GradScaler(enabled=use_amp)
    scalar_meter = misc.ScalarMeter(lr="step_last:.3e", loss="samp_avg:.4f", data_time="step_sum:.3f", batch_time="step_sum:.3f", acc_top1="samp_avg:>6.2%", acc_top5="samp_avg:>6.2%")
    _btimer = ttime()
    for i_batch, (images, target) in tqdm.tqdm(enumerate(dataloader, 1), total=len(dataloader), dynamic_ncols=True, disable=not GVM.args.show_bar):
        data_time = ttime() - _btimer

        images: Tensor = images.cuda(non_blocking=True, device=args.device)
        target: Tensor = target.cuda(non_blocking=True, device=args.device)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            curr_logits: Tensor = model(images)

        output = []

        if GVM.args.simu_logits:
            fake_hard_sample_logit = torch.rand_like(curr_logits[:, :1]) * GVM.args.alpha + GVM.args.beta  # [B, 1]
            output = torch.cat([curr_logits, fake_hard_sample_logit], dim=1)
        else:
            output = curr_logits

        loss: Tensor = criterion(output / temperature, target)

        optimizer.zero_grad()
        amp_scalar.scale(loss).backward()
        amp_scalar.step(optimizer)
        amp_scalar.update()

        acc_top1, acc_top5 = misc.calc_accuracy(output, target, topk=(1, 2))
        batch_time = ttime() - _btimer

        scalar_meter.add_step_value(len(images), lr=optimizer.param_groups[0]['lr'], loss=loss.item(), data_time=data_time, batch_time=batch_time, acc_top1=acc_top1, acc_top5=acc_top5)
        _btimer = ttime()

    _epoch_scalar_str = scalar_meter.format_outout(scalar_meter.update_epoch_average_value())
    return _epoch_scalar_str


def cache_state(taskid: int, model: VisionTransformer):
    if taskid == 0:
        base_params = OrderedDict()
    task_params = OrderedDict()

    for n, p in model.named_parameters():
        if p.requires_grad:
            task_params[n] = p.clone()
        else:
            if taskid == 0:
                base_params[n] = p.clone()
            else:
                assert torch.all(GVM.param_dict[f'base_params'][n] == p.to(GVM.param_dict[f'base_params'][n].device))

    if taskid == 0:
        assert not 'base_params' in GVM.param_dict
        GVM.param_dict['base_params'] = base_params

    assert not f'task_params_{taskid}' in GVM.param_dict
    GVM.param_dict[f'task_params_{taskid}'] = task_params


def train_one_task(taskid: int, task_classes: list[int], model: VisionTransformer, basemode: bool = False, **kwargs) -> VisionTransformer:
    args = GVM.args
    if args.epochs == 0:
        return model

    if basemode:
        print(f"*" * 55 + " Start Training (Stage 1) " + "*" * 55)
    else:
        print(f"*" * 60 + " Start Training " + "*" * 60)
    _ttimer = ttime()
    _ntstr = str(GVM.cl_mngr.num_tasks)

    model: VisionTransformer = set_model_mode(model, training=True, verbose=taskid == 0, basemode=basemode)
    dataset = define_dataset(GVM, task_classes, training=True, transform_type='autoaug', use_label_map=args.seperate_head, expand_times=args.expand_times, verbose=taskid == 0)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, timeout=30, num_workers=args.workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda(device=args.device)
    param_groups = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = create_optimizer_v2(param_groups, opt=args.optimizer, lr=args.lr, weight_decay=args.weight_decay)
    if not basemode or taskid != 0:
        epochs = args.epochs
        scheduler, num_epochs = create_scheduler_v2(optimizer, sched=args.lr_sch, num_epochs=epochs, decay_epochs=args.decay_epochs, decay_milestones=args.decay_milestones,
                                                    decay_rate=args.decay_rate, min_lr=args.min_lr, warmup_epochs=args.warmup_epochs, warmup_lr=args.min_lr)
        assert num_epochs == epochs

    for epoch in range(0, args.epochs + 1):
        if epoch > 0:
            _epoch_scalar_str = train_one_epoch(dataloader, model, criterion, optimizer, args.use_amp, args.temperature)
            print(f"Task [{taskid + 1:>{len(_ntstr)}}/{_ntstr}] Epoch [{epoch:>{len(_nestr := str(args.epochs))}}/{_nestr}]:: {_epoch_scalar_str}")
            if not basemode or taskid != 0:
                scheduler.step(epoch)

    if not basemode:
        cache_state(taskid, model)

    print(f"Task [{taskid + 1:>{len(_ntstr)}}/{_ntstr}]:: Training time = {misc.format_duration(ttime() - _ttimer)}")
    return model


def _get_feature(output: Tensor, representation: str) -> Tensor:
    assert output.ndim == 3, f"{output.shape}"

    match representation:
        case 'cls_token':
            feature = F.normalize(output, dim=1)[:, 0]
        case 'embed_mean':
            feature = F.normalize(output, dim=1).mean(1)
        case _:
            raise NotImplementedError(f"{representation}")
    assert feature.ndim == 2, f"{feature.shape}"

    return feature


def extract_prototypes(taskid: int, task_classes: list[int], model: VisionTransformer, num_per_class: int = 1, representation: str = '', **kwargs) -> tuple[Tensor, Tensor]:
    print(f"*" * 57 + " Extract Prototypes " + "*" * 57)
    _ttimer = ttime()

    dataset = define_dataset(GVM, task_classes, training=True, transform_type='prototype', use_eval_transform=True, verbose=taskid == 0)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=args.eval_workers, pin_memory=True, timeout=30 if args.eval_workers > 0 else 0)

    _feature_list = []
    _label_list = []
    model = set_model_mode(model, training=False)
    for i_batch, (images, target) in tqdm.tqdm(enumerate(dataloader, 1), total=len(dataloader), dynamic_ncols=True, disable=not GVM.args.show_bar):
        images: Tensor = images.cuda(non_blocking=True, device=args.device)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=GVM.args.use_amp):
            with torch.no_grad():
                output: Tensor = model.forward_features(images)  # [B, N, D]
        feature = _get_feature(output, representation)

        _feature_list.append(feature.cpu())
        _label_list.append(target.cpu())
    _feature_list = torch.cat(_feature_list)
    _label_list = torch.cat(_label_list)

    _prototype_dict = OrderedDict()
    _label_dict = OrderedDict()
    for _c in task_classes:
        _idx_mask = _label_list == _c
        assert torch.any(_idx_mask), f"task_classes={task_classes}, _label_list={torch.unique(_label_list)}"

        _feats = _feature_list[_idx_mask]

        if num_per_class == 1:
            _prototype_dict[_c] = torch.mean(_feats, 0, keepdim=True)
        elif num_per_class > 1:
            _kmeans = KMeans(n_clusters=num_per_class, init='k-means++', n_init=10)
            _kmeans.fit(_feats.numpy())
            _prototype_dict[_c] = torch.from_numpy(_kmeans.cluster_centers_)
        else:
            raise ValueError(f"Number of prototypes ({num_per_class}) should be >=1.")

        _label_dict[_c] = torch.as_tensor(_c, dtype=torch.long).repeat(num_per_class)

    prototype_array = torch.cat(list(_prototype_dict.values())).cpu()
    prototype_array = F.normalize(prototype_array, dim=1)
    label_array = torch.cat(list(_label_dict.values())).cpu()

    print(f"prototypes_per_class={num_per_class}, prototype_array.shape={list(prototype_array.shape)}, label_array.shape={list(label_array.shape)}")
    print(f"Task [{taskid + 1:>{len(_ntstr := str(GVM.cl_mngr.num_tasks))}}/{_ntstr}]:: Extracting prototype time = {misc.format_duration(ttime() - _ttimer)}")

    assert prototype_array.ndim == 2, f"{prototype_array.shape}"
    assert len(prototype_array) == len(label_array)

    return prototype_array, label_array


def _nearest_label_idx(model: VisionTransformer, single_image: Tensor, num_nn_idx: int = 1, representation: str = '') -> Tensor:
    assert single_image.ndim == 3, f"{single_image.shape}"
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=GVM.args.use_amp):
        with torch.no_grad():
            output: Tensor = model.forward_features(single_image.unsqueeze(0))
    feature = _get_feature(output, representation)
    if GVM.args.use_feat_norm:
        feature = F.normalize(feature, dim=1)

    dist_ary = torch.square(feature - GVM.cache_dict['prototype_array']).sum(1)
    assert dist_ary.ndim == 1, f"{dist_ary.shape}"

    if num_nn_idx == 1:
        label_idx = dist_ary.argmin(0, keepdim=True)
    elif num_nn_idx > 1:
        label_idx = dist_ary.argsort(0, descending=False)[:num_nn_idx]
    else:
        raise ValueError(f"num_nn_idx={num_nn_idx}")

    assert label_idx.ndim == 1, f"{label_idx.shape}"
    return label_idx


def _count_with_order(input: Tensor) -> tuple[Tensor, Tensor]:
    assert input.ndim == 1, f"{input.shape}"
    elems, cnts = torch.unique(input, return_counts=True)
    first_appear = torch.as_tensor([torch.where(input == elem)[0].amin() for elem in elems], dtype=torch.long, device=input.device)
    _, apper_order_idx = torch.sort(first_appear)
    ordered_elems = elems[apper_order_idx]
    ordered_cnts = cnts[apper_order_idx]
    return ordered_elems, ordered_cnts


def _nearest_pred_labels(label_idx: Tensor, num_cls_limit: int = 1) -> Tensor:
    assert label_idx.ndim == 1

    num_cls_limit = min(num_cls_limit, len(label_idx))

    if num_cls_limit == 1:
        preds: Tensor = torch.as_tensor(GVM.cache_dict['label_array'][label_idx[0]]).unsqueeze(0)
    elif num_cls_limit > 1:
        label_vals: Tensor = GVM.cache_dict['label_array'][label_idx]
        assert label_vals.ndim == 1, f"{label_vals.shape}"

        unique_labels, unique_counts = _count_with_order(label_vals)
        _, count_sorted_index = torch.sort(unique_counts, descending=True, stable=True)
        count_sorted_labels = unique_labels[count_sorted_index]

        preds: Tensor = count_sorted_labels[:num_cls_limit]
    else:
        raise ValueError(f"num_cls_limit={num_cls_limit}")

    assert preds.ndim == 1, f"{preds.shape}"
    return preds


def infer_nn_preds(model: VisionTransformer, images: Tensor, num_nn_idx: int = 1, num_cls_limit: int = 1) -> list[Tensor]:
    assert images.ndim == 4, f"{images.shape}"
    _partial_nearest_label_idx = partial(_nearest_label_idx, num_nn_idx=num_nn_idx, representation=GVM.args.representation)
    label_idxs = torch.vmap(_partial_nearest_label_idx, in_dims=(None, 0))(model, images)
    preds_list = [_nearest_pred_labels(label_idx, num_cls_limit) for label_idx in label_idxs]
    return preds_list


def _label_to_taskid(global_label: int) -> int:
    if not 'label_to_taskid_dict' in GVM.cache_dict:
        GVM.cache_dict['label_to_taskid_dict'] = {}
    if global_label not in GVM.cache_dict['label_to_taskid_dict']:
        for k, v in GVM.label_map_l2g.items():
            if v == global_label:
                GVM.cache_dict['label_to_taskid_dict'][global_label] = int(k.split(':')[0])
                break
    return GVM.cache_dict['label_to_taskid_dict'][global_label]


def infer_nn_taskids(preds_array: Tensor) -> list[int]:
    taskids = []
    for pred in preds_array.tolist():
        if (_inferred_taskid := _label_to_taskid(pred)) not in taskids:
            taskids.append(_inferred_taskid)
    return taskids


def merge_head(model: VisionTransformer, task_id: int, target_classes: list[int]) -> VisionTransformer:
    copyed_model = deepcopy(model)
    _mh = deepcopy(copyed_model.head)
    _mdevice = _mh.weight.device
    _mdtype = _mh.weight.dtype
    copyed_model.head = _mh.__class__(_mh.in_features, len(target_classes), _mh.bias is not None, _mdevice, _mdtype)
    assert _mh.out_features == len(GVM.cl_mngr.current_task_classes), f"{_mh.out_features}, {len(GVM.cl_mngr.current_task_classes)}"
    _hw = torch.cat([GVM.param_dict[f'task_params_{_t}']['head.weight'].data.to(_mdevice, _mdtype) for _t in range(task_id + 1)])
    assert copyed_model.head.weight.data.shape == _hw.shape
    copyed_model.head.weight.data = _hw

    if _mh.bias is not None:
        _hb = torch.cat([GVM.param_dict[f'task_params_{_t}']['head.bias'].data.to(_mdevice, _mdtype) for _t in range(task_id + 1)])
        assert copyed_model.head.bias.data.shape == _hb.shape
        copyed_model.head.bias.data = _hb

    return copyed_model


def _make_prediction(task_model: VisionTransformer, taskid_list: list[int], target_classes: list[int], task_id: int, single_image: Tensor, **kwargs) -> Tensor:
    assert single_image.ndim == 3, f"{single_image.shape}"
    assert isinstance(taskid_list, list), f"{type(taskid_list)}"

    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=GVM.args.use_amp):
        with torch.no_grad():
            feature = torch.cat([misc.load_params(GVM.param_dict, task_model, f'task_params_{_taskid}').forward_features(single_image.unsqueeze(0)) for _taskid in taskid_list])

    model = merge_head(task_model, task_id, target_classes)
    scores_list = torch.softmax(model.forward_head(feature), 1)
    assert scores_list.ndim == 2, f"{scores_list.shape}"
    _max_scores, preds = scores_list.max(dim=1)

    _max_scores: Tensor
    preds: Tensor
    assert _max_scores.ndim == 1
    assert len(preds) == len(taskid_list)

    assert preds.ndim == 1

    if kwargs.get('sort_by_score', True):
        preds = preds[_max_scores.argsort(descending=True)]

    assert preds.ndim == 1, f"{preds.shape}"
    return preds


def evaluate_one_task(train_taskid: int, eval_taskid: int, eval_task_classes: list[int], target_classes: list[int], model: VisionTransformer) -> OrderedDict[str, float]:
    _ttimer = ttime()

    dataset = define_dataset(GVM, eval_task_classes, training=False, transform_type='autoaug', verbose=train_taskid == eval_taskid == 0, return_index=True)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=args.eval_workers, pin_memory=True, timeout=30 if args.eval_workers > 0 else 0)

    dataset_proto = define_dataset(GVM, eval_task_classes, training=False, transform_type='prototype', verbose=train_taskid == eval_taskid == 0, return_index=True)
    dataloader_proto = DataLoader(dataset_proto, batch_size=100, shuffle=False, num_workers=args.eval_workers, pin_memory=True, timeout=30 if args.eval_workers > 0 else 0)
    assert len(dataset_proto) == len(dataset)
    assert len(dataloader_proto) == len(dataloader)

    set_model_mode(model, training=False)
    scalar_meter = misc.ScalarMeter(ncm_top1="samp_avg:>6.2%", ncm_topnn="samp_avg:>6.2%", ncm_nn="samp_avg:.2f",
                                    task_top1="samp_avg:>6.2%", task_topnn="samp_avg:>6.2%", task_nn="samp_avg:.2f",
                                    acc_top1="samp_avg:>6.2%", acc_topnn="samp_avg:>6.2%", acc_nn="samp_avg:.2f")

    for (images, target, idx), (images_p, _, idx_p) in tqdm.tqdm(zip(dataloader, dataloader_proto), total=len(dataloader), dynamic_ncols=True, disable=not GVM.args.show_bar):
        images: Tensor = images.cuda(non_blocking=True, device=args.device)
        target: Tensor = target.cuda(non_blocking=True, device=args.device)
        images_p: Tensor = images_p.cuda(non_blocking=True, device=args.device)

        assert torch.all(idx == idx_p)

        inferred_preds_list = infer_nn_preds(GVM.cache_dict['query_function'], images_p, num_nn_idx=GVM.args.num_nn_idx, num_cls_limit=GVM.args.num_cls_limit)

        ncm_preds = []
        task_preds = []
        task_targets = torch.as_tensor([_label_to_taskid(_t.item()) for _t in target])
        preds = []
        for i_img, (_image, _inferred_preds) in enumerate(zip(images, inferred_preds_list)):
            ncm_preds.append(_count_with_order(_inferred_preds)[0])
            _inferred_taskids = infer_nn_taskids(_inferred_preds)
            task_preds.append(torch.as_tensor(_inferred_taskids))
            preds.append(_make_prediction(task_model=model, taskid_list=_inferred_taskids, single_image=_image, task_id=train_taskid, coarse_preds=ncm_preds[-1], target_classes=target_classes))

        acc_top1, acc_topnn, acc_nn = misc.calc_acc_topnn_dynamically(preds, target)
        task_top1, task_topnn, task_nn = misc.calc_acc_topnn_dynamically(task_preds, task_targets)
        ncm_top1, ncm_topnn, ncm_nn = misc.calc_acc_topnn_dynamically(ncm_preds, target)
        scalar_meter.add_step_value(target.shape[0], acc_top1=acc_top1, acc_topnn=acc_topnn, acc_nn=acc_nn, ncm_top1=ncm_top1, ncm_topnn=ncm_topnn, ncm_nn=ncm_nn, task_top1=task_top1, task_topnn=task_topnn, task_nn=task_nn)

    result_dict = scalar_meter.update_epoch_average_value()

    print(f"Task [{train_taskid + 1}/{GVM.cl_mngr.num_tasks}]:: Eval [{eval_taskid + 1:>{len(_tt := str(train_taskid + 1))}}/{_tt}]: eval_time={ttime() - _ttimer:.1f}s, {scalar_meter.format_outout(result_dict)}")

    result_dict['num_samples'] = len(dataset)

    return result_dict


def evaluate_tasks_sofar(train_taskid: int, model: VisionTransformer):
    print(f"*" * 60 + " Start Evaluation " + "*" * 60)

    for _n in ('prototype_array', 'label_array'):
        GVM.cache_dict[_n] = GVM.cache_dict[_n].cuda(device=args.device)
    set_model_mode(GVM.cache_dict['query_function'], training=False)

    if 'sample_level_avg_acc' not in GVM.cache_dict:
        GVM.cache_dict['sample_level_avg_acc'] = {'coarse_acc': np.zeros(GVM.cl_mngr.num_tasks), 'task_acc': np.zeros(GVM.cl_mngr.num_tasks), 'fine_acc': np.zeros(GVM.cl_mngr.num_tasks)}
    sample_level_avg_acc_meter = misc.ScalarMeter(coarse_acc="samp_avg:>6.2%", task_acc="samp_avg:>6.2%", fine_acc="samp_avg:>6.2%")

    sofar_task_classes = GVM.cl_mngr.sofar_task_classes
    target_classes = list(itertools.chain(*sofar_task_classes))

    for eval_taskid, eval_task_classes in enumerate(sofar_task_classes):
        model = misc.load_params(GVM.param_dict, model, f'task_params_{eval_taskid}')

        result_dict = evaluate_one_task(train_taskid, eval_taskid, eval_task_classes, target_classes, model)
        GVM.acc_mat_dict['coarse'][train_taskid, eval_taskid] = result_dict['ncm_top1']
        GVM.acc_mat_dict['task'][train_taskid, eval_taskid] = result_dict['task_top1']
        GVM.acc_mat_dict['fine'][train_taskid, eval_taskid] = result_dict['acc_top1']

        sample_level_avg_acc_meter.add_step_value(result_dict['num_samples'], coarse_acc=result_dict['ncm_top1'], task_acc=result_dict['task_top1'], fine_acc=result_dict['acc_top1'])

    for k, v in sample_level_avg_acc_meter.update_epoch_average_value().items():
        GVM.cache_dict['sample_level_avg_acc'][k][train_taskid] = v

    for _n in ('prototype_array', 'label_array'):
        GVM.cache_dict[_n] = GVM.cache_dict[_n].cpu()
    GVM.cache_dict['query_function'].cpu()

    fgt_mat = misc.calc_forgetting(GVM.acc_mat_dict['fine'])
    GVM.forgetting_dict['fine_fgt_mat'] = fgt_mat

    print(f"\nACC_MAT (task {train_taskid + 1}):\n" + f"{GVM.acc_mat_dict['fine']}")
    print(f"\nFGT_MAT (task {train_taskid + 1}):\n" + f"{fgt_mat}")
    print(f"\nsample_level_avg_acc (task {train_taskid + 1}):\n" + f"{GVM.cache_dict['sample_level_avg_acc']['fine_acc']}")


def ending_info(**kwargs):
    print(f"{'#' * 60} End of all the tasks {'#' * 60}")

    average_forgetting = misc.calc_sample_level_forgetting(GVM.forgetting_dict['fine_fgt_mat'])
    print(f"task_avg_for: {average_forgetting}")
    print(f"For@1(average) = {np.mean(average_forgetting):.2%}, For@1(last) = {average_forgetting[-1]:.2%}")

    _acc_mat = GVM.cache_dict['sample_level_avg_acc']['fine_acc']
    print(f"task_avg_acc:: {_acc_mat}")
    print(f"Acc@1(average) = {np.mean(_acc_mat):.2%}, Acc@1(last) = {_acc_mat[-1]:.2%}")
    print(f"Totoal time = {misc.format_duration(ttime() - kwargs['exp_start_time'])}")


if __name__ == "__main__":
    args = getargs()
    seed_etc_options(args.seed)

    GVM.args = args
    _dataset_class_manager = ImagePathDatasetClassManager(**{args.dataset: args.data_root})
    GVM.path_data_dict = {'train': _dataset_class_manager[args.dataset](train=True),
                          'eval': _dataset_class_manager[args.dataset](train=False)}
    GVM.cl_mngr = ClassIncrementalManager(GVM.path_data_dict['eval'].class_list, args.num_tasks, args.seed, shuffle=False)
    GVM.acc_mat_dict = OrderedDict(coarse=np.zeros([_nt := GVM.cl_mngr.num_tasks, _nt]), task=np.zeros([_nt, _nt]), fine=np.zeros([_nt, _nt]))
    GVM.forgetting_dict = OrderedDict(fine_fgt_mat=np.zeros([_nt, _nt]))
    GVM.cache_dict = {}
    GVM.param_dict = {}
    GVM.label_map_g2l = {}
    GVM.label_map_l2g = {}

    _exp_start_time = ttime()

    _prompt_args_dict = misc.get_specific_args_dict(args, 'prompt_')
    stage1_prompt_arges_dict = {}
    stage1_prompt_arges_dict['prompt_len'] = args.stage1_prompt_len
    stage1_prompt_arges_dict['prompt_start_block'] = 0
    stage1_prompt_arges_dict['prompt_end_block'] = args.stage1_prompt_layer - 1
    stage1_prompt_arges_dict['prompt_init'] = args.prompt_init
    basemode = bool(args.stage1_prompt_len)
    base_model: VisionTransformer = create_model(
        args.model,
        pretrained=True,
        num_classes=len(
            GVM.cl_mngr.all_classes) //
        args.num_tasks if args.seperate_head else len(
            GVM.cl_mngr.all_classes),
        prompt_args_dict=stage1_prompt_arges_dict)

    for taskid, current_task_classes in GVM.cl_mngr:
        print(f"{'#' * 61} Task: [{taskid + 1}/{_nt}] {'#' * 61}")
        print(f"Current classes ({len(current_task_classes)}): {current_task_classes}")

        model: VisionTransformer = create_model(args.model, pretrained=True, num_classes=len(current_task_classes) if args.seperate_head else len(GVM.cl_mngr.all_classes),
                                                prompt_args_dict=_prompt_args_dict)
        stage1_model: VisionTransformer = create_model(
            args.model,
            pretrained=True,
            num_classes=len(
                GVM.cl_mngr.all_classes) //
            args.num_tasks if args.seperate_head else len(
                GVM.cl_mngr.all_classes),
            prompt_args_dict=stage1_prompt_arges_dict)

        GVM.update_label_maps(taskid, current_task_classes)

        if basemode:
            if taskid == 0:
                base_model = train_one_task(taskid, current_task_classes, base_model, basemode=basemode)
            elif args.momentum != 1.0:
                stage1_model = train_one_task(taskid, current_task_classes, stage1_model, basemode=basemode)
                base_model.cuda(device=args.device)
                for _, (block1, block2) in enumerate(zip(base_model.blocks, stage1_model.blocks)):
                    if hasattr(block1, "prompt") and block1.prompt is not None:
                        with torch.no_grad():
                            block1.prompt.prompt.copy_(args.momentum * block1.prompt.prompt + (1 - args.momentum) * block2.prompt.prompt)

        model = train_one_task(taskid, current_task_classes, model)

        GVM.cache_dict['query_function'] = base_model
        _prototype_array, _label_array = extract_prototypes(taskid, current_task_classes, GVM.cache_dict['query_function'], representation=args.representation)
        GVM.cache_prototypes(('prototype_array', _prototype_array), ('label_array', _label_array))

        evaluate_tasks_sofar(taskid, model)

    ending_info(exp_start_time=_exp_start_time)
