import os
import os.path as osp
from collections import OrderedDict
import torch
import numpy as np
from torch import Tensor
import argparse
from typing import Any


def str2bool(s):
    if isinstance(s, bool):
        return s
    true_list = ('y', 'true', '1')
    false_list = ('n', 'false', '0')
    s = str(s).lower()
    if s in true_list:
        return True
    elif s in false_list:
        return False
    else:
        raise argparse.ArgumentTypeError(f"'{s}' should be in {true_list} or {false_list}")


def format_duration(delta_time: float) -> str:
    delta_time = round(delta_time)
    return f"{delta_time // 3600:02d}:{delta_time // 60 % 60:02d}:{delta_time % 60:02d}"


def load_params(source: os.PathLike | dict, model: torch.nn.Module, *param_names: str, strict: bool = False, remove_prefix="_orig_mod.") -> torch.nn.Module:
    param_dict = {'base_params': None, 'task_params': None, 'query_function': None}
    assert len(param_names) <= len(param_dict), "Only support base-params, task-params and query function now."
    for param_name in param_names:
        if isinstance(source, os.PathLike):
            ckpt_path = osp.join(source, f'{param_name}.pth')
            assert osp.exists(ckpt_path), f"\"{ckpt_path}\" does not exists."
            params = torch.load(ckpt_path, map_location='cpu')
        elif isinstance(source, dict):
            assert param_name in source.keys(), f"{param_name} does not exist in the dict."
            params = source[param_name]
        else:
            raise TypeError(f"type(source)={type(source)}")
        for n in param_dict.keys():
            if n in param_name:
                assert param_dict[n] is None, "Cannot not load more than one task_params"
                param_dict[n] = params

    mearged_state_dict: OrderedDict[str, Tensor] = OrderedDict()
    for v in param_dict.values():
        if v is not None:
            mearged_state_dict.update(v)

    if remove_prefix:
        for n in list(mearged_state_dict.keys()):
            if n.startswith(remove_prefix):
                mearged_state_dict[n[len(remove_prefix):]] = mearged_state_dict[n]
                mearged_state_dict.pop(n)

    strict_loading = all([v is not None for v in param_dict.values()]) or strict

    missing_keys, unexpected_keys = model.load_state_dict(mearged_state_dict, strict_loading)
    assert len(unexpected_keys) == 0, f"It is expected that all the loaded keys exist in the model. param_names={param_names}. unexpected_keys={unexpected_keys}."
    return model


class ScalarMeter():
    def __init__(self, formatter_sep=", ", **varname_format: dict[str, str]):
        '''
        varname_format: var1="reduce_mode:format_str", e.g., loss="samp_avg:.4f".
            reduce_mode: step_first, step_last, step_sum, step_avg, samp_sum, samp_avg (default).
            format_str: the same as the string format in python, "" represents not specified.
        '''
        self.batch_counter = 0

        assert len(varname_format) > 0, "No variables!"

        self.value_list_dict: OrderedDict[str, list[float]] = OrderedDict([(n, []) for n in varname_format.keys()])  # {'var1': [val11, val12, ...], 'var2':[val21, val22, ...], ...}
        self.sample_counter_list = []

        format_dict = {}
        reduce_dict = {}
        for name, form in varname_format.items():
            if not form:
                reduce_dict[name] = 'samp_avg'
                format_dict[name] = ''
            else:
                assert ':' in form, f"use {name}=\"reduce_mode:format_str\""
                reduce_mode, format_str = form.split(':')
                assert isinstance(reduce_mode, str) and isinstance(format_str, str), f"{type(reduce_mode)}, {type(format_str)}"
                reduce_dict[name] = 'samp_avg' if not reduce_mode else reduce_mode
                format_dict[name] = format_str

        self.scalar_formatter = ScalarFormatter(formatter_sep, **format_dict)

        self.reduce_dict = reduce_dict
        self.varname_format = varname_format

    def add_step_value(self, num_samples: int, **step_value_dict: dict[str, float]):
        '''
        step_value_dict: var1=val1, var2=val2
        This function should be only called once for each step.
        '''
        for k, v in step_value_dict.items():
            assert k in self.value_list_dict, f"{k}"
            assert isinstance(v, float), f"unsupported type:{type(v)}"
            self.value_list_dict[k].append(v)
        self.sample_counter_list.append(num_samples)
        self.batch_counter += 1

    def update_epoch_average_value(self) -> OrderedDict[str, float]:
        reduced_value_dict = OrderedDict()
        sc_ary: np.ndarray = np.array(self.sample_counter_list)
        for name, v_list in self.value_list_dict.items():
            assert len(v_list) >= 1, f"{name}: {v_list}"
            v_ary: np.ndarray = np.array(v_list)
            assert v_ary.shape == sc_ary.shape
            match reduce_mode := self.reduce_dict[name]:
                case 'samp_avg':
                    v_reduced = np.dot(v_ary, sc_ary) / np.sum(sc_ary)
                case 'samp_sum':
                    v_reduced = np.dot(v_ary, sc_ary)
                case 'step_avg':
                    v_reduced = np.mean(v_ary)
                case 'step_sum':
                    v_reduced = np.sum(v_ary)
                case 'step_first':
                    v_reduced = v_ary[0]
                case 'step_last':
                    v_reduced = v_ary[-1]
                case _:
                    raise NotImplementedError(f"Unsupported reduce_mode: {reduce_mode}")

            reduced_value_dict[name] = v_reduced

        for k in self.value_list_dict.keys():
            self.value_list_dict[k].clear()
        self.sample_counter_list.clear()

        return reduced_value_dict

    def format_outout(self, reduced_value_dict: OrderedDict) -> str:
        return self.scalar_formatter(**reduced_value_dict)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: {[f'{k}: {v}' for k, v in self.varname_format.items()]}"


class ScalarFormatter():
    def __init__(self, sep=" ", **var_format_dict: OrderedDict[str, str]) -> None:
        '''
        var_format_dict: {var_str: ""}
        '''
        self.var_format_dict = var_format_dict
        assert isinstance(sep, str)
        self.sep = sep

    def add_var_format(self, var_str: str, var_format: str):
        self.var_format_dict[var_str] = var_format

    def __call__(self, **value_dict: OrderedDict[str, float]) -> str:
        output_str = []
        for var_str in value_dict.keys():
            assert var_str in self.var_format_dict.keys(), f"{var_str}"
            if self.var_format_dict[var_str] == "":
                output_str.append(f"{var_str}={{{var_str}}}")
            else:
                output_str.append(f"{var_str}={{{var_str}:{self.var_format_dict[var_str]}}}")
        output_str = self.sep.join(output_str)
        return output_str.format(**value_dict)


def calc_accuracy(preds: Tensor, target: Tensor, topk=(1,)) -> list[float]:
    assert target.ndim == 1, f"{target.shape}"
    assert preds.shape[0] == target.shape[0], f"{preds.shape}, {target.shape}"
    assert target.dtype == torch.long

    maxk = max(topk)
    res = []
    if preds.dtype == torch.long:
        if preds.ndim == 1:
            assert maxk == 1, f"topk={topk} shoule be (1, ) in this case."
            _pred: Tensor = preds.unsqueeze(1)  # [b, k=1]
        elif preds.ndim == 2:
            assert maxk <= preds.shape[1], f"{maxk}, {preds.shape}"
            _pred: Tensor = preds[:, :maxk]  # [b, k]
        for _p in _pred:
            assert len(_p.unique()) == _pred.shape[1], "The predictions should be unqiue."
    elif preds.dtype in (torch.float16, torch.float32):
        assert preds.ndim == 2
        _, _pred = preds.topk(maxk, 1, True, True)
    else:
        raise TypeError(f"dtype={preds.dtype}")

    _pred: Tensor = _pred.t()  # [b, k] -> [k, b]
    correct = _pred.eq(target.view(1, -1).expand_as(_pred))  # [b] -> [1, b] -> [k, b]

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True).item()
        res.append(correct_k / target.shape[0])
    return res


def calc_inc_acc(acc_mat: np.ndarray) -> np.ndarray:
    assert (n := acc_mat.shape[0]) == acc_mat.shape[1]
    task_avg_acc = []
    for i_task, acc_task_ary in enumerate(acc_mat):
        assert np.all(acc_task_ary[i_task + 1:] == 0), "acc_mat should be a lower triangular array"
        task_avg_acc.append(np.mean(acc_task_ary[:i_task + 1]))
    task_avg_acc = np.array(task_avg_acc)
    return task_avg_acc


def calc_forgetting(acc_mat: np.ndarray) -> np.ndarray:
    num_task = acc_mat.shape[0]
    for_mat = np.zeros_like(acc_mat)
    max_acc = np.max(acc_mat, 0)  # 每列的最大值，即每个task上最好的性能
    for i in range(num_task):
        for j in range(i):
            if not acc_mat[i, j] == 0:
                for_mat[i, j] = max_acc[j] - acc_mat[i, j]

    return for_mat


def calc_sample_level_forgetting(for_mat: np.ndarray) -> np.ndarray:
    num_task = for_mat.shape[0]
    sample_level_forgetting = np.zeros(num_task)
    for i in range(1, num_task):
        sample_level_forgetting[i] = for_mat[i, 0:i].mean()

    return sample_level_forgetting


def make_label_maps(taskid: int, task_classes: list[int]) -> tuple[dict[int, int], dict[str, int]]:
    g2l_map = {}
    l2g_map = {}
    for idx, cls in enumerate(task_classes):
        g2l_map[cls] = idx
        l2g_map[f'{taskid}:{idx}'] = cls
    return g2l_map, l2g_map


def calc_acc_topnn_dynamically(preds: list[Tensor] | Tensor, target: Tensor) -> tuple[float, float, float]:
    if isinstance(preds, list):
        assert len(preds) == len(target), f"{len(preds)}, {target.shape}"
    elif isinstance(preds, Tensor):
        if preds.ndim == 1:
            preds = torch.clone(preds).unsqueeze(1)
        else:
            assert preds.ndim == 2
    else:
        raise TypeError(f"{type(preds)}")

    acc_top1 = []
    acc_topnn = []
    mean_nn = []
    for _p, _t in zip(preds, target):
        _mnn = len(_p)
        _ac1, _acnn = calc_accuracy(_p.unsqueeze(0), _t.unsqueeze(0), topk=(1, _mnn))
        acc_top1.append(_ac1)
        acc_topnn.append(_acnn)
        mean_nn.append(_mnn)
    acc_top1 = sum(acc_top1) / len(acc_top1)
    acc_topnn = sum(acc_topnn) / len(acc_topnn)
    mean_nn = sum(mean_nn) / len(mean_nn)
    return acc_top1, acc_topnn, mean_nn


def get_specific_args_dict(args: argparse.Namespace, *start_str_list: str) -> dict[str, Any]:
    for _s in start_str_list:
        assert isinstance(_s, str)
    dst_args_dict = {}
    for k, v in args._get_kwargs():
        for _s in start_str_list:
            if k.startswith(_s):
                dst_args_dict[k] = v
    return dst_args_dict
