# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
from pathlib import Path
import cv2
import numpy as np
import torch
from functools import lru_cache

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class DetectionValidator(BaseValidator):
    """
    支持深度感知过滤的目标检测验证器
    实现流程: 预处理 → 模型推理 → 后处理(NMS+深度筛选) → 指标计算 → 结果输出
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """初始化检测验证器并配置深度感知参数"""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU向量用于mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # 用于自动标注
        self.depth_dir = args.get('depth_dir', None)  # 深度图像目录
        self.depth_threshold = args.get('depth_threshold', (0.001, 50))  # 深度过滤阈值(米)
        self.depth_scale = 1000.0  # 深度单位转换因子
        self.use_depth_cache = args.get('use_depth_cache', True)  # 是否缓存深度图像

    def load_depth_image(self, img_path):
        """加载并预处理深度图像，支持多通道转单通道"""
        if not self.depth_dir:
            return None

        img_filename = os.path.basename(img_path)
        depth_filename = f"{os.path.splitext(img_filename)[0]}.png"
        depth_path = os.path.join(self.depth_dir, depth_filename)

        if not os.path.exists(depth_path):
            LOGGER.warning(f"深度图像不存在: {depth_path}")
            return None

        # 读取深度图
        depth_img = cv2.imread(depth_path, -1)
        if depth_img is None:
            LOGGER.warning(f"无法读取深度图像: {depth_path}")
            return None

        # 检查图像通道数，转为单通道
        if depth_img.ndim == 3:  # 多通道图像(如RGB)
            LOGGER.info(f"将多通道深度图转为单通道: {depth_path}")
            depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)  # 转为灰度图

        # 验证是否为单通道
        if depth_img.ndim != 2:
            LOGGER.warning(f"无效的深度图像: {depth_path}（无法转为单通道）")
            return None

        # 转换为米并调整尺寸
        depth_meters = depth_img.astype(np.float32) / self.depth_scale
        depth_meters = cv2.resize(depth_meters, (self.args.imgsz, self.args.imgsz))
        return depth_meters

    # def filter_by_depth(self, pred, depth_img):
    #     """基于深度图像过滤预测框，并记录深度信息"""
    #     min_depth, max_depth = self.depth_threshold
    #     if pred.device != torch.device('cpu'):
    #         pred = pred.cpu()  # 确保与深度图像(NumPy数组)在同一设备
    #
    #     pred_np = pred.numpy()
    #     valid_indices = []
    #     # 新增：用于存储每个框的深度值（平均深度）
    #     pred_depths = []
    #
    #     for i in range(len(pred_np)):
    #         x1, y1, x2, y2 = map(int, pred_np[i, :4])
    #         # 边界校正
    #         x1 = max(0, x1)
    #         y1 = max(0, y1)
    #         x2 = min(depth_img.shape[1], x2)
    #         y2 = min(depth_img.shape[0], y2)
    #
    #         if x1 >= x2 or y1 >= y2:
    #             continue  # 无效边界框
    #
    #         # 提取边界框内的深度值
    #         bbox_depth = depth_img[y1:y2, x1:x2]
    #         valid_depths = bbox_depth[(bbox_depth > 0) & np.isfinite(bbox_depth)]
    #
    #         if len(valid_depths) == 0:
    #             continue  # 无有效深度值
    #
    #         # 计算平均深度并过滤
    #         avg_depth = np.mean(valid_depths)
    #         if min_depth <= avg_depth <= max_depth:
    #             valid_indices.append(i)
    #             pred_depths.append(avg_depth)  # 记录平均深度
    #             LOGGER.debug(
    #                 f"保留框 {i}: 类别={int(pred_np[i, 5])}, 置信度={pred_np[i, 4]:.3f}, 平均深度={avg_depth:.2f}米"
    #             )
    #
    #     # 新增：将深度信息添加到预测结果中（扩展pred的维度，如在第7列存储深度值）
    #     if valid_indices:
    #         # 将pred转换为CPU tensor（若不在CPU）
    #         pred = pred[torch.tensor(valid_indices, device=pred.device)]
    #         # 创建深度值的tensor并扩展到pred中
    #         depths_tensor = torch.tensor(pred_depths, device=pred.device).unsqueeze(1)
    #         pred = torch.cat([pred, depths_tensor], dim=1)  # 预测结果变为 [x1,y1,x2,y2,conf,cls,depth]
    #     else:
    #         pred = torch.empty((0, 7), device=pred.device)  # 预留深度列
    #
    #     return pred
    def filter_by_depth(self, pred, depth_img):
        """基于深度图像过滤预测框，并记录深度信息（使用中位数）"""
        min_depth, max_depth = self.depth_threshold
        if pred.device != torch.device('cpu'):
            pred = pred.cpu()  # 确保与深度图像(NumPy数组)在同一设备

        pred_np = pred.numpy()
        valid_indices = []
        pred_depths = []  # 存储每个框的深度值（中位数）

        for i in range(len(pred_np)):
            x1, y1, x2, y2 = map(int, pred_np[i, :4])
            # 边界校正
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(depth_img.shape[1], x2)
            y2 = min(depth_img.shape[0], y2)

            if x1 >= x2 or y1 >= y2:
                continue  # 无效边界框

            # 提取边界框内的深度值
            bbox_depth = depth_img[y1:y2, x1:x2]
            valid_depths = bbox_depth[(bbox_depth > 0) & np.isfinite(bbox_depth)]

            if len(valid_depths) == 0:
                continue  # 无有效深度值

            # 计算中位数深度并过滤
            median_depth = np.median(valid_depths)  # 使用中位数替代平均
            if min_depth <= median_depth <= max_depth:
                valid_indices.append(i)
                pred_depths.append(median_depth)  # 记录中位数深度
                LOGGER.debug(
                    f"保留框 {i}: 类别={int(pred_np[i, 5])}, 置信度={pred_np[i, 4]:.3f}, 中位数深度={median_depth:.2f}米"
                )

        # 将深度信息添加到预测结果中
        if valid_indices:
            pred = pred[torch.tensor(valid_indices, device=pred.device)]
            depths_tensor = torch.tensor(pred_depths, device=pred.device).unsqueeze(1)
            pred = torch.cat([pred, depths_tensor], dim=1)  # 扩展为 [x1,y1,x2,y2,conf,cls,depth]
        else:
            pred = torch.empty((0, 7), device=pred.device)  # 预留深度列

        return pred

    # def filter_by_depth(self, pred, depth_img):
    #     """基于预测框中心点的深度值过滤预测结果"""
    #     min_depth, max_depth = self.depth_threshold
    #     if pred.device != torch.device('cpu'):
    #         pred = pred.cpu()  # 确保与深度图像(NumPy数组)在同一设备
    #
    #     pred_np = pred.numpy()
    #     valid_indices = []
    #
    #     for i in range(len(pred_np)):
    #         x1, y1, x2, y2 = map(int, pred_np[i, :4])
    #         # 边界校正
    #         x1 = max(0, x1)
    #         y1 = max(0, y1)
    #         x2 = min(depth_img.shape[1], x2)
    #         y2 = min(depth_img.shape[0], y2)
    #
    #         if x1 >= x2 or y1 >= y2:
    #             continue  # 无效边界框
    #
    #         # 计算预测框中心点坐标
    #         center_x = int((x1 + x2) // 2)
    #         center_y = int((y1 + y2) // 2)
    #
    #         # 检查中心点是否在有效范围内
    #         if 0 <= center_x < depth_img.shape[1] and 0 <= center_y < depth_img.shape[0]:
    #             # 获取中心点的深度值
    #             center_depth = depth_img[center_y, center_x]
    #
    #             # 检查深度值是否有效且在阈值范围内
    #             if 0 < center_depth < float('inf') and min_depth <= center_depth <= max_depth:
    #                 valid_indices.append(i)
    #                 LOGGER.debug(
    #                     f"保留框 {i}: 类别={int(pred_np[i, 5])}, 置信度={pred_np[i, 4]:.3f}, 中心点深度={center_depth:.2f}米")
    #
    #     return pred[torch.tensor(valid_indices, device=pred.device)] if valid_indices else torch.empty((0, 6),
    #                                                                                                    device=pred.device)

    # def filter_by_depth(self, pred, depth_img):
    #     """基于深度图像过滤预测框，返回有效索引的预测结果"""
    #     min_depth, max_depth = self.depth_threshold
    #     if pred.device != torch.device('cpu'):
    #         pred = pred.cpu()  # 确保与深度图像(NumPy数组)在同一设备
    #
    #     pred_np = pred.numpy()
    #     valid_indices = []
    #
    #     for i in range(len(pred_np)):
    #         x1, y1, x2, y2 = map(int, pred_np[i, :4])
    #         # 边界校正
    #         x1 = max(0, x1)
    #         y1 = max(0, y1)
    #         x2 = min(depth_img.shape[1], x2)
    #         y2 = min(depth_img.shape[0], y2)
    #
    #         if x1 >= x2 or y1 >= y2:
    #             continue  # 无效边界框
    #
    #         # 提取边界框内的深度值
    #         bbox_depth = depth_img[y1:y2, x1:x2]
    #         valid_depths = bbox_depth[(bbox_depth > 0) & np.isfinite(bbox_depth)]
    #
    #         if len(valid_depths) == 0:
    #             continue  # 无有效深度值
    #
    #         # 计算平均深度并过滤
    #         avg_depth = np.mean(valid_depths)
    #         if min_depth <= avg_depth <= max_depth:
    #             valid_indices.append(i)
    #             LOGGER.debug(
    #                 f"保留框 {i}: 类别={int(pred_np[i, 5])}, 置信度={pred_np[i, 4]:.3f}, 平均深度={avg_depth:.2f}米")
    #
    #     return pred[torch.tensor(valid_indices, device=pred.device)] if valid_indices else torch.empty((0, 6),
    #                                                                                                    device=pred.device)

    def preprocess(self, batch):
        """预处理批次数据，为模型推理做准备"""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = (
                [
                    torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                    for i in range(nb)
                ]
                if self.args.save_hybrid
                else []
            )  # 用于自动标注

        return batch

    def init_metrics(self, model):
        """初始化评估指标和数据集相关参数"""
        val = self.data.get(self.args.split, "")  # 验证集路径
        self.is_coco = isinstance(val, str) and "coco" in val and val.endswith(f"{os.sep}val2017.txt")  # 是否为COCO数据集
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # 是否为LVIS数据集
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(len(model.names)))
        self.args.save_json |= (self.is_coco or self.is_lvis) and not self.training  # 如果训练COCO，在最终验证时运行
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def get_desc(self):
        """返回描述验证指标的格式化字符串"""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds):
        pred_list = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
        )

        if self.depth_dir:
            for i in range(len(pred_list)):
                pred = pred_list[i]
                if len(pred) == 0:
                    continue

                img_path = self.dataloader.dataset.im_files[i]
                if self.use_depth_cache:
                    depth_img = self._cached_load_depth_image(img_path)
                else:
                    depth_img = self.load_depth_image(img_path)

                if depth_img is not None:
                    # 执行深度过滤（在CPU上）
                    valid_pred = self.filter_by_depth(pred, depth_img)

                    # 确保结果回到与输入相同的设备
                    if valid_pred.device != pred.device:
                        valid_pred = valid_pred.to(pred.device)

                    pred_list[i] = valid_pred

        return pred_list

    @lru_cache(maxsize=128)  # 缓存最近128张深度图像
    def _cached_load_depth_image(self, img_path):
        """带缓存的深度图像加载方法"""
        return self.load_depth_image(img_path)

    def _prepare_batch(self, si, batch):
        """准备单个样本的真实标签数据"""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # 目标框
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # 转换到原图尺寸
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """准备单个样本的预测数据，转换到原图尺寸"""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # 转换到原图尺寸
        return predn

    def update_metrics(self, preds, batch):
        """更新评估指标，基于深度筛选后的预测结果"""
        for si, pred in enumerate(preds):
            if len(pred) == 0:
                continue
            # 提取深度信息（假设深度在第6列，索引为6）
            depths = pred[:, 6].cpu().numpy()
            for i, depth in enumerate(depths):
                LOGGER.info(f"检测框 {i + 1} 深度：{depth:.2f} 米")
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()

            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # 预测处理
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # 评估
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # 保存结果
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                file = self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, pbatch["ori_shape"], file)

    def finalize_metrics(self, *args, **kwargs):
        """设置最终的指标值，包括速度和混淆矩阵"""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """返回指标统计结果"""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # 转为numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        return self.metrics.results_dict

    def print_results(self):
        """打印验证结果，包括总体指标和类别级指标"""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # 打印格式
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(
                f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # 打印每个类别的结果
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
                )

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        返回正确预测的矩阵
        Args:
            detections (torch.Tensor): 检测结果，形状[N, 6] (x1, y1, x2, y2, conf, class)
            gt_bboxes (torch.Tensor): 真实边界框，形状[M, 4] (x1, y1, x2, y2)
            gt_cls (torch.Tensor): 真实类别，形状[M]
        Returns:
            (torch.Tensor): 正确预测矩阵，形状[N, 10] (10个IoU级别)
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def build_dataset(self, img_path, mode="val", batch=None):
        """构建YOLO数据集"""
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        """构建并返回数据加载器"""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # 返回数据加载器

    def plot_val_samples(self, batch, ni):
        """绘制验证样本"""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """绘制预测结果"""
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # 预测结果

    def save_one_txt(self, predn, save_conf, shape, file):
        """将预测结果保存为TXT文件(YOLO格式)"""
        gn = torch.tensor(shape)[[1, 0, 1, 0]]  # 归一化增益whwh
        for *xyxy, conf, cls in predn.tolist():
            xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化xywh
            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 标签格式
            with open(file, "a") as f:
                f.write(("%g " * len(line)).rstrip() % line + "\n")

    def pred_to_json(self, predn, filename):
        """将预测结果序列化为COCO JSON格式"""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy中心到左上角
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])]
                                   + (1 if self.is_lvis else 0),  # 索引从1开始(如果是lvis)
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """评估JSON格式的预测结果，返回性能统计"""
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            pred_json = self.save_dir / "predictions.json"  # 预测结果
            anno_json = (
                    self.data["path"]
                    / "annotations"
                    / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )  # 标注文件
            pkg = "pycocotools" if self.is_coco else "lvis"
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")
                if self.is_coco:
                    from pycocotools.coco import COCO  # noqa
                    from pycocotools.cocoeval import COCOeval  # noqa

                    anno = COCO(str(anno_json))  # 初始化标注API
                    pred = anno.loadRes(str(pred_json))  # 初始化预测API(必须传递字符串，而非Path)
                    val = COCOeval(anno, pred, "bbox")
                else:
                    from lvis import LVIS, LVISEval

                    anno = LVIS(str(anno_json))  # 初始化标注API
                    pred = anno._load_json(str(pred_json))  # 初始化预测API(必须传递字符串，而非Path)
                    val = LVISEval(anno, pred, "bbox")
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # 要评估的图像
                val.evaluate()
                val.accumulate()
                val.summarize()
                if self.is_lvis:
                    val.print_results()  # 显式调用打印结果
                # 更新mAP50-95和mAP50
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP50"], val.results["AP"]]
                )
            except Exception as e:
                LOGGER.warning(f"{pkg} unable to run: {e}")
        return stats