import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型（完全保留你的路径）
    model = YOLO(r'D:\yolov10\runs\train1\v10n111-yellow-correction\weights\best.pt')

    # 验证（仅添加加速参数，所有原始配置不变）
    results = model.val(
        data=r"D:\yolov10\yellow\datasets_output\corrected_dataset\data.yaml",  # 原始：验证数据集路径
        imgsz=640,  # 原始：图像尺寸
        conf=0.25,  # 原始：置信度阈值
        iou=0.45,  # 原始：IOU阈值
        save_json=True,  # 原始：保存JSON格式
        project='runs/train/yellow',  # 原始：结果保存路径
        name='v10n-CICE-correction',  # 原始：结果名称（修正多余空字符串）
        device='0',  # 原始：指定GPU设备

        # 仅新增3个核心加速参数（不影响结果，只提速度）
        batch=16,  # 批量推理（8G显存设8，16G设16，GPU并行处理）
        workers=8,  # 多线程加载数据（避免GPU等数据）
        # save=False,  # 关闭预测图保存（IO耗时占比高，不影响JSON）
        # show=False,  # 关闭实时显示（无GUI耗时）
        # save_txt=False,  # 关闭TXT保存（减少冗余IO）
        # plots=False  # 关闭图表生成（减少额外计算）
    )