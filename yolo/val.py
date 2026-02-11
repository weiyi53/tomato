import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO(r'D:\yolov10\runs\train\yolov8n+corrected-yuanshi\weights\best.pt')  # 指定你训练后的权重路径

    # 验证
    results = model.val(data=r"D:\yolov10\corrected\data.yaml",  # 验证数据集路径
                        imgsz=640,  # 图像尺寸
                        conf=0.25,  # 置信度阈值
                        iou=0.45,  # IOU阈值
                        save_json=True,  # 保存结果为JSON格式
                        project='runs/val',  # 验证结果保存路径
                        name='yolov8n+corrected-yuanshi-my',  # 验证结果名称
                        device='0'  # 指定使用的设备
                        )


