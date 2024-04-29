from ultralytics import YOLO

model = YOLO('YOLOv8n-seg.pt')

if __name__ == '__main__':
    results = model.train(data=r'E:\Company\AITrain\convert_data\area\dataset.yaml',epochs=20)
    # model.export(format='onnx', simplify=True, dynamic=True)