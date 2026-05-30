A test to run YOLO11 (nano) using OxiONNX.

```
wget https://huggingface.co/giangndm/yolo11-onnx/resolve/main/yolo11n_640.onnx
```

```
Model loaded successfully!
Inputs: ["images"]
Metadata: ModelMetadata { producer_name: "pytorch", producer_version: "2.3.1", domain: "", graph_name: "main_graph", ir_version: 10, opset_imports: [("", 11)], custom_metadata: {"batch": "1", "imgsz": "[640, 640]", "date": "2025-01-16T21:05:27.412120", "version": "8.3.12", "docs": "https://docs.ultralytics.com", "stride": "32", "task": "detect", "description": "Ultralytics YOLO11n model trained on /usr/src/ultralytics/ultralytics/cfg/datasets/coco.yaml", "license": "AGPL-3.0 License (https://ultralytics.com/license)", "names": "{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}", "author": "Ultralytics"} }
Output Index [0]: Name = output0, Type = F32, Shape = [Some(1), Some(84), Some(8400)], Dim Params = [None, None, None]
Error: Internal("reshape: element count mismatch (33600 vs [1, 128, 20, 20])")
```
