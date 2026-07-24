import os
import argparse
# Try to import ultralytics for YOLO training
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

def train_yolo_segmentation(
    dataset_yaml: str, 
    epochs: int = 100, 
    imgsz: int = 640, 
    batch: int = 16,
    device: str = "0",
    model_size: str = "n"
):
    """
    Trains a YOLOv8-Seg model for tooth segmentation using dental panoramic or intraoral datasets.
    Supports datasets exported from Roboflow in YOLOv8-seg format.
    """
    if YOLO is None:
        print("Error: The 'ultralytics' library is required to run YOLO segmentation training.")
        print("Please install it using: pip install ultralytics")
        return
        
    model_name = f"yolov8{model_size}-seg.pt"
    print(f"Initializing YOLOv8-Seg model: {model_name}...")
    model = YOLO(model_name)
    
    print(f"Starting training on dataset {dataset_yaml}...")
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="orthofinix_seg",
        name="teeth_segmentation",
        save=True,
        plots=True,
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,
        optimizer="AdamW"
    )
    
    print("Training complete. Validating model performance...")
    val_results = model.val()
    print(f"Validation mAP50-95: {val_results.results_dict['metrics/mAP50-95(B)']}")
    
    print("Exporting model to ONNX and TFLite formats for production deployment...")
    model.export(format="onnx")
    model.export(format="tflite")
    print("Model exports completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8-Seg Tooth Segmentation Model")
    parser.add_argument("--data", type=str, default="dataset.yaml", help="Path to Roboflow dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, 0, 1, cuda)")
    parser.add_argument("--model_size", type=str, default="n", choices=["n", "s", "m", "l", "x"], help="YOLO model size")
    
    args = parser.parse_args()
    
    # Create a dummy dataset config if it doesn't exist for demonstration
    if not os.path.exists(args.data):
        print(f"Warning: Dataset file '{args.data}' not found. Writing a template dataset.yaml...")
        with open(args.data, "w") as f:
            f.write(f"""# OrthofinixAI Teeth Instance Segmentation Dataset Template
path: ./dataset
train: images/train
val: images/val
test: images/test

names:
  0: Tooth_Upper_Anterior
  1: Tooth_Upper_Posterior
  2: Tooth_Lower_Anterior
  3: Tooth_Lower_Posterior
  4: Orthodontic_Bracket
""")
            
    train_yolo_segmentation(
        dataset_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        model_size=args.model_size
    )
