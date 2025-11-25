from ultralytics import YOLO
import torch

def main():
    # 1. Hardware Check
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Starting training on: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
    
    # 2. Load the Model
    # We use 'yolo11m.pt' (Medium) for the best balance on GTX 1650.
    # It will download automatically on the first run.
    model = YOLO('yolo11m.pt') 

    # 3. Train with "Sim-to-Real" Hyperparameters
    results = model.train(
        # --- Data & Storage ---
        data='data.yaml',
        project='AVS325_Submission',
        name='SpaceSafety_Model',
        
        # --- Hardware Constraints (GTX 1650 optimized) ---
        imgsz=640,       # Do not go higher or VRAM will overflow
        batch=4,         # Safe batch size for 4GB VRAM. Try 8 if stable.
        workers=2,       # Lower workers to save system RAM
        device=device,
        
        # --- Training Length ---
        epochs=100,      # Synthetic data needs time to converge
        patience=20,     # Stop if no improvement for 20 epochs
        
        # --- The "Sim-to-Real" Augmentation Strategy ---
        # These params "dirty" the clean synthetic data
        hsv_h=0.015,     # Hue shift (handle strange space lights)
        hsv_s=0.7,       # Saturation shift (handle vivid vs dull assets)
        hsv_v=0.4,       # Value/Brightness shift (CRITICAL for space lighting)
        degrees=15.0,    # Rotation (objects float in space)
        flipud=0.5,      # Flip Upside Down (no gravity in space)
        fliplr=0.5,      # Flip Left/Right
        
        # --- Occlusion Handling ---
        mosaic=1.0,      # 100% chance to combine images (handles occlusion)
        mixup=0.15,      # Blend images together (transparency/confusion)
        
        # --- Refinement Phase ---
        close_mosaic=15, # Turn OFF mosaic for the last 15 epochs to see "real" scenes
        
        # --- Advanced Loss Tuning ---
        box=7.5,         # Increase box loss gain (focus on precise bounding boxes)
        cls=0.5,         # Class loss gain
    )

    # 4. Export the Model
    # This 'best.pt' is what you submit and use in the app
    print("✅ Training Complete. Best model saved in /AVS325_Submission/SpaceSafety_Model/weights/best.pt")

if __name__ == '__main__':
    main()