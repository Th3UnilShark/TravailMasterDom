from fastai.vision.all import *
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================

# UPDATE THIS PATH to point to your folder containing 'train' and 'valid' subfolders
# Example: "C:/Users/Admin/Documents/TravailMasterDom/my_cats_dogs"
# Or relative path: "./my_cats_dogs"
path = Path("./stab_data") 

# Check if the path exists to prevent immediate crashes
if not path.exists():
    raise FileNotFoundError(f"Path not found: {path}. Please create the folder structure as described in the comments.")

# ==========================================
# 2. DATA LOADING
# ==========================================

# We use from_folder because our data is organized in subdirectories (cat/, dog/)
# valid_pct is set to 0.2 here because we want to save 20% of data for validation, 
# The seed ensures reproducibility if we were splitting automatically.
dls = ImageDataLoaders.from_folder(
    path, 
    valid_pct=0.2,  # If you only have one folder, this splits it. If you have train/valid, this is ignored.
    seed=42,        # Random seed for reproducibility
    item_tfms=Resize(448), # Resize every image to 224x224 pixels (standard for ResNet)
    batch_tfms=aug_transforms(), # Apply random augmentations (flip, rotate, zoom) to training batches only
    num_workers=0  # <--- CRITICAL CHANGE: Disables worker processes
)

# Debug: Print out what the dataloaders think the classes are
print(f"Classes detected: {dls.vocab}")
print(f"Number of training batches: {len(dls.train)}")
print(f"Number of validation batches: {len(dls.valid)}")

# ==========================================
# 3. MODEL CREATION
# ==========================================

# Create a vision learner using ResNet34 architecture
# metrics=[accuracy, error_rate] gives you both success and failure rates
learn = vision_learner(dls, resnet34, metrics=[accuracy, error_rate])

# Evaluate the model BEFORE any training to establish a baseline.
print("\n" + "="*50)
print("📊 STEP 3a: Evaluating Baseline (Pre-Training)")
print("="*50)

# Run validation (this updates learn.recorder with metrics)
learn.validate()

# Access metrics from the recorder
# learn.recorder.metrics is a list of AvgMetric objects
# Each AvgMetric has a .value property that returns a Tensor
# We need .item() to convert Tensor to Python float
if hasattr(learn.recorder, 'metrics') and len(learn.recorder.metrics) > 0:
    metrics_list = learn.recorder.metrics
    
    # First metric is always loss
    baseline_loss = metrics_list[0].value.item()
    
    # Second metric is typically accuracy (but verify by checking length)
    if len(metrics_list) > 1:
        baseline_accuracy = metrics_list[1].value.item()
        # Error rate should be 1 - accuracy
        baseline_error = 1 - baseline_accuracy
    else:
        baseline_accuracy = 0
        baseline_error = 0
    
    print(f"Baseline Accuracy (Pre-trained): {baseline_accuracy * 100:.2f}%")
    print(f"Baseline Error Rate:             {baseline_error * 100:.2f}%")
    print(f"Baseline Loss:                   {baseline_loss:.4f}")
else:
    print("Could not retrieve metrics from recorder.")

print("Note: This reflects the model's performance using only ImageNet weights.")
print("-"*50)

# Optional: Show a summary of the model architecture
# learn.summary() 

# ==========================================
# 4. TRAINING
# ==========================================


print("Starting training...")

# Fine-tune the model:
# 1. Freeze the base layers (ResNet) and train only the new head (1 epoch)
# 2. Unfreeze and train the whole network for 1 epoch (you can increase this to 3-5)
# Note: fine_tune(1) means 1 epoch of full training after freezing.
# If you want more training, change to learn.fine_tune(3) or learn.fit_one_cycle(5)
learn.fine_tune(15)

# Save the model
learn.export("cat_dog_model.pkl")
print("Training complete. Model saved as 'cat_dog_model.pkl'")

# Optional: Test the model on a single image
# Replace 'test_image.jpg' with a path to a new image
# pred_class, pred_idx, outputs = learn.predict("test_image.jpg")
# print(f"Prediction: {pred_class}")