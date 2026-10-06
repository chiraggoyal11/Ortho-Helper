#!/usr/bin/env python3
"""
Resume Training Script - Continue from checkpoint
"""

import tensorflow as tf
from tensorflow import keras
import os
from pathlib import Path

def resume_knee_training(checkpoint_path, additional_epochs=10):
    """
    Resume training from checkpoint
    
    Args:
        checkpoint_path: Path to saved checkpoint model
        additional_epochs: Number of additional epochs to train
    """
    
    print(f"🔄 Resuming training from checkpoint: {checkpoint_path}")
    
    # Load the saved model
    try:
        model = keras.models.load_model(checkpoint_path)
        print("✅ Model loaded successfully!")
        print(f"📊 Model architecture: {model.input.shape} -> {model.output.shape}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Print current model performance (if available)
    try:
        # You might want to evaluate on validation set here
        print("📈 Loaded model ready for continued training")
    except Exception as e:
        print(f"⚠️ Note: {e}")
    
    # Setup callbacks for continued training
    checkpoint_dir = Path(checkpoint_path).parent
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "continued_best_model.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,  # Reduced patience for resumed training
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # More aggressive LR reduction
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    print(f"🎯 Ready to continue training for {additional_epochs} more epochs")
    print("💡 Remember to:")
    print("   1. Load your training data")
    print("   2. Set up data generators") 
    print("   3. Call model.fit() with the loaded model")
    print("   4. Use the callbacks defined above")
    
    return model, callbacks

def quick_resume_example():
    """Example of how to use the resume function"""
    
    # Path to your checkpoint
    checkpoint_path = "models/checkpoints/knee_conditions_DenseNet121/best_model.h5"
    
    if os.path.exists(checkpoint_path):
        print(f"✅ Checkpoint found: {checkpoint_path}")
        
        # Resume training
        model, callbacks = resume_knee_training(checkpoint_path, additional_epochs=20)
        
        if model:
            print("\n🚀 To continue training, use this model with your data:")
            print(f"model.fit(")
            print(f"    train_dataset,")
            print(f"    validation_data=val_dataset,")
            print(f"    epochs=20,  # Additional epochs")
            print(f"    callbacks=callbacks,")
            print(f"    initial_epoch=21  # Start from epoch 21")
            print(f")")
            
    else:
        print(f"❌ Checkpoint not found at: {checkpoint_path}")
        print("Available checkpoints:")
        checkpoint_base = Path("models/checkpoints")
        if checkpoint_base.exists():
            for item in checkpoint_base.rglob("*.h5"):
                print(f"   📁 {item}")

if __name__ == "__main__":
    quick_resume_example()