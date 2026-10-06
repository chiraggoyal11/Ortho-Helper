#!/usr/bin/env python3
"""
Enhanced Streamlit Training Interface with Resume Functionality
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import json
import os
from pathlib import Path
import time
from typing import Dict, Any

class StreamlitTrainingWithResume:
    """Enhanced training interface with resume capability"""
    
    def __init__(self):
        self.model_dir = Path("models")
        self.checkpoint_dir = self.model_dir / "checkpoints"
        
    def check_existing_checkpoints(self, dataset_name: str, architecture: str) -> Dict[str, Any]:
        """Check for existing training checkpoints"""
        checkpoint_path = self.checkpoint_dir / f"{dataset_name}_{architecture}"
        
        checkpoint_info = {
            'exists': False,
            'model_path': None,
            'history_path': None,
            'can_resume': False,
            'completed_epochs': 0
        }
        
        if checkpoint_path.exists():
            model_path = checkpoint_path / "best_model.h5"
            history_path = self.model_dir / f"{dataset_name}_{architecture}_history.json"
            
            if model_path.exists():
                checkpoint_info['exists'] = True
                checkpoint_info['model_path'] = str(model_path)
                checkpoint_info['can_resume'] = True
                
                # Check training history for completed epochs
                if history_path.exists():
                    try:
                        with open(history_path, 'r') as f:
                            history = json.load(f)
                        checkpoint_info['completed_epochs'] = len(history.get('loss', []))
                        checkpoint_info['history_path'] = str(history_path)
                    except:
                        pass
        
        return checkpoint_info
    
    def display_resume_options(self, checkpoint_info: Dict[str, Any], dataset_name: str, architecture: str):
        """Display resume training options"""
        
        if checkpoint_info['exists']:
            st.success(f"✅ **Found existing checkpoint for {dataset_name} + {architecture}**")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info(f"""
                📊 **Checkpoint Details:**
                - **Completed Epochs:** {checkpoint_info['completed_epochs']}
                - **Model File:** {Path(checkpoint_info['model_path']).name}
                - **Status:** Ready to resume or use as final model
                """)
            
            with col2:
                st.markdown("### 🎯 **Options:**")
                
            # Action buttons
            st.markdown("### 🚀 **Choose Action:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📁 **Use as Final Model**", key="use_final", help="Copy checkpoint to main model directory"):
                    self.use_checkpoint_as_final(checkpoint_info, dataset_name)
                    
            with col2:
                if st.button("🔄 **Resume Training**", key="resume_train", help="Continue training from checkpoint"):
                    return "resume"
                    
            with col3:
                if st.button("🆕 **Start Fresh**", key="start_fresh", help="Start new training (keeps backup)"):
                    return "fresh"
            
            return "options_shown"
        
        else:
            st.info(f"🆕 **No existing checkpoint found for {dataset_name} + {architecture}**")
            return "no_checkpoint"
    
    def use_checkpoint_as_final(self, checkpoint_info: Dict[str, Any], dataset_name: str):
        """Copy checkpoint to main model directory"""
        try:
            source_path = checkpoint_info['model_path']
            dest_path = self.model_dir / f"{dataset_name}_model.h5"
            
            # Copy model
            import shutil
            shutil.copy2(source_path, dest_path)
            
            st.success(f"✅ **Model Ready!** Checkpoint copied to: `{dest_path.name}`")
            st.balloons()
            
            # Show model info
            try:
                model = keras.models.load_model(str(dest_path))
                st.info(f"""
                🎯 **Model Information:**
                - **Total Parameters:** {model.count_params():,}
                - **Input Shape:** {model.input.shape}
                - **Output Classes:** {model.output.shape[-1]}
                - **Ready for inference!** 🚀
                """)
            except Exception as e:
                st.warning(f"Model copied but couldn't load info: {e}")
                
        except Exception as e:
            st.error(f"❌ Error copying model: {e}")
    
    def resume_training_from_checkpoint(self, checkpoint_info: Dict[str, Any], 
                                      dataset_name: str, architecture: str, 
                                      additional_epochs: int = 10):
        """Resume training from checkpoint"""
        
        st.markdown("### 🔄 **Resuming Training**")
        
        try:
            # Load existing model
            with st.spinner("📂 Loading checkpoint model..."):
                model = keras.models.load_model(checkpoint_info['model_path'])
            
            st.success("✅ Model loaded successfully!")
            
            # Show current progress
            completed = checkpoint_info['completed_epochs']
            total_new = completed + additional_epochs
            
            st.info(f"""
            📊 **Resume Training Plan:**
            - **Previously Completed:** {completed} epochs
            - **Additional Epochs:** {additional_epochs} 
            - **New Total:** {total_new} epochs
            """)
            
            # Setup new callbacks for continued training
            checkpoint_path = Path(checkpoint_info['model_path']).parent
            
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    filepath=str(checkpoint_path / "resumed_best_model.h5"),
                    monitor='val_accuracy',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=1
                ),
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=0.00001,
                    verbose=1
                )
            ]
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Note: You would need to load your dataset here
            st.warning("""
            ⚠️ **Dataset Loading Required:**
            To complete resume training, you need to:
            1. Load your training and validation datasets
            2. Call model.fit() with initial_epoch parameter
            3. Use the loaded model and callbacks above
            
            Example:
            ```python
            model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs={total_new},
                initial_epoch={completed},
                callbacks=callbacks
            )
            ```
            """)
            
            return model, callbacks
            
        except Exception as e:
            st.error(f"❌ Error resuming training: {e}")
            return None, None

def enhanced_training_interface():
    """Enhanced training interface with resume functionality"""
    
    st.markdown("## 🚀 **Enhanced Model Training with Resume**")
    
    trainer = StreamlitTrainingWithResume()
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        architecture = st.selectbox(
            "🏗️ **Model Architecture:**",
            ["DenseNet121", "ResNet50", "EfficientNetB0"],
            help="Choose the neural network architecture"
        )
    
    with col2:
        dataset_name = st.selectbox(
            "📊 **Dataset:**",
            ["knee_conditions", "bone_fracture", "chest_conditions"],
            help="Choose the dataset to train on"
        )
    
    # Check for existing checkpoints
    checkpoint_info = trainer.check_existing_checkpoints(dataset_name, architecture)
    
    # Display resume options or start fresh
    action = trainer.display_resume_options(checkpoint_info, dataset_name, architecture)
    
    if action == "resume":
        st.markdown("---")
        
        additional_epochs = st.number_input(
            "🔢 **Additional epochs to train:**",
            min_value=1, max_value=50, value=10,
            help="How many more epochs to train"
        )
        
        if st.button("▶️ **Start Resume Training**", type="primary"):
            model, callbacks = trainer.resume_training_from_checkpoint(
                checkpoint_info, dataset_name, architecture, additional_epochs
            )
            
            if model is not None:
                st.success("🎯 **Ready to resume!** Load your dataset and call model.fit()")
    
    elif action == "fresh":
        st.markdown("---")
        st.info("🆕 **Starting fresh training** - previous checkpoint will be backed up")
        # Here you would call the regular training interface

def main():
    """Main function for testing"""
    st.set_page_config(
        page_title="Enhanced Training Interface",
        page_icon="🚀",
        layout="wide"
    )
    
    enhanced_training_interface()

if __name__ == "__main__":
    main()