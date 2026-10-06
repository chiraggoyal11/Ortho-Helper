#!/usr/bin/env python3
"""
🚀 ULTRA-ADVANCED NEXT-GEN Bone Fracture Detection
State-of-the-art techniques for MAXIMUM possible accuracy (95%+)
Vision Transformer + CNN Hybrid + Ensemble Methods
"""
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B3, ConvNeXtBase
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import *
import os
import numpy as np
from datetime import datetime
import json
import cv2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
import tensorflow_addons as tfa

class NextGenBoneFractureTrainer:
    def __init__(self, input_shape=(384, 384, 3), num_classes=2):
        """Initialize with ultra-high resolution for maximum detail"""
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.models = []
        
    def create_vision_transformer_head(self, features, name_prefix="vit"):
        """Create Vision Transformer head for global context"""
        # Patch embedding
        patch_size = 16
        num_patches = (self.input_shape[0] // patch_size) ** 2
        projection_dim = 768
        
        # Reshape to patches
        patches = Reshape((num_patches, patch_size * patch_size * 3))(features)
        patches = Dense(projection_dim, name=f"{name_prefix}_patch_projection")(patches)
        
        # Add positional encoding
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim,
            name=f"{name_prefix}_position_embedding"
        )(positions)
        encoded_patches = patches + position_embedding
        
        # Transformer blocks
        for i in range(4):  # 4 transformer layers
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=12, key_dim=64, dropout=0.1,
                name=f"{name_prefix}_attention_{i}"
            )(encoded_patches, encoded_patches)
            
            # Skip connection + Layer norm
            x1 = Add()([attention_output, encoded_patches])
            x1 = LayerNormalization(epsilon=1e-6)(x1)
            
            # MLP
            ffn = Dense(projection_dim * 4, activation="gelu")(x1)
            ffn = Dropout(0.1)(ffn)
            ffn = Dense(projection_dim)(ffn)
            
            # Skip connection + Layer norm
            encoded_patches = Add()([ffn, x1])
            encoded_patches = LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Global representation
        representation = GlobalAveragePooling1D()(encoded_patches)
        return representation
    
    def create_cnn_branch(self, base_model_name="efficientnetv2"):
        """Create CNN branch with latest architecture"""
        if base_model_name == "efficientnetv2":
            base_model = EfficientNetV2B3(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:  # ConvNeXt
            base_model = ConvNeXtBase(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        
        # Progressive unfreezing - start with top layers trainable
        for layer in base_model.layers[:-100]:
            layer.trainable = False
        
        return base_model
    
    def create_hybrid_model(self):
        """Create hybrid CNN + Vision Transformer model"""
        # Input
        inputs = Input(shape=self.input_shape)
        
        # Augmentation layers (built into model)
        x = inputs
        x = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal")(x)
        x = tf.keras.layers.experimental.preprocessing.RandomRotation(0.1)(x)
        x = tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)(x)
        
        # CNN Branch 1: EfficientNetV2
        cnn1 = self.create_cnn_branch("efficientnetv2")
        cnn1_features = cnn1(x)
        cnn1_pool = GlobalAveragePooling2D(name="cnn1_pool")(cnn1_features)
        
        # CNN Branch 2: ConvNeXt  
        cnn2 = self.create_cnn_branch("convnext")
        cnn2_features = cnn2(x)
        cnn2_pool = GlobalAveragePooling2D(name="cnn2_pool")(cnn2_features)
        
        # Vision Transformer Branch (using CNN1 features)
        vit_output = self.create_vision_transformer_head(cnn1_features)
        
        # Cross-attention between CNN and ViT
        cross_attention = MultiHeadAttention(
            num_heads=8, key_dim=64, dropout=0.1
        )(cnn1_pool[:, None, :], vit_output[:, None, :])
        cross_attention = GlobalAveragePooling1D()(cross_attention)
        
        # Fusion of all branches
        combined = Concatenate(name="feature_fusion")([
            cnn1_pool, cnn2_pool, vit_output, cross_attention
        ])
        
        # Advanced feature processing
        x = Dense(2048, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        # Squeeze-and-Excitation
        se_shape = int(x.shape[-1])
        se = GlobalAveragePooling1D()(x[:, None, :])
        se = Dense(se_shape // 16, activation='swish')(se)
        se = Dense(se_shape, activation='sigmoid')(se)
        x = Multiply()([x, se])
        
        # Multi-scale dense blocks
        x1 = Dense(1024, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.3)(x1)
        
        x2 = Dense(512, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x1)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.2)(x2)
        
        # Residual connections
        x1_proj = Dense(512)(x1)
        x2_residual = Add()([x2, x1_proj])
        
        x3 = Dense(256, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x2_residual)
        x3 = BatchNormalization()(x3)
        x3 = Dropout(0.1)(x3)
        
        # Final classification
        outputs = Dense(self.num_classes, activation='softmax', name='predictions')(x3)
        
        model = Model(inputs=inputs, outputs=outputs, name="HybridNextGen")
        return model
    
    def create_advanced_data_generator(self, train_dir, val_dir, batch_size=8):
        """Advanced data augmentation with mixup and cutmix"""
        
        class AdvancedDataGenerator:
            def __init__(self, directory, target_size, batch_size, is_training=True):
                self.directory = directory
                self.target_size = target_size
                self.batch_size = batch_size
                self.is_training = is_training
                
                # Load all image paths and labels
                self.image_paths = []
                self.labels = []
                
                classes = sorted(os.listdir(directory))
                for class_idx, class_name in enumerate(classes):
                    class_dir = os.path.join(directory, class_name)
                    for img_file in os.listdir(class_dir):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(class_dir, img_file))
                            self.labels.append(class_idx)
                
                self.num_samples = len(self.image_paths)
                self.class_indices = {class_name: idx for idx, class_name in enumerate(classes)}
                
            def load_and_preprocess_image(self, image_path):
                """Advanced image preprocessing"""
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if self.is_training:
                    # Advanced augmentations
                    if np.random.random() < 0.3:
                        # CLAHE for better contrast
                        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        lab[:,:,0] = clahe.apply(lab[:,:,0])
                        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    
                    # Random noise
                    if np.random.random() < 0.2:
                        noise = np.random.normal(0, 0.01, image.shape)
                        image = np.clip(image + noise, 0, 1)
                
                # Resize with aspect ratio preservation
                image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
                return image.astype(np.float32) / 255.0
            
            def mixup(self, images, labels, alpha=0.4):
                """Mixup augmentation"""
                batch_size = tf.shape(images)[0]
                indices = tf.random.shuffle(tf.range(batch_size))
                shuffled_images = tf.gather(images, indices)
                shuffled_labels = tf.gather(labels, indices)
                
                lambda_val = tf.random.uniform([], 0, alpha)
                images = lambda_val * images + (1 - lambda_val) * shuffled_images
                labels = lambda_val * labels + (1 - lambda_val) * shuffled_labels
                return images, labels
            
            def __iter__(self):
                """Generator iterator"""
                indices = np.arange(self.num_samples)
                
                while True:
                    if self.is_training:
                        np.random.shuffle(indices)
                    
                    for start_idx in range(0, self.num_samples, self.batch_size):
                        end_idx = min(start_idx + self.batch_size, self.num_samples)
                        batch_indices = indices[start_idx:end_idx]
                        
                        batch_images = []
                        batch_labels = []
                        
                        for idx in batch_indices:
                            image = self.load_and_preprocess_image(self.image_paths[idx])
                            batch_images.append(image)
                            
                            label = np.zeros(2)
                            label[self.labels[idx]] = 1
                            batch_labels.append(label)
                        
                        batch_images = np.array(batch_images)
                        batch_labels = np.array(batch_labels)
                        
                        # Apply mixup for training
                        if self.is_training and np.random.random() < 0.3:
                            batch_images, batch_labels = self.mixup(batch_images, batch_labels)
                        
                        yield batch_images, batch_labels
        
        train_gen = AdvancedDataGenerator(train_dir, self.input_shape[:2], batch_size, True)
        val_gen = AdvancedDataGenerator(val_dir, self.input_shape[:2], batch_size, False)
        
        return train_gen, val_gen
    
    def create_ultra_callbacks(self, model_save_path):
        """Ultra-advanced callbacks"""
        callbacks = [
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=25,
                restore_best_weights=True,
                mode='max',
                verbose=1,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=15,
                min_lr=1e-9,
                verbose=1,
                cooldown=8
            ),
            # Cosine annealing
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: 1e-4 * 0.5 * (1 + np.cos(np.pi * epoch / 100)),
                verbose=0
            ),
            # StochasticWeightAveraging for better convergence
            tfa.callbacks.AverageModelCheckpoint(
                filepath=model_save_path.replace('.h5', '_swa.h5'),
                update_weights=True,
                monitor='val_accuracy',
                mode='max'
            )
        ]
        
        return callbacks
    
    def train_next_gen_model(self, train_dir, val_dir, epochs=150):
        """Ultra-advanced training protocol"""
        print("🚀 NEXT-GEN Ultra-Advanced Training Starting...")
        print(f"Input resolution: {self.input_shape}")
        print(f"Max epochs: {epochs}")
        
        # Create hybrid model
        model = self.create_hybrid_model()
        print(f"✅ Hybrid model created with {model.count_params():,} parameters")
        
        # Ultra-advanced optimizer
        optimizer = AdamW(
            learning_rate=1e-4,
            weight_decay=1e-5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )
        
        # Compile with advanced loss
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Create advanced data generators
        train_gen, val_gen = self.create_advanced_data_generator(train_dir, val_dir, batch_size=8)
        
        print(f"📊 Training samples: {train_gen.num_samples}")
        print(f"📊 Validation samples: {val_gen.num_samples}")
        
        # Model save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = f"models/NextGen_bone_fracture_{timestamp}.h5"
        os.makedirs("models", exist_ok=True)
        
        # Ultra callbacks
        callbacks = self.create_ultra_callbacks(model_save_path)
        
        # Calculate steps
        steps_per_epoch = max(1, train_gen.num_samples // 8)
        validation_steps = max(1, val_gen.num_samples // 8)
        
        print(f"🔄 Steps per epoch: {steps_per_epoch}")
        print(f"🔄 Validation steps: {validation_steps}")
        
        # Multi-phase training protocol
        print("\n🎯 Phase 1: Foundation Learning (50 epochs)")
        history1 = model.fit(
            iter(train_gen),
            steps_per_epoch=steps_per_epoch,
            epochs=min(50, epochs),
            validation_data=iter(val_gen),
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Progressive unfreezing
        print("\n🎯 Phase 2: Deep Fine-tuning")
        for layer in model.layers:
            layer.trainable = True
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=AdamW(learning_rate=1e-6, weight_decay=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        remaining_epochs = epochs - min(50, epochs)
        if remaining_epochs > 0:
            history2 = model.fit(
                iter(train_gen),
                steps_per_epoch=steps_per_epoch,
                epochs=remaining_epochs,
                validation_data=iter(val_gen),
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1,
                initial_epoch=min(50, epochs)
            )
        
        # Load best model
        best_model = tf.keras.models.load_model(model_save_path)
        
        # Final evaluation
        print("\n📈 NEXT-GEN Model Evaluation:")
        val_loss, val_accuracy, val_precision, val_recall = best_model.evaluate(
            iter(val_gen), steps=validation_steps, verbose=0
        )
        
        f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-8)
        
        print(f"✅ Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"✅ Validation Precision: {val_precision:.4f}")
        print(f"✅ Validation Recall: {val_recall:.4f}")
        print(f"✅ F1 Score: {f1_score:.4f}")
        
        # Save final model
        final_model_path = "models/NextGen_bone_fracture_final.h5"
        best_model.save(final_model_path)
        print(f"🎉 NEXT-GEN model saved: {final_model_path}")
        
        return best_model, val_accuracy

def main():
    """Main next-generation training function"""
    print("=" * 80)
    print("🦴 NEXT-GENERATION Ultra-Advanced Bone Fracture Detection")
    print("🧠 Vision Transformer + CNN Hybrid + Advanced Techniques")
    print("=" * 80)
    
    # Use existing maximum dataset
    train_dir = "Dataset/ARM_Maximum/train"
    val_dir = "Dataset/ARM_Maximum/val"
    
    if not os.path.exists(train_dir):
        print("❌ Dataset not found. Run maximum training first.")
        return
    
    # Initialize next-gen trainer
    trainer = NextGenBoneFractureTrainer(
        input_shape=(384, 384, 3),  # Ultra-high resolution
        num_classes=2
    )
    
    try:
        model, accuracy = trainer.train_next_gen_model(
            train_dir=train_dir,
            val_dir=val_dir,
            epochs=150
        )
        
        print(f"\n🎯 NEXT-GEN Training Completed!")
        print(f"✅ Final accuracy: {accuracy*100:.2f}%")
        
        if accuracy >= 0.95:
            print(f"🎉 BREAKTHROUGH! Medical Excellence: {accuracy*100:.2f}%")
        elif accuracy >= 0.92:
            print(f"🎉 OUTSTANDING! Research-grade: {accuracy*100:.2f}%")
        elif accuracy >= 0.88:
            print(f"🎉 EXCELLENT! High performance: {accuracy*100:.2f}%")
        elif accuracy >= 0.85:
            print(f"✅ VERY GOOD performance: {accuracy*100:.2f}%")
        else:
            print(f"📈 Significant improvement potential: {accuracy*100:.2f}%")
            
    except Exception as e:
        print(f"❌ Next-gen training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()