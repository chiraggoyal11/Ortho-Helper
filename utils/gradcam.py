import numpy as np
import cv2
import tensorflow as tf
from PIL import Image


class GradCAM:
    """
    Genuine Grad-CAM implementation for the nested DenseNet121 bone model.

    Target:
        densenet121 / conv5_block16_2_conv -> (7, 7, 32)

    No synthetic/fallback heatmaps are generated.
    """

    def __init__(self, model, layer_name="conv5_block16_2_conv"):
        self.model = model
        self.layer_name = layer_name

        self.base_model = None
        self.target_layer = None
        self.grad_model = None

        self._build_nested_grad_model()

    def _build_nested_grad_model(self):
        # The bone model contains DenseNet121 as a nested Functional model.
        try:
            self.base_model = self.model.get_layer("densenet121")
        except Exception:
            self.base_model = None

        if self.base_model is None:
            raise ValueError(
                "Grad-CAM requires a nested layer named 'densenet121'."
            )

        try:
            self.target_layer = self.base_model.get_layer(self.layer_name)
        except Exception as exc:
            raise ValueError(
                f"Grad-CAM target layer '{self.layer_name}' was not found "
                f"inside DenseNet121."
            ) from exc

        # IMPORTANT:
        # Build this model entirely inside the DenseNet graph.
        # Do NOT combine self.model.input with target_layer.output.
        self.grad_model = tf.keras.Model(
            inputs=self.base_model.input,
            outputs=[
                self.target_layer.output,
                self.base_model.output,
            ],
        )

    def _classification_score(self, features, class_index=None):
        """
        Apply the actual outer classification head to DenseNet features.

        Actual saved model architecture:
            DenseNet121
            -> GlobalAveragePooling2D
            -> BatchNormalization
            -> Dropout
            -> Dense(256)
            -> BatchNormalization
            -> Dropout
            -> Dense(1, sigmoid)
        """

        x = self.model.get_layer("global_average_pooling2d")(features)

        x = self.model.get_layer("batch_normalization")(
            x, training=False
        )

        x = self.model.get_layer("dropout")(
            x, training=False
        )

        x = self.model.get_layer("dense")(x)

        x = self.model.get_layer("batch_normalization_1")(
            x, training=False
        )

        x = self.model.get_layer("dropout_1")(
            x, training=False
        )

        prediction = self.model.get_layer("dense_1")(x)

        # Saved model has a single sigmoid output.
        prediction = tf.reshape(prediction, [-1])

        if class_index is None:
            # Positive/abnormal class.
            return prediction

        if int(class_index) == 1:
            # Positive/abnormal class.
            return prediction

        if int(class_index) == 0:
            # Normal class = 1 - abnormal probability.
            return 1.0 - prediction

        raise ValueError(
            f"Invalid class_index={class_index} for binary sigmoid model."
        )

    def generate_heatmap(
        self,
        image_array,
        class_index=None,
        alpha=0.4,
    ):
        """
        Generate genuine Grad-CAM.

        Returns:
            float32 heatmap with shape (7, 7), values in [0, 1].
        """

        image = np.asarray(image_array, dtype=np.float32)

        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        if image.ndim != 4:
            raise ValueError(
                f"Expected image shape (H,W,C) or (1,H,W,C), got {image.shape}"
            )

        if image.shape[-1] != 3:
            raise ValueError(
                f"Expected 3-channel image, got {image.shape}"
            )

        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, dense_features = self.grad_model(
                image_tensor, training=False
            )

            score = self._classification_score(
                dense_features,
                class_index=class_index,
            )

        grads = tape.gradient(score, conv_outputs)

        if grads is None:
            raise RuntimeError(
                "Grad-CAM gradients are None. "
                "The target layer is not connected to the classification output."
            )

        if conv_outputs.shape.rank != 4:
            raise RuntimeError(
                f"Target activation must be 4-D, got {conv_outputs.shape}"
            )

        # Grad-CAM:
        # average gradients over spatial dimensions
        weights = tf.reduce_mean(
            grads,
            axis=(1, 2),
            keepdims=True,
        )

        # Weighted combination of feature maps
        cam = tf.reduce_sum(
            conv_outputs * weights,
            axis=-1,
        )

        # Keep only positive evidence
        cam = tf.maximum(cam, 0.0)

        # Remove batch dimension
        cam = cam[0]

        max_value = tf.reduce_max(cam)

        if float(max_value.numpy()) <= 1e-12:
            raise RuntimeError(
                "Grad-CAM produced an all-zero heatmap."
            )

        cam = cam / (max_value + tf.keras.backend.epsilon())

        heatmap = cam.numpy().astype(np.float32)

        # Strict validation: this must be genuine 7x7 Grad-CAM.
        if heatmap.shape != (7, 7):
            raise RuntimeError(
                f"Unexpected Grad-CAM shape: {heatmap.shape}; expected (7, 7)."
            )

        return np.clip(heatmap, 0.0, 1.0)

    def detect_concern_regions(
        self,
        heatmap,
        condition_name="Condition",
        threshold=0.6,
    ):
        """
        Detect high-activation regions.

        Note:
        These are attention regions, NOT medically validated fracture
        bounding boxes.
        """

        heatmap = np.asarray(heatmap, dtype=np.float32)

        if heatmap.size == 0:
            return []

        mask = (heatmap >= threshold).astype(np.uint8)

        # Work at heatmap resolution.
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        regions = []

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            if area < 2:
                continue

            regions.append({
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "area": int(area),
                "confidence": float(np.max(heatmap[labels == i])),
                "condition": condition_name,
            })

        return sorted(
            regions,
            key=lambda r: r["confidence"],
            reverse=True,
        )

    def create_superimposed_image(
        self,
        image,
        heatmap,
        alpha=0.4,
        colormap=cv2.COLORMAP_JET,
        diagnosis_result=None,
        condition_name="Fracture",
        **kwargs,
    ):
        """
        Create the visualization used by the Streamlit app.
        """

        if isinstance(image, Image.Image):
            original = np.array(image.convert("RGB"))
        else:
            original = np.asarray(image)

            if original.ndim == 2:
                original = cv2.cvtColor(
                    original.astype(np.uint8),
                    cv2.COLOR_GRAY2RGB,
                )

            if original.shape[-1] == 4:
                original = original[:, :, :3]

        original = np.clip(original, 0, 255).astype(np.uint8)

        heatmap = np.asarray(heatmap, dtype=np.float32)

        if heatmap.ndim != 2:
            raise ValueError(
                f"Expected 2-D heatmap, got {heatmap.shape}"
            )

        heatmap_resized = cv2.resize(
            heatmap,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        heatmap_uint8 = np.uint8(
            np.clip(heatmap_resized, 0.0, 1.0) * 255
        )

        colored = cv2.applyColorMap(
            heatmap_uint8,
            colormap,
        )

        colored = cv2.cvtColor(
            colored,
            cv2.COLOR_BGR2RGB,
        )

        overlay = cv2.addWeighted(
            original,
            1.0 - alpha,
            colored,
            alpha,
            0,
        )

        return Image.fromarray(overlay)

    # Compatibility alias used by some older code.
    def create_overlay(self, image, heatmap, alpha=0.4, **kwargs):
        return self.create_superimposed_image(
            image,
            heatmap,
            alpha=alpha,
            **kwargs,
        )


def generate_gradcam_heatmap(
    model,
    image_array,
    class_index=None,
    model_type="bone",
    intensity=0.4,
    diagnosis_result=None,
    condition_name="Fracture",
    **kwargs,
):
    """
    Public API used by app.py.

    Returns a PIL image containing genuine Grad-CAM.
    """

    gradcam = GradCAM(model)

    heatmap = gradcam.generate_heatmap(
        image_array,
        class_index=class_index,
    )

    return gradcam.create_superimposed_image(
        image_array,
        heatmap,
        alpha=float(intensity),
        diagnosis_result=diagnosis_result,
        condition_name=condition_name,
    )
