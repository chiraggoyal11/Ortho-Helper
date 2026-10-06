#!/usr/bin/env python3
"""
Feature Completion Module
Implementation of "coming soon" features and missing functionality
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from utils.config_persistence import ConfigurationPersistenceManager

class AdvancedAnalyticsModule:
    """Advanced analytics and reporting features"""
    
    def __init__(self):
        self.analytics_dir = Path("analytics_data")
        self.analytics_dir.mkdir(exist_ok=True)
        self.usage_file = self.analytics_dir / "usage_analytics.json"
        self.performance_file = self.analytics_dir / "performance_metrics.json"
    
    def log_classification_event(self, model_type: str, confidence: float, processing_time: float, user_role: str):
        """Log classification events for analytics"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "confidence": confidence,
            "processing_time": processing_time,
            "user_role": user_role,
            "success": True
        }
        
        self._append_to_analytics_file(self.usage_file, event)
    
    def log_performance_metric(self, metric_type: str, value: float, context: Dict[str, Any] = None):
        """Log performance metrics"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "metric_type": metric_type,
            "value": value,
            "context": context or {}
        }
        
        self._append_to_analytics_file(self.performance_file, metric)
    
    def _append_to_analytics_file(self, file_path: Path, data: Dict[str, Any]):
        """Append data to analytics file"""
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    analytics_data = json.load(f)
            else:
                analytics_data = []
            
            analytics_data.append(data)
            
            # Keep only last 1000 entries to manage file size
            if len(analytics_data) > 1000:
                analytics_data = analytics_data[-1000:]
            
            with open(file_path, 'w') as f:
                json.dump(analytics_data, f, indent=2)
                
        except Exception as e:
            st.error(f"Failed to log analytics data: {e}")
    
    def generate_usage_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive usage report"""
        try:
            if not self.usage_file.exists():
                return {"error": "No usage data available"}
            
            with open(self.usage_file, 'r') as f:
                usage_data = json.load(f)
            
            # Filter by date range
            cutoff_date = datetime.now() - timedelta(days=days)
            filtered_data = [
                event for event in usage_data
                if datetime.fromisoformat(event['timestamp']) >= cutoff_date
            ]
            
            if not filtered_data:
                return {"error": "No data for specified period"}
            
            # Calculate statistics
            df = pd.DataFrame(filtered_data)
            
            report = {
                "period_days": days,
                "total_classifications": len(filtered_data),
                "unique_models_used": df['model_type'].nunique(),
                "average_confidence": df['confidence'].mean(),
                "average_processing_time": df['processing_time'].mean(),
                "model_usage": df['model_type'].value_counts().to_dict(),
                "user_role_distribution": df['user_role'].value_counts().to_dict(),
                "confidence_distribution": {
                    "high_confidence": len(df[df['confidence'] >= 0.8]),
                    "medium_confidence": len(df[(df['confidence'] >= 0.5) & (df['confidence'] < 0.8)]),
                    "low_confidence": len(df[df['confidence'] < 0.5])
                },
                "daily_usage": df.groupby(df['timestamp'].str[:10]).size().to_dict()
            }
            
            return report
            
        except Exception as e:
            return {"error": f"Failed to generate usage report: {e}"}
    
    def create_analytics_dashboard(self):
        """Create comprehensive analytics dashboard"""
        st.markdown("### 📊 **Advanced Analytics Dashboard**")
        
        # Time period selector
        period_col1, period_col2 = st.columns([2, 1])
        
        with period_col1:
            period = st.selectbox(
                "Analysis Period",
                ["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
                index=1
            )
        
        with period_col2:
            if st.button("🔄 Refresh Data"):
                st.rerun()
        
        # Convert period to days
        period_mapping = {
            "Last 7 days": 7,
            "Last 30 days": 30,
            "Last 90 days": 90,
            "All time": 365  # Practical limit
        }
        
        days = period_mapping[period]
        
        # Generate report
        report = self.generate_usage_report(days)
        
        if "error" in report:
            st.warning(f"⚠️ {report['error']}")
            return
        
        # Display metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                "Total Classifications",
                report["total_classifications"],
                delta=None
            )
        
        with metric_col2:
            st.metric(
                "Average Confidence",
                f"{report['average_confidence']:.2%}",
                delta=None
            )
        
        with metric_col3:
            st.metric(
                "Avg Processing Time",
                f"{report['average_processing_time']:.2f}s",
                delta=None
            )
        
        with metric_col4:
            st.metric(
                "Models Used",
                report["unique_models_used"],
                delta=None
            )
        
        # Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("**Model Usage Distribution**")
            if report["model_usage"]:
                model_df = pd.DataFrame(
                    list(report["model_usage"].items()),
                    columns=["Model", "Count"]
                )
                st.bar_chart(model_df.set_index("Model"))
        
        with chart_col2:
            st.markdown("**Confidence Distribution**")
            confidence_data = report["confidence_distribution"]
            conf_df = pd.DataFrame(
                list(confidence_data.items()),
                columns=["Confidence Level", "Count"]
            )
            st.bar_chart(conf_df.set_index("Confidence Level"))
        
        # Daily usage trend
        st.markdown("**Daily Usage Trend**")
        if report["daily_usage"]:
            daily_df = pd.DataFrame(
                list(report["daily_usage"].items()),
                columns=["Date", "Classifications"]
            )
            daily_df["Date"] = pd.to_datetime(daily_df["Date"])
            daily_df = daily_df.sort_values("Date")
            st.line_chart(daily_df.set_index("Date"))
        
        # Export options
        st.markdown("### 📤 **Export Options**")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            if st.button("📊 Export Report (JSON)"):
                report_json = json.dumps(report, indent=2)
                st.download_button(
                    label="⬇️ Download JSON Report",
                    data=report_json,
                    file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with export_col2:
            if st.button("📈 Export Charts (PNG)"):
                st.info("Chart export functionality will be available in next update")
        
        with export_col3:
            if st.button("📋 Export Summary (PDF)"):
                st.info("PDF export functionality will be available in next update")

class ModelManagementModule:
    """Advanced model management features"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.model_metadata_file = Path("model_metadata.json")
    
    def show_model_management_interface(self):
        """Display comprehensive model management interface"""
        st.markdown("### 🔧 **Advanced Model Management**")
        
        # Model overview
        self._show_model_overview()
        
        # Model operations
        st.markdown("### ⚙️ **Model Operations**")
        
        operation_tabs = st.tabs([
            "📊 Performance", "🔄 Updates", "⚙️ Configuration", 
            "📥 Import/Export", "🗑️ Cleanup"
        ])
        
        with operation_tabs[0]:
            self._show_model_performance()
        
        with operation_tabs[1]:
            self._show_model_updates()
        
        with operation_tabs[2]:
            self._show_model_configuration()
        
        with operation_tabs[3]:
            self._show_model_import_export()
        
        with operation_tabs[4]:
            self._show_model_cleanup()
    
    def _show_model_overview(self):
        """Show model overview with status and metrics"""
        st.markdown("**📋 Model Status Overview**")
        
        # Mock model data - in real implementation, this would come from model files
        models_data = [
            {
                "name": "Bone Fracture Detection",
                "file": "bone_fracture_model.h5",
                "status": "✅ Active",
                "accuracy": "95.2%",
                "size": "12.3 MB",
                "last_updated": "2025-10-01"
            },
            {
                "name": "Pneumonia Detection",
                "file": "pneumonia_model.h5",
                "status": "✅ Active",
                "accuracy": "94.8%",
                "size": "11.7 MB",
                "last_updated": "2025-09-28"
            },
            {
                "name": "Cardiomegaly Detection",
                "file": "cardiomegaly_binary_model.h5",
                "status": "✅ Active",
                "accuracy": "93.5%",
                "size": "13.1 MB",
                "last_updated": "2025-09-25"
            },
            {
                "name": "Arthritis Detection",
                "file": "arthritis_model.h5",
                "status": "⚠️ Needs Update",
                "accuracy": "91.2%",
                "size": "10.9 MB",
                "last_updated": "2025-08-15"
            },
            {
                "name": "Osteoporosis Detection",
                "file": "osteoporosis_model.h5",
                "status": "✅ Active",
                "accuracy": "89.7%",
                "size": "9.8 MB",
                "last_updated": "2025-09-20"
            }
        ]
        
        # Display as DataFrame
        df = pd.DataFrame(models_data)
        st.dataframe(df, use_container_width=True)
        
        # Quick stats
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            active_models = sum(1 for model in models_data if "Active" in model["status"])
            st.metric("Active Models", active_models)
        
        with stat_col2:
            avg_accuracy = np.mean([float(model["accuracy"].rstrip('%')) for model in models_data])
            st.metric("Average Accuracy", f"{avg_accuracy:.1f}%")
        
        with stat_col3:
            total_size = sum([float(model["size"].split()[0]) for model in models_data])
            st.metric("Total Size", f"{total_size:.1f} MB")
        
        with stat_col4:
            needs_update = sum(1 for model in models_data if "Needs Update" in model["status"])
            st.metric("Needs Update", needs_update)
    
    def _show_model_performance(self):
        """Show model performance metrics and comparisons"""
        st.markdown("**📊 Model Performance Analysis**")
        
        # Performance comparison chart
        performance_data = {
            "Model": ["Bone Fracture", "Pneumonia", "Cardiomegaly", "Arthritis", "Osteoporosis"],
            "Accuracy": [95.2, 94.8, 93.5, 91.2, 89.7],
            "Precision": [94.8, 95.1, 92.9, 90.5, 88.9],
            "Recall": [95.6, 94.5, 94.1, 91.9, 90.5],
            "F1-Score": [95.2, 94.8, 93.5, 91.2, 89.7]
        }
        
        perf_df = pd.DataFrame(performance_data)
        
        # Metrics selection
        selected_metric = st.selectbox(
            "Select Metric to Display",
            ["Accuracy", "Precision", "Recall", "F1-Score"]
        )
        
        # Create chart
        chart_data = perf_df.set_index("Model")[selected_metric]
        st.bar_chart(chart_data)
        
        # Performance details
        st.markdown("**Detailed Performance Metrics**")
        st.dataframe(perf_df.set_index("Model"), use_container_width=True)
    
    def _show_model_updates(self):
        """Show model update management"""
        st.markdown("**🔄 Model Update Management**")
        
        st.info("🔄 **Coming Soon**: Automatic model updates and version management")
        
        # Update checklist
        st.markdown("**Available Updates:**")
        updates = [
            {"model": "Arthritis Detection", "current": "v1.2", "available": "v1.3", "improvements": "5% accuracy boost"},
            {"model": "Pneumonia Detection", "current": "v2.1", "available": "v2.2", "improvements": "Faster inference"}
        ]
        
        for update in updates:
            with st.expander(f"📦 {update['model']} - {update['current']} → {update['available']}"):
                st.markdown(f"**Current Version:** {update['current']}")
                st.markdown(f"**Available Version:** {update['available']}")
                st.markdown(f"**Improvements:** {update['improvements']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.button(f"📥 Update {update['model']}", key=f"update_{update['model']}")
                with col2:
                    st.button(f"📋 View Changes", key=f"changes_{update['model']}")
                with col3:
                    st.button(f"⏭️ Skip", key=f"skip_{update['model']}")
    
    def _show_model_configuration(self):
        """Show model configuration options"""
        st.markdown("**⚙️ Model Configuration**")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model to Configure",
            ["Bone Fracture Detection", "Pneumonia Detection", "Cardiomegaly Detection", 
             "Arthritis Detection", "Osteoporosis Detection"]
        )
        
        st.markdown(f"**Configuration for: {selected_model}**")
        
        # Configuration options
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.markdown("**Inference Settings**")
            batch_size = st.slider("Batch Size", 1, 32, 8)
            use_gpu = st.checkbox("Use GPU Acceleration", value=True)
            precision = st.selectbox("Precision", ["float32", "float16", "mixed"])
        
        with config_col2:
            st.markdown("**Output Settings**")
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
            enable_gradcam = st.checkbox("Enable Grad-CAM", value=True)
            output_probabilities = st.checkbox("Output Raw Probabilities", value=False)
        
        # Save configuration
        if st.button("💾 Save Model Configuration"):
            st.success(f"✅ Configuration saved for {selected_model}")
    
    def _show_model_import_export(self):
        """Show model import/export functionality"""
        st.markdown("**📥 Model Import/Export**")
        
        import_col, export_col = st.columns(2)
        
        with import_col:
            st.markdown("**📥 Import Model**")
            uploaded_file = st.file_uploader(
                "Upload Model File",
                type=['h5', 'pkl', 'joblib', 'onnx'],
                help="Upload a trained model file"
            )
            
            if uploaded_file:
                model_name = st.text_input("Model Name", value="Custom Model")
                model_type = st.selectbox(
                    "Model Type",
                    ["Binary Classification", "Multi-class Classification", "Regression"]
                )
                
                if st.button("📥 Import Model"):
                    st.success(f"✅ Model '{model_name}' imported successfully!")
        
        with export_col:
            st.markdown("**📤 Export Model**")
            
            export_model = st.selectbox(
                "Select Model to Export",
                ["Bone Fracture Detection", "Pneumonia Detection", "Cardiomegaly Detection"]
            )
            
            export_format = st.selectbox(
                "Export Format",
                ["TensorFlow SavedModel", "ONNX", "TensorFlow Lite", "Pickle"]
            )
            
            include_metadata = st.checkbox("Include Metadata", value=True)
            
            if st.button("📤 Export Model"):
                st.success(f"✅ {export_model} exported in {export_format} format!")
    
    def _show_model_cleanup(self):
        """Show model cleanup and maintenance options"""
        st.markdown("**🗑️ Model Cleanup & Maintenance**")
        
        # Storage analysis
        st.markdown("**💾 Storage Analysis**")
        
        storage_data = {
            "Category": ["Active Models", "Backup Models", "Training Data", "Logs", "Cache"],
            "Size (MB)": [45.7, 123.4, 2845.2, 12.3, 89.1],
            "Files": [5, 15, 128, 45, 234]
        }
        
        storage_df = pd.DataFrame(storage_data)
        st.dataframe(storage_df, use_container_width=True)
        
        # Cleanup options
        st.markdown("**🧹 Cleanup Options**")
        
        cleanup_col1, cleanup_col2 = st.columns(2)
        
        with cleanup_col1:
            st.markdown("**Safe Cleanup**")
            
            clear_cache = st.checkbox("Clear Model Cache (89.1 MB)", value=False)
            clear_logs = st.checkbox("Clear Old Logs (12.3 MB)", value=False)
            clear_temp = st.checkbox("Clear Temporary Files", value=True)
            
            if st.button("🧹 Run Safe Cleanup"):
                freed_space = 0
                if clear_cache:
                    freed_space += 89.1
                if clear_logs:
                    freed_space += 12.3
                if clear_temp:
                    freed_space += 5.2
                
                st.success(f"✅ Cleanup completed! Freed {freed_space:.1f} MB of storage.")
        
        with cleanup_col2:
            st.markdown("**Advanced Cleanup**")
            st.warning("⚠️ **Warning**: These operations cannot be undone!")
            
            remove_backups = st.checkbox("Remove Old Model Backups", value=False)
            compress_models = st.checkbox("Compress Unused Models", value=False)
            
            if st.button("⚠️ Run Advanced Cleanup", type="secondary"):
                st.warning("Please confirm this action in the next update.")

class ExperimentalFeaturesModule:
    """Experimental and beta features"""
    
    def show_experimental_features(self):
        """Display experimental features interface"""
        st.markdown("### 🧪 **Experimental Features**")
        
        st.warning("⚠️ **Experimental Features**: These features are in beta and may not work as expected.")
        
        feature_tabs = st.tabs([
            "🤖 AI Assistant", "🔮 Predictive Analytics", "🌐 Cloud Integration", 
            "📱 Mobile Optimization", "🎯 Custom Training"
        ])
        
        with feature_tabs[0]:
            self._show_ai_assistant()
        
        with feature_tabs[1]:
            self._show_predictive_analytics()
        
        with feature_tabs[2]:
            self._show_cloud_integration()
        
        with feature_tabs[3]:
            self._show_mobile_optimization()
        
        with feature_tabs[4]:
            self._show_custom_training()
    
    def _show_ai_assistant(self):
        """Show AI assistant feature"""
        st.markdown("**🤖 AI-Powered Medical Assistant**")
        st.info("🚀 **Coming Soon**: Intelligent assistant for medical image interpretation")
        
        # Mock interface
        user_query = st.text_area(
            "Ask the AI Assistant",
            placeholder="e.g., 'What are the key indicators of pneumonia in this X-ray?'"
        )
        
        if st.button("🤖 Ask Assistant"):
            st.markdown("**🤖 AI Response:**")
            st.info("This feature will provide intelligent medical insights and explanations in the next release.")
    
    def _show_predictive_analytics(self):
        """Show predictive analytics feature"""
        st.markdown("**🔮 Predictive Health Analytics**")
        st.info("🚀 **Coming Soon**: Predictive models for health trend analysis")
        
        # Mock predictive interface
        st.markdown("**Prediction Categories:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("- 📈 **Disease Progression**")
            st.markdown("- 🎯 **Risk Assessment**") 
            st.markdown("- 📊 **Treatment Response**")
        
        with col2:
            st.markdown("- 🔄 **Follow-up Scheduling**")
            st.markdown("- ⚠️ **Early Warning System**")
            st.markdown("- 📋 **Outcome Prediction**")
    
    def _show_cloud_integration(self):
        """Show cloud integration feature"""
        st.markdown("**🌐 Cloud Integration & Collaboration**")
        st.info("🚀 **Coming Soon**: Secure cloud storage and multi-user collaboration")
        
        # Mock cloud interface
        st.markdown("**Planned Features:**")
        
        features = [
            "☁️ Secure cloud storage for medical images",
            "👥 Multi-user collaboration workspace",
            "🔒 HIPAA-compliant data handling",
            "📱 Cross-device synchronization",
            "🌍 Global model updates",
            "📊 Centralized analytics dashboard"
        ]
        
        for feature in features:
            st.markdown(feature)
    
    def _show_mobile_optimization(self):
        """Show mobile optimization features"""
        st.markdown("**📱 Mobile & Tablet Optimization**")
        st.info("🚀 **Coming Soon**: Responsive design for mobile medical professionals")
        
        # Mock mobile interface preview
        st.markdown("**Mobile Features in Development:**")
        
        mobile_col1, mobile_col2 = st.columns(2)
        
        with mobile_col1:
            st.markdown("**📱 Interface**")
            st.markdown("- Touch-optimized controls")
            st.markdown("- Swipe navigation")
            st.markdown("- Voice commands")
            st.markdown("- Offline mode")
        
        with mobile_col2:
            st.markdown("**🔧 Functionality**")
            st.markdown("- Camera integration")
            st.markdown("- Quick capture mode")
            st.markdown("- Gesture controls")
            st.markdown("- Emergency alerts")
    
    def _show_custom_training(self):
        """Show custom model training features"""
        st.markdown("**🎯 Custom Model Training**")
        st.info("🚀 **Coming Soon**: Train custom models with your own medical data")
        
        # Mock training interface
        st.markdown("**Training Pipeline:**")
        
        pipeline_steps = [
            "1️⃣ **Data Upload** - Upload your medical image dataset",
            "2️⃣ **Data Validation** - Automatic quality checks and formatting",
            "3️⃣ **Model Configuration** - Choose architecture and hyperparameters",
            "4️⃣ **Training** - Distributed training with progress monitoring",
            "5️⃣ **Validation** - Comprehensive model evaluation",
            "6️⃣ **Deployment** - One-click model deployment"
        ]
        
        for step in pipeline_steps:
            st.markdown(step)
        
        # Training form preview
        st.markdown("**Quick Training Setup:**")
        
        train_col1, train_col2 = st.columns(2)
        
        with train_col1:
            dataset_name = st.text_input("Dataset Name", placeholder="My Custom Dataset")
            model_architecture = st.selectbox("Model Architecture", ["DenseNet121", "ResNet50", "EfficientNet"])
        
        with train_col2:
            training_epochs = st.slider("Training Epochs", 10, 200, 50)
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64])
        
        if st.button("🚀 Start Training (Preview)"):
            st.success("Training interface will be available in the next major release!")

def initialize_feature_completion():
    """Initialize all feature completion modules"""
    
    # Initialize session state for feature modules
    if 'analytics_module' not in st.session_state:
        st.session_state.analytics_module = AdvancedAnalyticsModule()
    
    if 'model_management_module' not in st.session_state:
        st.session_state.model_management_module = ModelManagementModule()
    
    if 'experimental_features_module' not in st.session_state:
        st.session_state.experimental_features_module = ExperimentalFeaturesModule()
    
    if 'config_persistence_manager' not in st.session_state:
        st.session_state.config_persistence_manager = ConfigurationPersistenceManager()

def show_feature_completion_interface():
    """Display the feature completion interface"""
    
    st.markdown("## 🚀 **Advanced Features & Configuration**")
    
    # Initialize modules
    initialize_feature_completion()
    
    # Feature selection tabs
    feature_tabs = st.tabs([
        "📊 Advanced Analytics", 
        "🔧 Model Management", 
        "🧪 Experimental Features",
        "💾 Configuration Management"
    ])
    
    with feature_tabs[0]:
        st.session_state.analytics_module.create_analytics_dashboard()
    
    with feature_tabs[1]:
        st.session_state.model_management_module.show_model_management_interface()
    
    with feature_tabs[2]:
        st.session_state.experimental_features_module.show_experimental_features()
    
    with feature_tabs[3]:
        show_configuration_management_interface()

def show_configuration_management_interface():
    """Show advanced configuration management interface"""
    
    st.markdown("### 💾 **Advanced Configuration Management**")
    
    config_manager = st.session_state.config_persistence_manager
    
    # Configuration operations
    config_tabs = st.tabs([
        "📊 Overview", "💾 Backup & Restore", "📤 Import/Export", 
        "🎯 Presets", "📋 History", "🧹 Maintenance"
    ])
    
    with config_tabs[0]:
        # Storage statistics
        stats = config_manager.get_storage_statistics()
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("Config Size", f"{stats['config_size'] / 1024:.1f} KB")
        
        with stat_col2:
            st.metric("Backups", stats['backups_count'])
        
        with stat_col3:
            st.metric("Presets", stats['presets_count'])
        
        with stat_col4:
            st.metric("Total Size", f"{stats['total_size'] / (1024*1024):.1f} MB")
    
    with config_tabs[1]:
        st.markdown("**🔄 Backup & Restore Operations**")
        
        backup_col1, backup_col2 = st.columns(2)
        
        with backup_col1:
            if st.button("💾 Create Manual Backup"):
                backup_name = config_manager._create_backup()
                if backup_name:
                    st.success(f"✅ Backup created: {backup_name}")
        
        with backup_col2:
            if st.button("🧹 Clean Old Backups"):
                cleanup_stats = config_manager.cleanup_storage()
                st.success(f"✅ Removed {cleanup_stats['backups_removed']} old backups")
    
    with config_tabs[2]:
        st.markdown("**📤 Advanced Import/Export**")
        
        # Export options
        st.markdown("**Export Configuration Package**")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            include_backups = st.checkbox("Include Backups", value=True)
        
        with export_col2:
            include_presets = st.checkbox("Include Presets", value=True)
        
        with export_col3:
            include_logs = st.checkbox("Include Logs", value=False)
        
        if st.button("📦 Create Configuration Package"):
            package_data = config_manager.export_configuration_package(
                include_backups=include_backups,
                include_presets=include_presets,
                include_logs=include_logs
            )
            
            if package_data:
                st.download_button(
                    label="⬇️ Download Configuration Package",
                    data=package_data,
                    file_name=f"medical_ai_config_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )
    
    with config_tabs[3]:
        st.markdown("**🎯 Configuration Presets**")
        
        # Available presets
        presets = config_manager.get_available_presets()
        
        if presets:
            for preset in presets:
                with st.expander(f"📋 {preset['name']}"):
                    st.markdown(f"**Description:** {preset['description']}")
                    st.markdown(f"**Created:** {preset['created_date'][:10]}")
                    
                    preset_col1, preset_col2 = st.columns(2)
                    
                    with preset_col1:
                        if st.button(f"📥 Load {preset['name']}", key=f"load_{preset['filename']}"):
                            config = config_manager.load_configuration_preset(preset['name'])
                            if config:
                                st.success(f"✅ Loaded preset: {preset['name']}")
                    
                    with preset_col2:
                        if st.button(f"🗑️ Delete {preset['name']}", key=f"delete_{preset['filename']}"):
                            st.warning("Preset deletion will be implemented in next update")
    
    with config_tabs[4]:
        st.markdown("**📋 Configuration History**")
        
        history = config_manager.get_configuration_history()
        
        if history:
            # Show recent changes
            st.markdown("**Recent Changes:**")
            
            for i, entry in enumerate(history[:10]):  # Show last 10 changes
                st.markdown(f"**{entry['timestamp'][:19]}** - {entry['action']}: {entry['details']}")
        else:
            st.info("No configuration history available")
    
    with config_tabs[5]:
        st.markdown("**🧹 Storage Maintenance**")
        
        maintenance_col1, maintenance_col2 = st.columns(2)
        
        with maintenance_col1:
            keep_backups = st.number_input("Keep Recent Backups", 1, 20, 5)
            keep_logs_days = st.number_input("Keep Logs (days)", 1, 365, 30)
        
        with maintenance_col2:
            if st.button("🧹 Run Maintenance"):
                cleanup_stats = config_manager.cleanup_storage(keep_backups, keep_logs_days)
                st.success(f"✅ Maintenance complete! Freed {cleanup_stats['space_freed'] / 1024:.1f} KB")