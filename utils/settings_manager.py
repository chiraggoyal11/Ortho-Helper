# Settings Management Module for Medical X-ray AI System

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import streamlit as st
import tempfile
import zipfile

class SettingsManager:
    """
    Comprehensive settings management system with persistence,
    import/export functionality, and default configuration management
    """
    
    def __init__(self, settings_dir: str = "settings"):
        self.settings_dir = Path(settings_dir)
        self.settings_dir.mkdir(parents=True, exist_ok=True)
        
        self.user_settings_file = self.settings_dir / "user_settings.json"
        self.system_settings_file = self.settings_dir / "system_settings.json"
        self.backup_dir = self.settings_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize default settings
        self.default_settings = self._get_default_settings()
        
        # Load existing settings or create defaults
        self._ensure_settings_files()
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default system settings"""
        return {
            # Model Configuration
            "model": {
                "confidence_threshold": 0.5,
                "batch_processing": False,
                "gradcam_intensity": 0.4,
                "show_boundaries": True,
                "auto_preprocessing": True,
                "enable_gpu": False,
                "cache_models": True
            },
            
            # Report Settings
            "reports": {
                "include_metadata": True,
                "include_preprocessing_info": False,
                "include_gradcam": True,
                "default_format": "PDF",
                "auto_download": False,
                "compress_reports": True
            },
            
            # System Settings
            "system": {
                "max_image_size_mb": 10,
                "dark_mode": False,
                "show_debug_info": False,
                "compact_layout": False
            },
            
            # Privacy Settings
            "privacy": {
                "auto_delete_images": True,
                "anonymous_feedback": True,
                "usage_analytics": False
            },
            
            # Session Management
            "session": {
                "timeout": "30 minutes",
                "auto_save_settings": True,
                "remember_preferences": True
            },
            
            # Metadata
            "metadata": {
                "created_date": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "version": "1.0",
                "user_role": None
            }
        }
    
    def _ensure_settings_files(self):
        """Ensure settings files exist with default values"""
        if not self.user_settings_file.exists():
            self.save_settings(self.default_settings)
        
        if not self.system_settings_file.exists():
            system_config = {
                "installation_date": datetime.now().isoformat(),
                "system_version": "1.0.0",
                "total_users": 0,
                "last_backup": None
            }
            with open(self.system_settings_file, 'w') as f:
                json.dump(system_config, f, indent=4)
    
    def load_settings(self) -> Dict[str, Any]:
        """Load user settings from file"""
        try:
            with open(self.user_settings_file, 'r') as f:
                settings = json.load(f)
                
                # Merge with defaults to ensure all keys exist
                merged_settings = self._merge_with_defaults(settings)
                return merged_settings
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            st.warning(f"Could not load settings: {e}. Using defaults.")
            return self.default_settings.copy()
    
    def _merge_with_defaults(self, user_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user settings with defaults to ensure all keys exist"""
        merged = self.default_settings.copy()
        
        for category, settings in user_settings.items():
            if category in merged:
                if isinstance(settings, dict):
                    merged[category].update(settings)
                else:
                    merged[category] = settings
        
        return merged
    
    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """Save settings to file with backup"""
        try:
            # Create backup of existing settings
            if self.user_settings_file.exists():
                self._create_backup()
            
            # Update metadata
            settings["metadata"]["last_modified"] = datetime.now().isoformat()
            if hasattr(st.session_state, 'user_role'):
                settings["metadata"]["user_role"] = st.session_state.user_role
            
            # Save to file
            with open(self.user_settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
            
            return True
            
        except Exception as e:
            st.error(f"Failed to save settings: {e}")
            return False
    
    def _create_backup(self):
        """Create backup of current settings"""
        if self.user_settings_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"settings_backup_{timestamp}.json"
            
            # Copy current settings to backup
            with open(self.user_settings_file, 'r') as src:
                settings = json.load(src)
            
            with open(backup_file, 'w') as dst:
                json.dump(settings, dst, indent=4)
            
            # Keep only last 10 backups
            self._cleanup_old_backups()
    
    def _cleanup_old_backups(self):
        """Keep only the most recent 10 backup files"""
        backup_files = list(self.backup_dir.glob("settings_backup_*.json"))
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old backups
        for old_backup in backup_files[10:]:
            try:
                old_backup.unlink()
            except OSError:
                pass
    
    def reset_to_defaults(self) -> bool:
        """Reset all settings to default values"""
        try:
            # Create backup before resetting
            if self.user_settings_file.exists():
                self._create_backup()
            
            # Save default settings
            defaults = self.default_settings.copy()
            defaults["metadata"]["last_modified"] = datetime.now().isoformat()
            defaults["metadata"]["reset_date"] = datetime.now().isoformat()
            
            return self.save_settings(defaults)
            
        except Exception as e:
            st.error(f"Failed to reset settings: {e}")
            return False
    
    def export_settings(self) -> Optional[bytes]:
        """Export settings as a JSON file"""
        try:
            settings = self.load_settings()
            
            # Add export metadata
            export_data = {
                "settings": settings,
                "export_info": {
                    "export_date": datetime.now().isoformat(),
                    "export_version": "1.0",
                    "system": "Medical X-ray AI Classification System"
                }
            }
            
            # Create JSON string
            json_string = json.dumps(export_data, indent=4)
            return json_string.encode('utf-8')
            
        except Exception as e:
            st.error(f"Failed to export settings: {e}")
            return None
    
    def import_settings(self, imported_data: bytes) -> bool:
        """Import settings from uploaded file"""
        try:
            # Parse JSON data
            json_string = imported_data.decode('utf-8')
            data = json.loads(json_string)
            
            # Validate structure
            if "settings" not in data:
                st.error("Invalid settings file format")
                return False
            
            # Create backup before importing
            self._create_backup()
            
            # Import settings
            imported_settings = data["settings"]
            
            # Merge with defaults and save
            merged_settings = self._merge_with_defaults(imported_settings)
            merged_settings["metadata"]["imported_date"] = datetime.now().isoformat()
            
            return self.save_settings(merged_settings)
            
        except Exception as e:
            st.error(f"Failed to import settings: {e}")
            return False
    
    def get_setting(self, category: str, key: str, default: Any = None) -> Any:
        """Get a specific setting value"""
        settings = self.load_settings()
        return settings.get(category, {}).get(key, default)
    
    def update_setting(self, category: str, key: str, value: Any) -> bool:
        """Update a specific setting value"""
        settings = self.load_settings()
        
        if category not in settings:
            settings[category] = {}
        
        settings[category][key] = value
        return self.save_settings(settings)
    
    def get_backups(self) -> list:
        """Get list of available backup files"""
        backup_files = list(self.backup_dir.glob("settings_backup_*.json"))
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        backups = []
        for backup_file in backup_files:
            stat = backup_file.stat()
            backups.append({
                "filename": backup_file.name,
                "date": datetime.fromtimestamp(stat.st_mtime),
                "size": stat.st_size,
                "path": backup_file
            })
        
        return backups
    
    def restore_backup(self, backup_filename: str) -> bool:
        """Restore settings from a backup file"""
        try:
            backup_file = self.backup_dir / backup_filename
            
            if not backup_file.exists():
                st.error("Backup file not found")
                return False
            
            # Load backup settings
            with open(backup_file, 'r') as f:
                backup_settings = json.load(f)
            
            # Create current backup before restoring
            self._create_backup()
            
            # Restore settings
            backup_settings["metadata"]["restored_date"] = datetime.now().isoformat()
            backup_settings["metadata"]["restored_from"] = backup_filename
            
            return self.save_settings(backup_settings)
            
        except Exception as e:
            st.error(f"Failed to restore backup: {e}")
            return False
    
    def apply_settings_to_session(self, settings: Dict[str, Any]):
        """Apply settings to Streamlit session state for runtime use"""
        
        # Apply system settings to session state
        if "system" in settings:
            for key, value in settings["system"].items():
                st.session_state[f"setting_{key}"] = value
        
        # Apply model settings to session state
        if "model" in settings:
            for key, value in settings["model"].items():
                st.session_state[f"model_{key}"] = value
        
        # Apply report settings to session state
        if "reports" in settings:
            for key, value in settings["reports"].items():
                st.session_state[f"report_{key}"] = value
    
    def get_settings_summary(self) -> Dict[str, str]:
        """Get a summary of current settings for display"""
        settings = self.load_settings()
        
        summary = {
            "Model Settings": f"Confidence: {settings['model']['confidence_threshold']:.2f}, "
                             f"Grad-CAM: {settings['model']['gradcam_intensity']:.1f}, "
                             f"GPU: {'Enabled' if settings['model']['enable_gpu'] else 'Disabled'}",
            
            "Report Settings": f"Format: {settings['reports']['default_format']}, "
                              f"Auto-download: {'Yes' if settings['reports']['auto_download'] else 'No'}, "
                              f"Compression: {'Yes' if settings['reports']['compress_reports'] else 'No'}",
            
            "System Settings": f"Theme: {'Dark' if settings['system']['dark_mode'] else 'Light'}, "
                              f"Max Size: {settings['system']['max_image_size_mb']}MB, "
                              f"Debug: {'On' if settings['system']['show_debug_info'] else 'Off'}",
            
            "Privacy Settings": f"Auto-delete: {'Yes' if settings['privacy']['auto_delete_images'] else 'No'}, "
                               f"Analytics: {'Enabled' if settings['privacy']['usage_analytics'] else 'Disabled'}, "
                               f"Timeout: {settings['session']['timeout']}"
        }
        
        return summary