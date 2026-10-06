#!/usr/bin/env python3
"""
Enhanced Configuration Persistence Module
Complete file I/O system for medical AI settings with advanced features
"""

import json
import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import streamlit as st
import tempfile
import hashlib

class ConfigurationPersistenceManager:
    """Advanced configuration persistence with comprehensive file I/O operations"""
    
    def __init__(self):
        self.config_dir = Path("medical_ai_config")
        self.settings_file = self.config_dir / "user_settings.json"
        self.presets_dir = self.config_dir / "presets"
        self.backups_dir = self.config_dir / "backups"
        self.exports_dir = self.config_dir / "exports"
        self.cache_dir = self.config_dir / "cache"
        self.logs_dir = self.config_dir / "logs"
        
        self._ensure_directories()
        self._setup_logging()
    
    def _ensure_directories(self):
        """Create all necessary directories"""
        for directory in [self.config_dir, self.presets_dir, self.backups_dir, 
                         self.exports_dir, self.cache_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup configuration change logging"""
        self.log_file = self.logs_dir / f"config_changes_{datetime.now().strftime('%Y-%m')}.log"
    
    def _log_change(self, action: str, details: str = ""):
        """Log configuration changes"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {action}: {details}\n"
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            st.warning(f"Failed to log change: {e}")
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for configuration integrity"""
        config_string = json.dumps(data, sort_keys=True)
        return hashlib.md5(config_string.encode()).hexdigest()
    
    def save_configuration(self, config: Dict[str, Any], create_backup: bool = True) -> bool:
        """Save configuration with integrity checking and backup"""
        try:
            # Add metadata
            config["metadata"] = {
                **config.get("metadata", {}),
                "last_modified": datetime.now().isoformat(),
                "checksum": self._calculate_checksum(config),
                "version": "2.0"
            }
            
            # Create backup if requested
            if create_backup and self.settings_file.exists():
                self._create_backup()
            
            # Write configuration
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            
            # Verify integrity
            if self._verify_configuration_integrity(config):
                self._log_change("SAVE_SUCCESS", f"Configuration saved with checksum {config['metadata']['checksum']}")
                return True
            else:
                self._log_change("SAVE_ERROR", "Configuration integrity check failed")
                return False
                
        except Exception as e:
            self._log_change("SAVE_ERROR", str(e))
            st.error(f"Failed to save configuration: {e}")
            return False
    
    def load_configuration(self) -> Dict[str, Any]:
        """Load configuration with integrity verification"""
        try:
            if not self.settings_file.exists():
                return self._get_default_configuration()
            
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Verify integrity if checksum exists
            if "metadata" in config and "checksum" in config["metadata"]:
                if not self._verify_configuration_integrity(config):
                    self._log_change("LOAD_WARNING", "Configuration integrity check failed, using backup")
                    return self._restore_from_backup() or self._get_default_configuration()
            
            self._log_change("LOAD_SUCCESS", f"Configuration loaded from {self.settings_file}")
            return config
            
        except Exception as e:
            self._log_change("LOAD_ERROR", str(e))
            st.error(f"Failed to load configuration: {e}")
            return self._get_default_configuration()
    
    def _verify_configuration_integrity(self, config: Dict[str, Any]) -> bool:
        """Verify configuration integrity using checksum"""
        if "metadata" not in config or "checksum" not in config["metadata"]:
            return True  # No checksum to verify
        
        stored_checksum = config["metadata"]["checksum"]
        config_copy = config.copy()
        
        # Remove checksum for calculation
        if "metadata" in config_copy:
            config_copy["metadata"] = {k: v for k, v in config_copy["metadata"].items() if k != "checksum"}
        
        calculated_checksum = self._calculate_checksum(config_copy)
        return stored_checksum == calculated_checksum
    
    def _create_backup(self) -> Optional[str]:
        """Create timestamped backup of current configuration"""
        try:
            if not self.settings_file.exists():
                return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"settings_backup_{timestamp}.json"
            backup_path = self.backups_dir / backup_filename
            
            shutil.copy2(self.settings_file, backup_path)
            self._log_change("BACKUP_CREATED", backup_filename)
            
            # Clean old backups (keep last 10)
            self._cleanup_old_backups()
            
            return backup_filename
            
        except Exception as e:
            self._log_change("BACKUP_ERROR", str(e))
            return None
    
    def _cleanup_old_backups(self, keep_count: int = 10):
        """Clean up old backup files, keeping only the most recent ones"""
        try:
            backup_files = list(self.backups_dir.glob("settings_backup_*.json"))
            
            if len(backup_files) > keep_count:
                # Sort by modification time (newest first)
                backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                # Remove old backups
                for old_backup in backup_files[keep_count:]:
                    old_backup.unlink()
                    self._log_change("BACKUP_CLEANUP", f"Removed old backup {old_backup.name}")
                    
        except Exception as e:
            self._log_change("CLEANUP_ERROR", str(e))
    
    def restore_from_backup(self, backup_filename: str) -> bool:
        """Restore configuration from specific backup"""
        try:
            backup_path = self.backups_dir / backup_filename
            
            if not backup_path.exists():
                st.error(f"Backup file not found: {backup_filename}")
                return False
            
            # Create backup of current settings before restoring
            current_backup = self._create_backup()
            
            # Restore from backup
            shutil.copy2(backup_path, self.settings_file)
            
            self._log_change("RESTORE_SUCCESS", f"Restored from backup {backup_filename}")
            return True
            
        except Exception as e:
            self._log_change("RESTORE_ERROR", str(e))
            st.error(f"Failed to restore from backup: {e}")
            return False
    
    def _restore_from_backup(self) -> Optional[Dict[str, Any]]:
        """Automatically restore from most recent backup"""
        try:
            backup_files = list(self.backups_dir.glob("settings_backup_*.json"))
            
            if not backup_files:
                return None
            
            # Get most recent backup
            latest_backup = max(backup_files, key=lambda x: x.stat().st_mtime)
            
            with open(latest_backup, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self._log_change("AUTO_RESTORE", f"Auto-restored from {latest_backup.name}")
            return config
            
        except Exception as e:
            self._log_change("AUTO_RESTORE_ERROR", str(e))
            return None
    
    def export_configuration_package(self, include_backups: bool = True, include_presets: bool = True, include_logs: bool = False) -> Optional[bytes]:
        """Export complete configuration package as ZIP"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
                with zipfile.ZipFile(tmp_file.name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    
                    # Add main configuration
                    if self.settings_file.exists():
                        zip_file.write(self.settings_file, "settings/user_settings.json")
                    
                    # Add backups if requested
                    if include_backups:
                        for backup_file in self.backups_dir.glob("*.json"):
                            zip_file.write(backup_file, f"backups/{backup_file.name}")
                    
                    # Add presets if requested
                    if include_presets:
                        for preset_file in self.presets_dir.glob("*.json"):
                            zip_file.write(preset_file, f"presets/{preset_file.name}")
                    
                    # Add logs if requested
                    if include_logs:
                        for log_file in self.logs_dir.glob("*.log"):
                            zip_file.write(log_file, f"logs/{log_file.name}")
                    
                    # Add export metadata
                    export_metadata = {
                        "export_date": datetime.now().isoformat(),
                        "export_version": "2.0",
                        "includes": {
                            "settings": True,
                            "backups": include_backups,
                            "presets": include_presets,
                            "logs": include_logs
                        }
                    }
                    
                    zip_file.writestr("export_metadata.json", json.dumps(export_metadata, indent=4))
                
                # Read the zip file
                with open(tmp_file.name, 'rb') as f:
                    zip_data = f.read()
                
                # Clean up temporary file
                os.unlink(tmp_file.name)
                
                self._log_change("EXPORT_SUCCESS", f"Configuration package exported ({len(zip_data)} bytes)")
                return zip_data
                
        except Exception as e:
            self._log_change("EXPORT_ERROR", str(e))
            st.error(f"Failed to export configuration package: {e}")
            return None
    
    def import_configuration_package(self, package_data: bytes) -> bool:
        """Import complete configuration package from ZIP"""
        try:
            # Create backup before importing
            self._create_backup()
            
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
                tmp_file.write(package_data)
                tmp_file.flush()
                
                with zipfile.ZipFile(tmp_file.name, 'r') as zip_file:
                    
                    # Extract to temporary directory
                    with tempfile.TemporaryDirectory() as temp_dir:
                        zip_file.extractall(temp_dir)
                        temp_path = Path(temp_dir)
                        
                        # Import settings
                        settings_file = temp_path / "settings" / "user_settings.json"
                        if settings_file.exists():
                            shutil.copy2(settings_file, self.settings_file)
                        
                        # Import backups
                        backups_dir = temp_path / "backups"
                        if backups_dir.exists():
                            for backup_file in backups_dir.glob("*.json"):
                                shutil.copy2(backup_file, self.backups_dir / backup_file.name)
                        
                        # Import presets
                        presets_dir = temp_path / "presets"
                        if presets_dir.exists():
                            for preset_file in presets_dir.glob("*.json"):
                                shutil.copy2(preset_file, self.presets_dir / preset_file.name)
                
                # Clean up temporary file
                os.unlink(tmp_file.name)
                
                self._log_change("IMPORT_SUCCESS", f"Configuration package imported ({len(package_data)} bytes)")
                return True
                
        except Exception as e:
            self._log_change("IMPORT_ERROR", str(e))
            st.error(f"Failed to import configuration package: {e}")
            return False
    
    def save_configuration_preset(self, name: str, config: Dict[str, Any], description: str = "") -> bool:
        """Save configuration as a named preset"""
        try:
            preset_data = {
                "name": name,
                "description": description,
                "configuration": config,
                "created_date": datetime.now().isoformat(),
                "version": "2.0"
            }
            
            preset_filename = f"{name.replace(' ', '_').lower()}.json"
            preset_path = self.presets_dir / preset_filename
            
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(preset_data, f, indent=4, ensure_ascii=False)
            
            self._log_change("PRESET_SAVED", f"Preset '{name}' saved as {preset_filename}")
            return True
            
        except Exception as e:
            self._log_change("PRESET_ERROR", str(e))
            st.error(f"Failed to save preset: {e}")
            return False
    
    def load_configuration_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """Load configuration from named preset"""
        try:
            preset_filename = f"{preset_name.replace(' ', '_').lower()}.json"
            preset_path = self.presets_dir / preset_filename
            
            if not preset_path.exists():
                st.error(f"Preset not found: {preset_name}")
                return None
            
            with open(preset_path, 'r', encoding='utf-8') as f:
                preset_data = json.load(f)
            
            self._log_change("PRESET_LOADED", f"Preset '{preset_name}' loaded")
            return preset_data["configuration"]
            
        except Exception as e:
            self._log_change("PRESET_LOAD_ERROR", str(e))
            st.error(f"Failed to load preset: {e}")
            return None
    
    def get_available_presets(self) -> List[Dict[str, str]]:
        """Get list of available configuration presets"""
        presets = []
        
        try:
            for preset_file in self.presets_dir.glob("*.json"):
                with open(preset_file, 'r', encoding='utf-8') as f:
                    preset_data = json.load(f)
                
                presets.append({
                    "name": preset_data["name"],
                    "description": preset_data.get("description", ""),
                    "created_date": preset_data["created_date"],
                    "filename": preset_file.name
                })
        
        except Exception as e:
            self._log_change("PRESET_LIST_ERROR", str(e))
        
        return sorted(presets, key=lambda x: x["created_date"], reverse=True)
    
    def get_configuration_history(self) -> List[Dict[str, Any]]:
        """Get configuration change history"""
        history = []
        
        try:
            for log_file in self.logs_dir.glob("*.log"):
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                # Parse log entry
                                timestamp_end = line.find('] ')
                                if timestamp_end > 0:
                                    timestamp_str = line[1:timestamp_end]
                                    action_details = line[timestamp_end + 2:].strip()
                                    
                                    action_end = action_details.find(': ')
                                    if action_end > 0:
                                        action = action_details[:action_end]
                                        details = action_details[action_end + 2:]
                                    else:
                                        action = action_details
                                        details = ""
                                    
                                    history.append({
                                        "timestamp": timestamp_str,
                                        "action": action,
                                        "details": details
                                    })
                            except Exception:
                                continue  # Skip malformed log entries
        
        except Exception as e:
            self._log_change("HISTORY_ERROR", str(e))
        
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage usage statistics"""
        stats = {
            "config_size": 0,
            "backups_count": 0,
            "backups_size": 0,
            "presets_count": 0,
            "presets_size": 0,
            "logs_size": 0,
            "total_size": 0
        }
        
        try:
            # Configuration file size
            if self.settings_file.exists():
                stats["config_size"] = self.settings_file.stat().st_size
            
            # Backups statistics
            backup_files = list(self.backups_dir.glob("*.json"))
            stats["backups_count"] = len(backup_files)
            stats["backups_size"] = sum(f.stat().st_size for f in backup_files)
            
            # Presets statistics
            preset_files = list(self.presets_dir.glob("*.json"))
            stats["presets_count"] = len(preset_files)
            stats["presets_size"] = sum(f.stat().st_size for f in preset_files)
            
            # Logs size
            log_files = list(self.logs_dir.glob("*.log"))
            stats["logs_size"] = sum(f.stat().st_size for f in log_files)
            
            # Total size
            stats["total_size"] = (stats["config_size"] + stats["backups_size"] + 
                                 stats["presets_size"] + stats["logs_size"])
        
        except Exception as e:
            self._log_change("STATS_ERROR", str(e))
        
        return stats
    
    def cleanup_storage(self, keep_backups: int = 5, keep_logs_days: int = 30) -> Dict[str, int]:
        """Clean up storage by removing old files"""
        cleanup_stats = {
            "backups_removed": 0,
            "logs_removed": 0,
            "space_freed": 0
        }
        
        try:
            # Clean old backups
            backup_files = list(self.backups_dir.glob("*.json"))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            for old_backup in backup_files[keep_backups:]:
                size = old_backup.stat().st_size
                old_backup.unlink()
                cleanup_stats["backups_removed"] += 1
                cleanup_stats["space_freed"] += size
            
            # Clean old logs
            cutoff_date = datetime.now() - timedelta(days=keep_logs_days)
            
            for log_file in self.logs_dir.glob("*.log"):
                if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                    size = log_file.stat().st_size
                    log_file.unlink()
                    cleanup_stats["logs_removed"] += 1
                    cleanup_stats["space_freed"] += size
            
            self._log_change("CLEANUP_SUCCESS", f"Removed {cleanup_stats['backups_removed']} backups, {cleanup_stats['logs_removed']} logs")
        
        except Exception as e:
            self._log_change("CLEANUP_ERROR", str(e))
        
        return cleanup_stats
    
    def _get_default_configuration(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "model": {
                "confidence_threshold": 0.5,
                "batch_processing": False,
                "gradcam_intensity": 0.4,
                "auto_preprocessing": True,
                "enable_gpu": False,
                "cache_models": True
            },
            "reports": {
                "include_metadata": True,
                "include_preprocessing_info": False,
                "include_gradcam": True,
                "default_format": "PDF",
                "auto_download": False,
                "compress_reports": True
            },
            "system": {
                "max_image_size_mb": 10,
                "dark_mode": False,
                "show_debug_info": False,
                "compact_layout": False
            },
            "privacy": {
                "auto_delete_images": True,
                "anonymous_feedback": True,
                "usage_analytics": False
            },
            "session": {
                "timeout": "30 minutes",
                "auto_save_settings": True,
                "remember_preferences": True
            },
            "advanced": {
                "preprocessing_presets": {},
                "custom_models": {},
                "experimental_features": False
            },
            "metadata": {
                "created_date": datetime.now().isoformat(),
                "version": "2.0",
                "configuration_type": "default"
            }
        }