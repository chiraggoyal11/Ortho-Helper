# Usage Tracking System for Medical X-ray AI System

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import streamlit as st
from pathlib import Path

class UsageTracker:
    """Track system usage and generate analytics"""
    
    def __init__(self, usage_file: str = "usage_data.json"):
        self.usage_file = usage_file
        self.session_file = "session_data.json"
        
    def log_classification(self, user_role: str, classification_type: str, 
                          prediction: str, confidence: float, processing_time: float = 0.0):
        """Log a classification event"""
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'classification',
            'user_role': user_role,
            'classification_type': classification_type,
            'prediction': prediction,
            'confidence': confidence,
            'processing_time': processing_time,
            'session_id': self._get_session_id()
        }
        
        self._save_event(event)
    
    def log_page_visit(self, page_name: str, user_role: str):
        """Log a page visit"""
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'page_visit',
            'page_name': page_name,
            'user_role': user_role,
            'session_id': self._get_session_id()
        }
        
        self._save_event(event)
    
    def log_model_training(self, user_role: str, model_type: str, 
                          epochs: int, success: bool):
        """Log a model training event"""
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'model_training',
            'user_role': user_role,
            'model_type': model_type,
            'epochs': epochs,
            'success': success,
            'session_id': self._get_session_id()
        }
        
        self._save_event(event)
    
    def log_report_generation(self, user_role: str, report_type: str):
        """Log a report generation event"""
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'report_generation',
            'user_role': user_role,
            'report_type': report_type,
            'session_id': self._get_session_id()
        }
        
        self._save_event(event)
    
    def log_user_login(self, user_role: str, username: str):
        """Log a user login"""
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'user_login',
            'user_role': user_role,
            'username': username,
            'session_id': self._get_session_id()
        }
        
        self._save_event(event)
    
    def _get_session_id(self) -> str:
        """Get or create session ID"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(hash(datetime.now()))[-6:]
        
        return st.session_state.session_id
    
    def _save_event(self, event: Dict[str, Any]):
        """Save event to usage file"""
        try:
            # Load existing events
            events = self._load_events()
            
            # Add new event
            events.append(event)
            
            # Keep only last 10000 events to prevent file from getting too large
            if len(events) > 10000:
                events = events[-10000:]
            
            # Save to file
            with open(self.usage_file, 'w') as f:
                json.dump(events, f, indent=2)
                
        except Exception as e:
            # Don't break the app if logging fails
            print(f"Warning: Could not log usage event: {str(e)}")
    
    def _load_events(self) -> List[Dict[str, Any]]:
        """Load existing events"""
        try:
            if os.path.exists(self.usage_file):
                with open(self.usage_file, 'r') as f:
                    return json.load(f)
            else:
                return []
        except:
            return []
    
    def get_usage_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get usage statistics for the specified number of days"""
        
        events = self._load_events()
        if not events:
            return self._get_empty_stats()
        
        # Filter events by date range
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_events = []
        
        for event in events:
            try:
                event_date = datetime.fromisoformat(event['timestamp'])
                if event_date >= cutoff_date:
                    recent_events.append(event)
            except:
                continue
        
        if not recent_events:
            return self._get_empty_stats()
        
        # Calculate statistics
        stats = {
            'total_events': len(recent_events),
            'date_range': f"Last {days} days",
            'classifications': self._count_by_type(recent_events, 'classification'),
            'page_visits': self._count_by_type(recent_events, 'page_visit'),
            'model_training': self._count_by_type(recent_events, 'model_training'),
            'report_generation': self._count_by_type(recent_events, 'report_generation'),
            'user_logins': self._count_by_type(recent_events, 'user_login'),
            'user_role_distribution': self._get_user_role_distribution(recent_events),
            'classification_types': self._get_classification_distribution(recent_events),
            'daily_activity': self._get_daily_activity(recent_events, days),
            'most_active_pages': self._get_most_active_pages(recent_events),
            'average_confidence': self._get_average_confidence(recent_events)
        }
        
        return stats
    
    def _get_empty_stats(self) -> Dict[str, Any]:
        """Return empty statistics when no data is available"""
        return {
            'total_events': 0,
            'message': 'No usage data available yet. Start using the system to see analytics!'
        }
    
    def _count_by_type(self, events: List[Dict], event_type: str) -> int:
        """Count events by type"""
        return len([e for e in events if e.get('event_type') == event_type])
    
    def _get_user_role_distribution(self, events: List[Dict]) -> Dict[str, int]:
        """Get distribution of events by user role"""
        distribution = {}
        for event in events:
            role = event.get('user_role', 'unknown')
            distribution[role] = distribution.get(role, 0) + 1
        return distribution
    
    def _get_classification_distribution(self, events: List[Dict]) -> Dict[str, int]:
        """Get distribution of classification types"""
        distribution = {}
        for event in events:
            if event.get('event_type') == 'classification':
                class_type = event.get('classification_type', 'unknown')
                distribution[class_type] = distribution.get(class_type, 0) + 1
        return distribution
    
    def _get_daily_activity(self, events: List[Dict], days: int) -> List[Dict[str, Any]]:
        """Get daily activity breakdown"""
        daily_counts = {}
        
        for event in events:
            try:
                event_date = datetime.fromisoformat(event['timestamp']).date()
                date_str = event_date.strftime('%Y-%m-%d')
                daily_counts[date_str] = daily_counts.get(date_str, 0) + 1
            except:
                continue
        
        # Fill in missing days with 0
        activity_list = []
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).date()
            date_str = date.strftime('%Y-%m-%d')
            activity_list.append({
                'date': date_str,
                'count': daily_counts.get(date_str, 0)
            })
        
        return list(reversed(activity_list))
    
    def _get_most_active_pages(self, events: List[Dict]) -> List[Dict[str, Any]]:
        """Get most visited pages"""
        page_counts = {}
        for event in events:
            if event.get('event_type') == 'page_visit':
                page = event.get('page_name', 'unknown')
                page_counts[page] = page_counts.get(page, 0) + 1
        
        # Sort by count and return top 5
        sorted_pages = sorted(page_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return [{'page': page, 'visits': count} for page, count in sorted_pages]
    
    def _get_average_confidence(self, events: List[Dict]) -> float:
        """Get average confidence of classifications"""
        confidences = []
        for event in events:
            if event.get('event_type') == 'classification' and 'confidence' in event:
                try:
                    confidences.append(float(event['confidence']))
                except:
                    continue
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def create_sample_data(self):
        """Create sample usage data for demonstration"""
        sample_events = [
            {
                'timestamp': (datetime.now() - timedelta(days=1)).isoformat(),
                'event_type': 'classification',
                'user_role': 'student',
                'classification_type': 'Bone Fracture Detection',
                'prediction': 'Fracture Detected',
                'confidence': 0.85,
                'processing_time': 2.3,
                'session_id': 'demo_session_1'
            },
            {
                'timestamp': (datetime.now() - timedelta(days=1, hours=2)).isoformat(),
                'event_type': 'page_visit',
                'page_name': 'X-ray Classification',
                'user_role': 'student',
                'session_id': 'demo_session_1'
            },
            {
                'timestamp': (datetime.now() - timedelta(days=2)).isoformat(),
                'event_type': 'classification',
                'user_role': 'doctor',
                'classification_type': 'Chest Condition Detection',
                'prediction': 'Pneumonia Detected',
                'confidence': 0.92,
                'processing_time': 1.8,
                'session_id': 'demo_session_2'
            },
            {
                'timestamp': (datetime.now() - timedelta(days=2, hours=1)).isoformat(),
                'event_type': 'report_generation',
                'user_role': 'doctor',
                'report_type': 'PDF',
                'session_id': 'demo_session_2'
            },
            {
                'timestamp': (datetime.now() - timedelta(days=3)).isoformat(),
                'event_type': 'model_training',
                'user_role': 'doctor',
                'model_type': 'bone_fracture',
                'epochs': 5,
                'success': True,
                'session_id': 'demo_session_3'
            }
        ]
        
        # Save sample data
        with open(self.usage_file, 'w') as f:
            json.dump(sample_events, f, indent=2)

# Global usage tracker instance
usage_tracker = UsageTracker()

# Convenience functions
def log_classification(user_role: str, classification_type: str, 
                      prediction: str, confidence: float, processing_time: float = 0.0):
    """Log a classification event"""
    usage_tracker.log_classification(user_role, classification_type, prediction, confidence, processing_time)

def log_page_visit(page_name: str, user_role: str):
    """Log a page visit"""
    usage_tracker.log_page_visit(page_name, user_role)

def log_model_training(user_role: str, model_type: str, epochs: int, success: bool):
    """Log a model training event"""
    usage_tracker.log_model_training(user_role, model_type, epochs, success)

def log_report_generation(user_role: str, report_type: str):
    """Log a report generation event"""
    usage_tracker.log_report_generation(user_role, report_type)

def log_user_login(user_role: str, username: str):
    """Log a user login"""
    usage_tracker.log_user_login(user_role, username)

def get_usage_statistics(days: int = 7) -> Dict[str, Any]:
    """Get usage statistics"""
    return usage_tracker.get_usage_statistics(days)

def create_sample_usage_data():
    """Create sample usage data for demonstration"""
    usage_tracker.create_sample_data()