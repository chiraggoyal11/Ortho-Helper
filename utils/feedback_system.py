# Feedback System Module for Medical X-ray AI System

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import os
import sqlite3
import hashlib

class FeedbackManager:
    """Manages user feedback collection and storage for continuous improvement"""
    
    def __init__(self, feedback_file: str = "feedback_data.json"):
        self.feedback_file = feedback_file
        self.predefined_feedback = {
            'positive': [
                "Accurate diagnosis - matches clinical assessment",
                "Helpful visualization with Grad-CAM",
                "Fast and reliable prediction",
                "User-friendly interface",
                "Good confidence calibration"
            ],
            'negative': [
                "Incorrect diagnosis - disagrees with clinical findings",
                "Poor image quality affected results",
                "Confidence level seems inappropriate",
                "Missing important features",
                "Interface could be improved"
            ],
            'neutral': [
                "Uncertain about accuracy - needs further verification",
                "Partial agreement with diagnosis",
                "Image quality could be better",
                "Need more context for proper assessment",
                "Results are reasonable but not definitive"
            ],
            'suggestions': [
                "Add more detailed explanations",
                "Improve preprocessing options",
                "Include additional visualization methods",
                "Provide comparative analysis features",
                "Add batch processing capabilities"
            ]
        }
    
    def get_predefined_feedback(self, category: str) -> List[str]:
        """Get predefined feedback options for a category"""
        return self.predefined_feedback.get(category, [])
    
    def collect_feedback(self, prediction_results: Dict[str, Any]) -> Dict[str, Any]:
        """Collect feedback from user through Streamlit interface"""
        st.markdown("### 💭 Provide Your Feedback")
        st.markdown("Your feedback helps improve the AI model's performance!")
        
        feedback_data = {}
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Feedback type selection
            feedback_type = st.selectbox(
                "📋 Feedback Type",
                ["Positive", "Negative", "Neutral", "Suggestion"],
                help="Select the type of feedback you want to provide"
            )
            
            # Get predefined options based on type
            category_key = feedback_type.lower().replace('suggestion', 'suggestions')
            predefined_options = self.get_predefined_feedback(category_key)
            
            # Predefined feedback selection
            selected_feedback = st.selectbox(
                "🎯 Quick Feedback",
                ["Select a predefined option..."] + predefined_options,
                help="Choose from common feedback options"
            )
            
            # Additional comments
            additional_comments = st.text_area(
                "💬 Additional Comments (Optional)",
                placeholder="Please provide any additional details, suggestions, or observations...",
                height=100
            )
            
            # Clinical experience (optional)
            clinical_experience = st.selectbox(
                "👨‍⚕️ Your Background (Optional)",
                ["Prefer not to say", "Medical Student", "Resident", "Radiologist", 
                 "General Practitioner", "Specialist", "Researcher", "Other"],
                help="This helps us understand the source of feedback"
            )
        
        with col2:
            # Rating system
            st.markdown("#### ⭐ Rate This Prediction")
            overall_rating = st.select_slider(
                "Overall Rating",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: f"{x} {'⭐' * x}"
            )
            
            # Specific aspect ratings
            st.markdown("**Detailed Ratings:**")
            accuracy_rating = st.slider("Accuracy", 1, 5, 3)
            confidence_rating = st.slider("Confidence Appropriateness", 1, 5, 3)
            usefulness_rating = st.slider("Clinical Usefulness", 1, 5, 3)
            
            # Would recommend?
            would_recommend = st.radio(
                "Would you recommend this system?",
                ["Yes", "Maybe", "No"]
            )
        
        # Submit feedback
        if st.button("📤 Submit Feedback", type="primary", width='stretch'):
            feedback_data = {
                'feedback_type': feedback_type,
                'selected_feedback': selected_feedback if selected_feedback != "Select a predefined option..." else "",
                'additional_comments': additional_comments,
                'clinical_experience': clinical_experience,
                'overall_rating': overall_rating,
                'accuracy_rating': accuracy_rating,
                'confidence_rating': confidence_rating,
                'usefulness_rating': usefulness_rating,
                'would_recommend': would_recommend,
                'prediction': prediction_results['prediction'],
                'confidence': prediction_results['confidence'],
                'model_type': prediction_results['type'],
                'timestamp': datetime.now(),
                'user_session': self._get_session_id()
            }
            
            # Save feedback
            if self.save_feedback(feedback_data):
                st.success("✅ Thank you for your feedback! It will help improve the system.")
                return feedback_data
            else:
                st.error("❌ Error saving feedback. Please try again.")
        
        return {}
    
    def save_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """Save feedback data to file"""
        try:
            # Load existing feedback
            existing_feedback = self.load_feedback()
            
            # Add new feedback
            existing_feedback.append(feedback_data)
            
            # Save to file
            with open(self.feedback_file, 'w') as f:
                json.dump(existing_feedback, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            st.error(f"Error saving feedback: {str(e)}")
            return False
    
    def load_feedback(self) -> List[Dict[str, Any]]:
        """Load existing feedback data"""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            else:
                return []
        except Exception as e:
            st.error(f"Error loading feedback: {str(e)}")
            return []
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get statistics from collected feedback"""
        feedback_list = self.load_feedback()
        
        if not feedback_list:
            return {"total_feedback": 0}
        
        # Basic statistics
        total_feedback = len(feedback_list)
        
        # Feedback type distribution
        type_distribution = {}
        for feedback in feedback_list:
            ftype = feedback.get('type', 'Unknown')  # Fixed: was 'feedback_type'
            type_distribution[ftype] = type_distribution.get(ftype, 0) + 1
        
        # Rating statistics
        overall_ratings = [f.get('rating', 0) for f in feedback_list if f.get('rating')]  # Fixed: was 'overall_rating'
        # For now, use the same rating for accuracy since the old data structure only has one rating field
        accuracy_ratings = overall_ratings
        
        # Calculate averages
        avg_overall_rating = sum(overall_ratings) / len(overall_ratings) if overall_ratings else 0
        avg_accuracy_rating = sum(accuracy_ratings) / len(accuracy_ratings) if accuracy_ratings else 0
        
        # Recommendation statistics - this field doesn't exist in current data, so create empty stats
        recommendation_stats = {'Yes': 0, 'No': 0}
        
        # Most common feedback - use comments field
        common_feedback = {}
        for feedback in feedback_list:
            comments = feedback.get('comments', '')
            if comments and comments.strip():
                common_feedback[comments] = common_feedback.get(comments, 0) + 1
        
        # Sort common feedback by frequency
        common_feedback = dict(sorted(common_feedback.items(), key=lambda x: x[1], reverse=True))
        
        statistics = {
            'total_feedback': total_feedback,
            'type_distribution': type_distribution,
            'avg_overall_rating': avg_overall_rating,
            'avg_accuracy_rating': avg_accuracy_rating,
            'recommendation_stats': recommendation_stats,
            'common_feedback': common_feedback,
            'recent_feedback': feedback_list[-5:] if feedback_list else []
        }
        
        return statistics
    
    def display_feedback_dashboard(self):
        """Display feedback statistics dashboard"""
        st.markdown("### 📊 Feedback Analytics Dashboard")
        
        stats = self.get_feedback_statistics()
        
        if stats['total_feedback'] == 0:
            st.info("No feedback collected yet. Encourage users to provide feedback!")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Feedback", stats['total_feedback'])
        
        with col2:
            st.metric("Avg Overall Rating", f"{stats['avg_overall_rating']:.1f}/5")
        
        with col3:
            st.metric("Avg Accuracy Rating", f"{stats['avg_accuracy_rating']:.1f}/5")
        
        with col4:
            recommend_yes = stats['recommendation_stats'].get('Yes', 0)
            recommend_total = sum(stats['recommendation_stats'].values())
            recommend_pct = (recommend_yes / recommend_total * 100) if recommend_total > 0 else 0
            st.metric("Would Recommend", f"{recommend_pct:.0f}%")
        
        # Feedback type distribution
        st.markdown("#### 📋 Feedback Type Distribution")
        if stats['type_distribution']:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            types = list(stats['type_distribution'].keys())
            counts = list(stats['type_distribution'].values())
            
            ax.bar(types, counts)
            ax.set_xlabel('Feedback Type')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Feedback Types')
            
            st.pyplot(fig)
        
        # Common feedback themes
        st.markdown("#### 🎯 Most Common Feedback")
        if stats['common_feedback']:
            for feedback_text, count in list(stats['common_feedback'].items())[:5]:
                st.write(f"• **{feedback_text}** ({count} times)")
        
        # Recent feedback
        st.markdown("#### 🕒 Recent Feedback")
        for feedback in stats['recent_feedback']:
            with st.expander(f"{feedback.get('type', 'Unknown')} feedback - {feedback.get('timestamp', 'Unknown')[:10]}"):
                st.write(f"**Rating:** {feedback.get('rating', 'N/A')}/5")
                st.write(f"**Prediction:** {feedback.get('prediction', 'Unknown')}")
                st.write(f"**Confidence:** {feedback.get('confidence', 'Unknown')}")
                if feedback.get('comments'):
                    st.write(f"**Comments:** {feedback.get('comments')}")
                else:
                    st.write("**Comments:** (none)")
    
    def export_feedback_data(self) -> pd.DataFrame:
        """Export feedback data as pandas DataFrame"""
        feedback_list = self.load_feedback()
        
        if not feedback_list:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(feedback_list)
        
        # Clean up timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def _get_session_id(self) -> str:
        """Generate a simple session ID for tracking purposes"""
        session_data = f"{datetime.now().strftime('%Y%m%d')}_{st.session_state.get('username', 'anonymous')}"
        return hashlib.md5(session_data.encode()).hexdigest()[:8]
    
    def analyze_feedback_trends(self) -> Dict[str, Any]:
        """Analyze trends in feedback over time"""
        df = self.export_feedback_data()
        
        if df.empty:
            return {}
        
        # Convert timestamp to datetime if it isn't already
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Group by date
            daily_feedback = df.groupby(df['timestamp'].dt.date).size()
            
            # Rating trends
            rating_trends = df.groupby(df['timestamp'].dt.date)['overall_rating'].mean()
            
            trends = {
                'daily_feedback_count': daily_feedback.to_dict(),
                'daily_avg_rating': rating_trends.to_dict(),
                'total_days_with_feedback': len(daily_feedback),
                'peak_feedback_day': daily_feedback.idxmax() if not daily_feedback.empty else None,
                'best_rating_day': rating_trends.idxmax() if not rating_trends.empty else None
            }
            
            return trends
        
        return {}

# Global feedback manager instance
feedback_manager = FeedbackManager()

def collect_feedback(prediction_results: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper function for collecting feedback"""
    return feedback_manager.collect_feedback(prediction_results)

def save_feedback(feedback_data: Dict[str, Any]) -> bool:
    """Wrapper function for saving feedback"""
    return feedback_manager.save_feedback(feedback_data)

def get_feedback_statistics() -> Dict[str, Any]:
    """Wrapper function for getting feedback statistics"""
    return feedback_manager.get_feedback_statistics()

def display_feedback_analytics():
    """Display feedback analytics in Streamlit"""
    feedback_manager.display_feedback_dashboard()

# Example usage and testing
if __name__ == "__main__":
    print("Feedback system module loaded successfully!")
    
    # Test with dummy data
    dummy_results = {
        'prediction': 'Fracture Detected',
        'confidence': 0.85,
        'type': 'bone',
        'model_used': 'Bone Fracture Detection Model'
    }
    
    print("Feedback system ready for user input")