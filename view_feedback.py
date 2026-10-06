# Feedback Viewer - Medical X-ray AI System
# View all stored feedback data

import json
import pandas as pd
from datetime import datetime
from pathlib import Path

def view_feedback_data():
    """Display all stored feedback data"""
    
    feedback_file = "feedback_data.json"
    
    print("🏥 Medical X-ray AI System - Feedback Data Viewer")
    print("=" * 60)
    
    if not Path(feedback_file).exists():
        print("❌ No feedback data found!")
        print(f"📁 Looking for: {Path(feedback_file).absolute()}")
        return
    
    try:
        # Load feedback data
        with open(feedback_file, 'r') as f:
            feedback_data = json.load(f)
        
        if not feedback_data:
            print("📝 Feedback file exists but is empty")
            return
        
        print(f"✅ Found {len(feedback_data)} feedback entries")
        print(f"📁 File location: {Path(feedback_file).absolute()}")
        print()
        
        # Display each feedback entry
        for i, feedback in enumerate(feedback_data, 1):
            print(f"📝 Feedback #{i}")
            print(f"   📅 Date: {feedback.get('timestamp', 'Unknown')}")
            print(f"   📋 Type: {feedback.get('type', 'Unknown')}")
            print(f"   ⭐ Rating: {feedback.get('rating', 'N/A')}/5")
            print(f"   🎯 Prediction: {feedback.get('prediction', 'Unknown')}")
            print(f"   📊 Confidence: {feedback.get('confidence', 'Unknown')}")
            
            comments = feedback.get('comments', '')
            if comments:
                print(f"   💬 Comments: {comments}")
            else:
                print(f"   💬 Comments: (none)")
            
            print("-" * 40)
        
        # Create summary statistics
        print("📊 SUMMARY STATISTICS:")
        print("-" * 40)
        
        # Rating statistics
        ratings = [f.get('rating', 0) for f in feedback_data if f.get('rating')]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            print(f"   ⭐ Average Rating: {avg_rating:.1f}/5")
            print(f"   📈 Total Ratings: {len(ratings)}")
        
        # Feedback types
        types = [f.get('type', 'Unknown') for f in feedback_data]
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        
        print(f"   📋 Feedback Types:")
        for ftype, count in type_counts.items():
            print(f"      - {ftype}: {count}")
        
        # Predictions feedback was about
        predictions = [f.get('prediction', 'Unknown') for f in feedback_data]
        pred_counts = {}
        for p in predictions:
            pred_counts[p] = pred_counts.get(p, 0) + 1
        
        print(f"   🎯 Predictions:")
        for pred, count in pred_counts.items():
            print(f"      - {pred}: {count}")
        
        print()
        print("💾 STORAGE DETAILS:")
        print(f"   📁 File: {Path(feedback_file).absolute()}")
        print(f"   📝 Format: JSON")
        print(f"   💽 Size: {Path(feedback_file).stat().st_size} bytes")
        
    except Exception as e:
        print(f"❌ Error reading feedback data: {str(e)}")

def export_feedback_to_csv():
    """Export feedback data to CSV for analysis"""
    
    feedback_file = "feedback_data.json"
    
    if not Path(feedback_file).exists():
        print("❌ No feedback data found to export!")
        return
    
    try:
        # Load feedback data
        with open(feedback_file, 'r') as f:
            feedback_data = json.load(f)
        
        if not feedback_data:
            print("📝 No feedback data to export")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(feedback_data)
        
        # Export to CSV
        csv_file = "feedback_export.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"✅ Feedback exported to: {Path(csv_file).absolute()}")
        print(f"📊 {len(df)} feedback entries exported")
        
    except Exception as e:
        print(f"❌ Error exporting feedback: {str(e)}")

if __name__ == "__main__":
    # View feedback data
    view_feedback_data()
    
    print("\n" + "=" * 60)
    
    # Ask if user wants to export to CSV
    export_choice = input("Export to CSV for analysis? (y/n): ").lower().strip()
    if export_choice in ['y', 'yes']:
        export_feedback_to_csv()
    
    print("\n🌐 View more feedback analytics in the web app:")
    print("   Navigate to 'Analytics' page at http://localhost:8511")