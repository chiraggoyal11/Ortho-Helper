# Advanced Feedback Database System for Medical X-ray AI
# Designed to handle lakhs of feedback entries efficiently

import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path

class FeedbackDatabase:
    """Database manager for storing and retrieving feedback efficiently"""
    
    def __init__(self, db_path: str = "feedback_database.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with optimized schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create feedback table with proper schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_type TEXT NOT NULL,
                rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                comments TEXT,
                prediction TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_role TEXT,
                username TEXT,
                session_id TEXT,
                created_date DATE
            )
        ''')
        
        # Create indexes for fast queries on large datasets
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rating ON feedback(rating)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON feedback(created_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_role ON feedback(user_role)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_compound ON feedback(created_date, feedback_type, rating)')
        
        conn.commit()
        conn.close()
    
    def migrate_json_data(self, json_file: str = "feedback_data.json") -> bool:
        """Migrate existing JSON feedback data to database"""
        if not Path(json_file).exists():
            return False
        
        try:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            
            if not json_data:
                return False
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for feedback in json_data:
                # Convert timestamp to proper format
                timestamp_str = feedback.get('timestamp', datetime.now().isoformat())
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                except:
                    timestamp = datetime.now()
                
                cursor.execute('''
                    INSERT OR IGNORE INTO feedback 
                    (feedback_type, rating, comments, prediction, confidence, timestamp, created_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    feedback.get('type', 'Unknown'),
                    feedback.get('rating', 0),
                    feedback.get('comments', ''),
                    feedback.get('prediction', ''),
                    feedback.get('confidence', 0.0),
                    timestamp,
                    timestamp.date()
                ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            st.error(f"Migration error: {str(e)}")
            return False
    
    def add_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """Add new feedback to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = datetime.now()
            
            cursor.execute('''
                INSERT INTO feedback 
                (feedback_type, rating, comments, prediction, confidence, timestamp, 
                 user_role, username, session_id, created_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback_data.get('type', 'Unknown'),
                feedback_data.get('rating', 0),
                feedback_data.get('comments', ''),
                feedback_data.get('prediction', ''),
                feedback_data.get('confidence', 0.0),
                timestamp,
                feedback_data.get('user_role', ''),
                feedback_data.get('username', ''),
                feedback_data.get('session_id', ''),
                timestamp.date()
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            return False
    
    def get_feedback_count(self, filters: Dict[str, Any] = None) -> int:
        """Get total count of feedback entries with optional filters"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT COUNT(*) FROM feedback"
        params = []
        
        if filters:
            conditions = []
            if filters.get('feedback_type'):
                conditions.append("feedback_type = ?")
                params.append(filters['feedback_type'])
            if filters.get('rating'):
                conditions.append("rating = ?")
                params.append(filters['rating'])
            if filters.get('date_from'):
                conditions.append("created_date >= ?")
                params.append(filters['date_from'])
            if filters.get('date_to'):
                conditions.append("created_date <= ?")
                params.append(filters['date_to'])
            if filters.get('search_text'):
                conditions.append("(comments LIKE ? OR prediction LIKE ?)")
                search_term = f"%{filters['search_text']}%"
                params.extend([search_term, search_term])
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        cursor.execute(query, params)
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_feedback_data(self, 
                         limit: int = 50, 
                         offset: int = 0, 
                         filters: Dict[str, Any] = None,
                         sort_by: str = 'timestamp',
                         sort_order: str = 'DESC') -> List[Dict[str, Any]]:
        """Get paginated feedback data with filters and sorting"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT id, feedback_type, rating, comments, prediction, confidence, 
                   timestamp, user_role, username, created_date
            FROM feedback
        """
        params = []
        
        if filters:
            conditions = []
            if filters.get('feedback_type'):
                conditions.append("feedback_type = ?")
                params.append(filters['feedback_type'])
            if filters.get('rating'):
                conditions.append("rating = ?")
                params.append(filters['rating'])
            if filters.get('date_from'):
                conditions.append("created_date >= ?")
                params.append(filters['date_from'])
            if filters.get('date_to'):
                conditions.append("created_date <= ?")
                params.append(filters['date_to'])
            if filters.get('search_text'):
                conditions.append("(comments LIKE ? OR prediction LIKE ?)")
                search_term = f"%{filters['search_text']}%"
                params.extend([search_term, search_term])
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        query += f" ORDER BY {sort_by} {sort_order} LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive feedback statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Basic stats
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total_feedback = cursor.fetchone()[0]
        
        if total_feedback == 0:
            return {"total_feedback": 0}
        
        # Average rating
        cursor.execute("SELECT AVG(rating) FROM feedback WHERE rating > 0")
        avg_rating = cursor.fetchone()[0] or 0
        
        # Feedback type distribution
        cursor.execute("SELECT feedback_type, COUNT(*) FROM feedback GROUP BY feedback_type")
        type_distribution = dict(cursor.fetchall())
        
        # Rating distribution
        cursor.execute("SELECT rating, COUNT(*) FROM feedback GROUP BY rating")
        rating_distribution = dict(cursor.fetchall())
        
        # Recent stats (last 30 days)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).date()
        cursor.execute("SELECT COUNT(*) FROM feedback WHERE created_date >= ?", (thirty_days_ago,))
        recent_feedback = cursor.fetchone()[0]
        
        # Monthly trend (last 12 months)
        cursor.execute("""
            SELECT strftime('%Y-%m', created_date) as month, COUNT(*) 
            FROM feedback 
            WHERE created_date >= date('now', '-12 months')
            GROUP BY strftime('%Y-%m', created_date)
            ORDER BY month
        """)
        monthly_trend = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_feedback': total_feedback,
            'avg_rating': round(avg_rating, 2),
            'type_distribution': type_distribution,
            'rating_distribution': rating_distribution,
            'recent_feedback': recent_feedback,
            'monthly_trend': monthly_trend
        }
    
    def export_to_csv(self, filename: str = None, filters: Dict[str, Any] = None) -> str:
        """Export feedback data to CSV"""
        if not filename:
            filename = f"feedback_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Get all data matching filters
        all_data = []
        limit = 1000
        offset = 0
        
        while True:
            batch = self.get_feedback_data(limit=limit, offset=offset, filters=filters)
            if not batch:
                break
            all_data.extend(batch)
            offset += limit
        
        if all_data:
            df = pd.DataFrame(all_data)
            df.to_csv(filename, index=False)
            return filename
        return None

# Global database instance
feedback_db = FeedbackDatabase()