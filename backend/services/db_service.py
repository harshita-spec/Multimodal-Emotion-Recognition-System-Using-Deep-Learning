import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "analysis_history.db")

class DatabaseService:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = os.path.abspath(db_path)
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_records (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                final_emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                used_modalities TEXT NOT NULL,
                fusion_method TEXT NOT NULL,
                transcript TEXT,
                full_response_json TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def save_analysis(self, record_dict: Dict[str, Any]):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        modalities_str = ",".join(record_dict.get("used_modalities", []))
        json_str = json.dumps(record_dict)
        
        cursor.execute("""
            INSERT OR REPLACE INTO analysis_records
            (id, timestamp, final_emotion, confidence, used_modalities, fusion_method, transcript, full_response_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record_dict["id"],
            record_dict["timestamp"],
            record_dict["final_emotion"],
            record_dict["confidence"],
            modalities_str,
            record_dict["fusion_method"],
            record_dict.get("transcript"),
            json_str
        ))
        conn.commit()
        conn.close()

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, final_emotion, confidence, used_modalities, fusion_method
            FROM analysis_records
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for r in rows:
            history.append({
                "id": r[0],
                "timestamp": r[1],
                "final_emotion": r[2],
                "confidence": r[3],
                "used_modalities": r[4].split(",") if r[4] else [],
                "fusion_method": r[5]
            })
        return history

    def get_record_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT full_response_json FROM analysis_records WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
        return None

db_service = DatabaseService()
