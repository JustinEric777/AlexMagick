import sqlite3
import json
import time
import os
from typing import Dict, Any, List, Optional
from threading import Lock
from config.storage_config import SQLITE_DB_PATH

class DBManager:
    _instance = None
    _lock = Lock()
    
    def __new__(cls, db_path=None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DBManager, cls).__new__(cls)
                cls._instance._init_db(db_path or SQLITE_DB_PATH)
            return cls._instance

    def _init_db(self, db_path):
        self.db_path = db_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._create_table()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _create_table(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Create task_history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS task_history (
            id TEXT PRIMARY KEY,
            task_type TEXT NOT NULL,
            description TEXT,
            inputs TEXT,
            outputs TEXT,
            status TEXT,
            start_time REAL,
            end_time REAL,
            duration REAL,
            cpu_usage REAL,
            memory_usage REAL,
            error_msg TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create indexes for faster search
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_task_type ON task_history(task_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON task_history(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON task_history(created_at)')
        
        conn.commit()
        conn.close()

    def insert_task(self, task_data: Dict[str, Any]):
        conn = self._get_conn()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT INTO task_history (
                id, task_type, description, inputs, outputs, status, 
                start_time, end_time, duration, cpu_usage, memory_usage, error_msg
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task_data['id'],
                task_data['task_type'],
                task_data.get('description', ''),
                json.dumps(task_data.get('inputs', {})),
                json.dumps(task_data.get('outputs', {})),
                task_data['status'],
                task_data['start_time'],
                task_data.get('end_time'),
                task_data.get('duration'),
                task_data.get('cpu_usage', 0.0),
                task_data.get('memory_usage', 0.0),
                task_data.get('error_msg', '')
            ))
            conn.commit()
        except Exception as e:
            print(f"Error inserting task: {e}")
        finally:
            conn.close()

    def update_task(self, task_id: str, updates: Dict[str, Any]):
        conn = self._get_conn()
        cursor = conn.cursor()
        
        fields = []
        values = []
        for k, v in updates.items():
            fields.append(f"{k} = ?")
            if isinstance(v, (dict, list)):
                values.append(json.dumps(v))
            else:
                values.append(v)
        
        values.append(task_id)
        
        sql = f"UPDATE task_history SET {', '.join(fields)} WHERE id = ?"
        
        try:
            cursor.execute(sql, values)
            conn.commit()
        except Exception as e:
            print(f"Error updating task: {e}")
        finally:
            conn.close()

    def search_tasks(self, 
                     task_type: Optional[str] = None, 
                     keyword: Optional[str] = None, 
                     status: Optional[str] = None, 
                     limit: int = 20, 
                     offset: int = 0) -> List[Dict]:
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM task_history WHERE 1=1"
        params = []
        
        if task_type and task_type != "All":
            query += " AND task_type = ?"
            params.append(task_type)
            
        if status and status != "All":
            query += " AND status = ?"
            params.append(status)
            
        if keyword:
            query += " AND (id LIKE ? OR description LIKE ?)"
            keyword_param = f"%{keyword}%"
            params.extend([keyword_param, keyword_param])
            
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            results = []
            for row in rows:
                item = dict(row)
                # Parse JSON fields
                try: item['inputs'] = json.loads(item['inputs'])
                except: pass
                try: item['outputs'] = json.loads(item['outputs'])
                except: pass
                results.append(item)
            return results
        finally:
            conn.close()

    def get_task_by_id(self, task_id: str) -> Optional[Dict]:
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM task_history WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            if row:
                item = dict(row)
                try: item['inputs'] = json.loads(item['inputs'])
                except: pass
                try: item['outputs'] = json.loads(item['outputs'])
                except: pass
                return item
            return None
        finally:
            conn.close()
