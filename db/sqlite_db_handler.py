import time
from .base_db_handler import DBHandler

import sqlite3
from datetime import datetime
from common_utils.myutils import ModelResult

class SQLiteDBHandler(DBHandler):
    def __init__(self, db_path: str = "results.db"):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_results (
                scene_name  TEXT,
                model_name  TEXT,
                created_at  INTEGER,
                glb         BLOB,
                pointcloud  BOOLEAN,
                transform_x REAL,
                transform_y REAL,
                transform_z REAL,
                scale_sx    REAL,
                scale_sy    REAL,
                scale_sz    REAL,
                rotation_x  REAL,
                rotation_y  REAL,
                rotation_z  REAL,
                rotation_w  REAL,
                PRIMARY KEY (scene_name, model_name)
            )
        """)
        self.conn.commit()

    def disconnect(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def insert_result(self, model_name: str, result: ModelResult):
        if not self.conn:
            raise RuntimeError("Database is not connected")

        # current Unix timestamp (seconds)
        created_at = int(time.time())
        self.conn.execute(
            """
            INSERT OR REPLACE INTO model_results (
                scene_name,
                model_name,
                created_at,
                glb,
                pointcloud,
                transform_x,
                transform_y,
                transform_z,
                scale_sx,
                scale_sy,
                scale_sz,
                rotation_x,
                rotation_y,
                rotation_z,
                rotation_w
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.scene_name,
                model_name,
                created_at,
                result.output,
                result.is_pointcloud,
                result.transform[0],
                result.transform[1],
                result.transform[2],
                result.scale[0],
                result.scale[1],
                result.scale[2],
                result.rotation[0],
                result.rotation[1],
                result.rotation[2],
                result.rotation[3],
            )
        )
        self.conn.commit()
