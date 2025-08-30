# relational_database.py

import os
import json
from log_utils import get_logger
from typing import Optional, List, Dict, Any

import mysql.connector
from mysql.connector import pooling, Error
from dotenv import load_dotenv
load_dotenv()


def _j(val) -> str:
    # store lists as JSON strings; safe & simple
    return json.dumps(val or [], ensure_ascii=False)


class RelationalDB:
    def __init__(self, pool_size: int = 5):
        self.logger = get_logger("rdbms")
        host = os.getenv("MYSQL_HOST")
        user = os.getenv("MYSQL_USER")
        password = os.getenv("MYSQL_PASSWORD")
        database = os.getenv("MYSQL_DB")
        if not all([host, user, password, database]):
            raise ValueError("MYSQL_* env vars missing")

        try:
            self.pool = pooling.MySQLConnectionPool(
                pool_name="langgraph_pool",
                pool_size=pool_size,
                pool_reset_session=True,
                host=host,
                user=user,
                password=password,
                database=database,
            )
        except Error as e:
            raise RuntimeError(f"MySQL pool error: {e}")

    def _conn(self):
        return self.pool.get_connection()

    # ----------------- utilities -----------------
    def user_exists_rdb(self, user_id: str) -> bool:
        sql = "SELECT 1 FROM users WHERE user_id=%s LIMIT 1"
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                return cur.fetchone() is not None
        finally:
            conn.close()

    # ----------------- schema -----------------
    def create_tables(self):
        # users table now includes 'categories' (TEXT JSON-string)
        create_users = """
        CREATE TABLE IF NOT EXISTS users (
            user_id VARCHAR(64) PRIMARY KEY,
            name VARCHAR(255),
            domain VARCHAR(255),
            skills TEXT,
            strengths TEXT,
            weaknesses TEXT,
            categories TEXT,
            thread_id VARCHAR(255),
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB;
        """

        # academic_summary with explicit metric columns
        create_academic = """
        CREATE TABLE IF NOT EXISTS academic_summary (
            user_id VARCHAR(64) PRIMARY KEY,
            question_attempted INT UNSIGNED NOT NULL DEFAULT 0,
            technical_accuracy DECIMAL(5,2) NOT NULL DEFAULT 0.00,
            reasoning_depth DECIMAL(5,2) NOT NULL DEFAULT 0.00,
            communication_clarity DECIMAL(5,2) NOT NULL DEFAULT 0.00,
            score_overall DECIMAL(5,2) DEFAULT 0.00,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        ) ENGINE=InnoDB;
        """

        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(create_users)
            cur.execute(create_academic)

            # Backward-compat: add categories if an older users table exists without it
            cur.execute("SHOW COLUMNS FROM users LIKE 'categories';")
            if not cur.fetchone():
                try:
                    cur.execute("ALTER TABLE users ADD COLUMN categories TEXT;")
                    self.logger.info("users: added missing 'categories' column")
                except Exception as e:
                    self.logger.warning(f"users: could not add 'categories' column ({e})")

            # Backward-compat: add thread_id if missing
            cur.execute("SHOW COLUMNS FROM users LIKE 'thread_id';")
            if not cur.fetchone():
                try:
                    cur.execute("ALTER TABLE users ADD COLUMN thread_id VARCHAR(255);")
                    self.logger.info("users: added missing 'thread_id' column")
                except Exception as e:
                    self.logger.warning(f"users: could not add 'thread_id' column ({e})")

            conn.commit()
            self.logger.info("tables ensured: users, academic_summary")
        finally:
            cur.close()
            conn.close()

    # ----------------- writes -----------------
    def upsert_user(
        self,
        user_id: str,
        name: Optional[str] = None,
        domain: Optional[str] = None,
        skills: Optional[List[str]] = None,
        strengths: Optional[List[str]] = None,
        weaknesses: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
    ):
        sql = """
        INSERT INTO users (user_id, name, domain, skills, strengths, weaknesses, categories, thread_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            name=VALUES(name),
            domain=VALUES(domain),
            skills=VALUES(skills),
            strengths=VALUES(strengths),
            weaknesses=VALUES(weaknesses),
            categories=VALUES(categories),
            thread_id=VALUES(thread_id);
        """
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        user_id,
                        name,
                        domain,
                        _j(skills),
                        _j(strengths),
                        _j(weaknesses),
                        _j(categories),
                        thread_id,
                    ),
                )
            conn.commit()
            self.logger.info(f"users upsert OK  user_id={user_id}")
        finally:
            conn.close()

    def ensure_academic_summary(self, user_id: str):
        # match the actual column names/types
        sql = """
        INSERT IGNORE INTO academic_summary
            (user_id, question_attempted, technical_accuracy, reasoning_depth, communication_clarity, score_overall)
        VALUES (%s, 0, 0.00, 0.00, 0.00, 0.00);
        """
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
            conn.commit()
            self.logger.info(f"academic_summary ensured  user_id={user_id}")
        finally:
            conn.close()

    def update_user_profile_partial(
        self,
        user_id: str,
        domain: Optional[str] = None,
        skills: Optional[List[str]] = None,
        strengths: Optional[List[str]] = None,
        weaknesses: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
    ):
        conn = self._conn()
        try:
            with conn.cursor(dictionary=True) as cur:
                cur.execute("SELECT * FROM users WHERE user_id=%s", (user_id,))
                row = cur.fetchone()

            if not row:
                self.upsert_user(user_id=user_id)
                with conn.cursor(dictionary=True) as cur:
                    cur.execute("SELECT * FROM users WHERE user_id=%s", (user_id,))
                    row = cur.fetchone()

            def _load_list(col):
                try:
                    return json.loads(row[col] or "[]") if col in row else []
                except Exception:
                    return []

            cur_skills = set(map(str, _load_list("skills")))
            cur_str = set(_load_list("strengths"))
            cur_weak = set(_load_list("weaknesses"))
            cur_cats = set(_load_list("categories"))

            if skills: cur_skills |= set(map(str, skills))
            if strengths: cur_str |= set(strengths)
            if weaknesses: cur_weak |= set(weaknesses)
            if categories: cur_cats |= set(categories)

            new_domain = domain if domain is not None else row.get("domain")
            new_thread_id = thread_id if thread_id is not None else row.get("thread_id")

            sql = """
            UPDATE users
            SET domain=%s, skills=%s, strengths=%s, weaknesses=%s, categories=%s, thread_id=%s
            WHERE user_id=%s
            """
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        new_domain,
                        _j(sorted(cur_skills, key=str.lower)),
                        _j(sorted(cur_str)),
                        _j(sorted(cur_weak)),
                        _j(sorted(cur_cats)),
                        new_thread_id,
                        user_id,
                    ),
                )
            conn.commit()
            self.logger.info(f"users partial update OK  user_id={user_id} domain={new_domain} thread_id={new_thread_id}")
        finally:
            conn.close()

    def set_user_thread_id(self, user_id: str, thread_id: str):
        sql = "UPDATE users SET thread_id=%s WHERE user_id=%s"
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (thread_id, user_id))
            conn.commit()
            self.logger.info(f"thread_id set  user_id={user_id} thread_id={thread_id}")
        finally:
            conn.close()


    # ----------------- reads -----------------
    def get_user_profile(self, user_id: str) -> dict | None:
        sql = """
            SELECT user_id, name, domain, skills, strengths, weaknesses, categories, thread_id, updated_at
            FROM users WHERE user_id=%s
        """
        conn = self._conn()
        try:
            with conn.cursor(dictionary=True) as cur:
                cur.execute(sql, (user_id,))
                row = cur.fetchone()
            if not row:
                self.logger.debug(f"get_user_profile: not found user_id={user_id}")
                return None

            def _parse(v):
                try:
                    return json.loads(v) if isinstance(v, str) and v.strip() else []
                except Exception:
                    return []

            profile = {
                "user_id": row["user_id"],
                "name": row.get("name"),
                "domain": row.get("domain"),
                "skills": _parse(row.get("skills")),
                "strengths": _parse(row.get("strengths")),
                "weaknesses": _parse(row.get("weaknesses")),
                "categories": _parse(row.get("categories")),
                "thread_id": row.get("thread_id"),
                "updated_at": row.get("updated_at"),
            }
            self.logger.debug(f"get_user_profile OK  user_id={user_id}")
            return profile
        finally:
            conn.close()

    def get_user_academic_score(self, user_id: str) -> dict | None:
        """
        Return academic scores for this user_id or None if missing.
        Uses the correct column names from schema.
        """
        sql = """
            SELECT user_id,
                   question_attempted,
                   technical_accuracy,
                   reasoning_depth,
                   communication_clarity,
                   score_overall
            FROM academic_summary
            WHERE user_id=%s
        """
        conn = self._conn()
        try:
            with conn.cursor(dictionary=True) as cur:
                cur.execute(sql, (user_id,))
                row = cur.fetchone()
            if not row:
                self.logger.debug(f"get_user_academic_score: not found user_id={user_id}")
                return None

            scores = {
                "user_id": row["user_id"],
                "question_attempted": int(row.get("question_attempted", 0)),
                "technical_accuracy": float(row.get("technical_accuracy", 0)),
                "reasoning_depth": float(row.get("reasoning_depth", 0)),
                "communication_clarity": float(row.get("communication_clarity", 0)),
                "score_overall": float(row.get("score_overall", 0.0)),
            }
            self.logger.debug(f"get_user_academic_score OK  user_id={user_id}")
            return scores
        finally:
            conn.close()
            
    def get_user_thread_id(self, user_id: str) -> Optional[str]:
        sql = "SELECT thread_id FROM users WHERE user_id=%s"
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                row = cur.fetchone()
            return row[0] if row and row[0] else None
        finally:
            conn.close()

    

    def update_academic_score(
        self,
        user_id: str,
        question_attempted: Optional[int] = None,
        technical_accuracy: Optional[int] = None,
        reasoning_depth: Optional[int] = None,
        communication_clarity: Optional[int] = None,
        score_overall: Optional[float] = None,
    ):
        """
        Update academic scores for a user. Only updates the fields provided.
        """
        sets, vals = [], []
        if technical_accuracy is not None:
            sets.append("technical_accuracy=%s")
            vals.append(float(technical_accuracy))
        if reasoning_depth is not None:
            sets.append("reasoning_depth=%s")
            vals.append(float(reasoning_depth))
        if communication_clarity is not None:
            sets.append("communication_clarity=%s")
            vals.append(float(communication_clarity))
        if score_overall is not None:
            sets.append("score_overall=%s")
            vals.append(float(score_overall))
        if question_attempted is not None:
            sets.append("question_attempted=%s")
            vals.append(int(question_attempted))

        if not sets:
            self.logger.debug(f"update_academic_score: nothing to update for user_id={user_id}")
            return

        sql = f"UPDATE academic_summary SET {', '.join(sets)} WHERE user_id=%s"
        vals.append(user_id)

        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(vals))
            conn.commit()
            self.logger.info(f"academic_summary updated user_id={user_id}")
        finally:
            conn.close()
