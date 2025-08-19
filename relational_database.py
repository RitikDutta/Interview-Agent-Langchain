# relational_database.py
# ----------------------
# Minimal MySQL wrapper with:
# - create_tables()  -> users, academic_summary
# - upsert_user()
# - ensure_academic_summary()
# - update_user_profile_partial() -> merge domain/skills/strengths/weaknesses

import os
import json
from typing import Optional, List, Dict, Any

import mysql.connector
from mysql.connector import pooling, Error


def _j(val) -> str:
    # store lists as JSON strings; safe & simple
    return json.dumps(val or [], ensure_ascii=False)


class RelationalDB:
    def __init__(self, pool_size: int = 5):
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
    # placeholder code , logic to be implemented later
    def user_exists_rdb(self, user_id: str) -> bool:
        sql = "SELECT 1 FROM users WHERE user_id=%s LIMIT 1"
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                return cur.fetchone() is not None
        finally:
            conn.close()

        

    def _conn(self):
        return self.pool.get_connection()

    def create_tables(self):
        create_users = """
        CREATE TABLE IF NOT EXISTS users (
            user_id VARCHAR(64) PRIMARY KEY,
            name VARCHAR(255),
            domain VARCHAR(255),
            skills TEXT,
            strengths TEXT,
            weaknesses TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB;
        """

        create_academic = """
        CREATE TABLE IF NOT EXISTS academic_summary (
            user_id VARCHAR(64) PRIMARY KEY,
            technical_accuracy TINYINT UNSIGNED NOT NULL DEFAULT 0,
            reasoning_depth TINYINT UNSIGNED NOT NULL DEFAULT 0,
            communication_clarity TINYINT UNSIGNED NOT NULL DEFAULT 0,
            score_overall DECIMAL(5,2) DEFAULT 0.00,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        ) ENGINE=InnoDB;
        """


        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(create_users)
            cur.execute(create_academic)
            conn.commit()
            print("[rdbms] tables ensured: users, academic_summary")
        finally:
            cur.close()
            conn.close()

    def upsert_user(
        self,
        user_id: str,
        name: Optional[str] = None,
        domain: Optional[str] = None,
        skills: Optional[List[str]] = None,
        strengths: Optional[List[str]] = None,
        weaknesses: Optional[List[str]] = None,
    ):
        # initialize as mostly empty; lists default to []
        sql = """
        INSERT INTO users (user_id, name, domain, skills, strengths, weaknesses)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            name=VALUES(name),
            domain=VALUES(domain),
            skills=VALUES(skills),
            strengths=VALUES(strengths),
            weaknesses=VALUES(weaknesses);
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
                    ),
                )
            conn.commit()
            print(f"[rdbms] users upsert OK  user_id={user_id}")
        finally:
            conn.close()

    def ensure_academic_summary(self, user_id: str):
        sql = """
        INSERT IGNORE INTO academic_summary (user_id, technical_accuracy, reasoning_depth, communication_clarity, score_overall)
        VALUES (%s, 0.00, 0.00, 0.00, 0.00);
        """
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
            conn.commit()
            print(f"[rdbms] academic_summary ensured  user_id={user_id}")
        finally:
            conn.close()

    def update_user_profile_partial(
        self,
        user_id: str,
        domain: Optional[str] = None,
        skills: Optional[List[str]] = None,
        strengths: Optional[List[str]] = None,
        weaknesses: Optional[List[str]] = None,
    ):
        """
        Merge updates into existing row: union-dedup of list fields, set domain if provided.
        """
        # 1) read existing
        conn = self._conn()
        try:
            with conn.cursor(dictionary=True) as cur:
                cur.execute("SELECT * FROM users WHERE user_id=%s", (user_id,))
                row = cur.fetchone()

            if not row:
                # if missing, create base and re-run
                self.upsert_user(user_id=user_id)
                with conn.cursor(dictionary=True) as cur:
                    cur.execute("SELECT * FROM users WHERE user_id=%s", (user_id,))
                    row = cur.fetchone()

            def _load_list(col):
                try:
                    return json.loads(row[col] or "[]") if col in row else []
                except Exception:
                    return []

            cur_skills = set(map(str.lower, _load_list("skills")))
            cur_str = set(_load_list("strengths"))
            cur_weak = set(_load_list("weaknesses"))

            if skills:
                cur_skills |= set(map(str.lower, skills))
            if strengths:
                cur_str |= set(strengths)
            if weaknesses:
                cur_weak |= set(weaknesses)

            new_domain = domain if domain is not None else row.get("domain")

            # 2) write back
            sql = """
            UPDATE users
               SET domain=%s, skills=%s, strengths=%s, weaknesses=%s
             WHERE user_id=%s
            """
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        new_domain,
                        _j(sorted(cur_skills)),
                        _j(sorted(cur_str)),
                        _j(sorted(cur_weak)),
                        user_id,
                    ),
                )
            conn.commit()
            print(f"[rdbms] users partial update OK  user_id={user_id} domain={new_domain}")
        finally:
            conn.close()

    def get_user_profile(self, user_id: str) -> dict | None:
        """
        Return a normalized profile row for this user_id or None if missing.
        List fields are parsed from JSON. Includes updated_at when present.
        """
        sql = "SELECT user_id, name, domain, skills, strengths, weaknesses, updated_at FROM users WHERE user_id=%s"
        conn = self._conn()
        try:
            with conn.cursor(dictionary=True) as cur:
                cur.execute(sql, (user_id,))
                row = cur.fetchone()
            if not row:
                print(f"[rdbms] get_user_profile: not found user_id={user_id}")
                return None

            # parse JSON list fields
            import json as _json
            def _parse(v):
                try:
                    return _json.loads(v) if isinstance(v, str) and v.strip() else []
                except Exception:
                    return []

            profile = {
                "user_id": row["user_id"],
                "name": row.get("name"),
                "domain": row.get("domain"),
                "skills": _parse(row.get("skills")),
                "strengths": _parse(row.get("strengths")),
                "weaknesses": _parse(row.get("weaknesses")),
                "updated_at": row.get("updated_at"),
            }
            print(f"[rdbms] get_user_profile OK  user_id={user_id}")
            return profile
        finally:
            conn.close()
