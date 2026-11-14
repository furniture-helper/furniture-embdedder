import os
from queue import Empty, LifoQueue
from threading import Lock
from typing import Any

import boto3
import psycopg2
import psycopg2.extras


class PostgresConnector:
    """Simple PostgreSQL pool connector."""

    def __init__(
            self,
            host: str,
            user: str,
            database: str,
            port: int,
            minconn: int = 1,
            maxconn: int = 5,
    ) -> None:
        self.host = host
        self.user = user
        self.database = database
        self.port = port
        self.minconn = minconn
        self.maxconn = maxconn
        self._lock = Lock()
        self._pool = LifoQueue(maxsize=maxconn)
        self._created = 0
        self._rds_client = boto3.client("rds", region_name=self._resolve_region())
        self._sslmode = os.environ.get("PG_SSLMODE", "require")

        for _ in range(minconn):
            self._pool.put(self._create_connection())
            self._created += 1

    @staticmethod
    def _resolve_region() -> str:
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if region is None:
            raise ValueError("AWS_REGION or AWS_DEFAULT_REGION environment variable is required.")
        return region

    def _create_connection(self) -> Any:
        auth_token = self._rds_client.generate_db_auth_token(
            DBHostname=self.host,
            Port=self.port,
            DBUsername=self.user,
        )
        conn = psycopg2.connect(
            host=self.host,
            user=self.user,
            password=auth_token,
            dbname=self.database,
            port=self.port,
            sslmode=self._sslmode,
        )
        conn.autocommit = False
        return conn

    def _borrow_connection(self) -> Any:
        """Borrow a database connection from the internal pool."""
        try:
            return self._pool.get_nowait()
        except Empty:
            with self._lock:
                if self._created < self.maxconn:
                    self._created += 1
                    return self._create_connection()

            # Avoid deadlocking forever when every connection is checked out.
            # If the pool is exhausted, fail fast instead of blocking the caller.
            try:
                return self._pool.get(timeout=5)
            except Empty as exc:
                raise RuntimeError(
                    "PostgreSQL connection pool exhausted; all connections are busy or failed."
                ) from exc

    def _return_connection(self, connection: Any, close: bool = False) -> None:
        """Return a borrowed connection back to the internal pool."""
        if close:
            try:
                connection.close()
            except Exception:
                pass
            finally:
                with self._lock:
                    self._created = max(0, self._created - 1)
            return

        try:
            self._pool.put_nowait(connection)
        except Exception:
            try:
                connection.close()
            except Exception:
                pass
            with self._lock:
                self._created = max(0, self._created - 1)

    def run_query(self, query: str, params: Any = None) -> list[Any]:
        """Run a read query and return all rows."""
        connection = self._borrow_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(query, params)
            if cursor.description is None:
                return []
            return cursor.fetchall()
        finally:
            cursor.close()
            self._return_connection(connection)

    def execute(self, query: str, params: Any = None, retries: int = 1) -> int:
        connection = self._borrow_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(query, params)
            affected_rows = cursor.rowcount if cursor.rowcount is not None else 0
            connection.commit()
            return affected_rows
        except Exception as e:
            try:
                connection.rollback()
            except Exception:
                pass
            self._return_connection(connection, close=True)
            connection = None
            if retries > 0:
                return self.execute(query, params, retries - 1)
            raise e
        finally:
            cursor.close()
            if connection is not None:
                self._return_connection(connection)

    def execute_many(self, query: str, params_seq: list[Any], batch_size: int = 20,
                     template: str | None = None) -> None:
        """
        Execute an INSERT/UPSERT for multiple rows using psycopg2 execute_values,
        which sends the whole batch as a single round trip.
        """
        for i in range(0, len(params_seq), batch_size):
            batch = params_seq[i:i + batch_size]
            self._execute_batch_with_retry(query, batch, template=template)

    def _execute_batch_with_retry(self, query: str, batch: list[Any], retries: int = 1,
                                  template: str | None = None) -> None:
        connection = self._borrow_connection()
        cursor = connection.cursor()
        try:
            psycopg2.extras.execute_values(cursor, query, batch, template=template)
            connection.commit()
        except Exception as e:
            try:
                connection.rollback()
            except Exception:
                pass
            self._return_connection(connection, close=True)
            connection = None
            if retries > 0:
                self._execute_batch_with_retry(query, batch, retries - 1, template=template)
            else:
                raise e
        finally:
            cursor.close()
            if connection is not None:
                self._return_connection(connection)

    def close(self) -> None:
        """Close all pooled database connections and reset the pool."""
        while True:
            try:
                connection = self._pool.get_nowait()
                connection.close()
            except Empty:
                break
        with self._lock:
            self._created = 0


__all__ = ["PostgresConnector"]

postgres_connector = PostgresConnector(
    host=os.environ["PG_HOST"],
    port=int(os.environ["PG_PORT"]),
    user=os.environ["RDS_IAM_USER"],
    database=os.environ["PG_DATABASE"],
)
