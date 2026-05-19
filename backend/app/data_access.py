import duckdb
import os
import pandas as pd
from functools import lru_cache
from .settings import settings
import threading
_LOCK = threading.Lock()

@lru_cache(maxsize=1)
def _conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=1;")  # 병렬 parquet scan 꺼서 deadlock 회피
    return con

def _source_is_parquet(path: str) -> bool:
    return path.endswith(".parquet") or path.endswith("/") or path.endswith(".pq")

def _normalize_parquet_path(path: str) -> str:
    # DuckDB works best with globs for directories.
    if os.path.isdir(path):
        if path.endswith(os.sep):
            return path + "**/*.parquet"
        return path + os.sep + "**/*.parquet"
    return path

def list_batteries() -> list[str]:
    con = _conn()
    src = settings.DATA_SOURCE
    if _source_is_parquet(src):
        q = "SELECT DISTINCT battery FROM read_parquet(?) ORDER BY battery"
        src2 = _normalize_parquet_path(src)
        with _LOCK:
            return [r[0] for r in con.execute(q, [src2]).fetchall()]
    else:
        q = "SELECT DISTINCT battery FROM read_csv_auto(?) ORDER BY battery"
        with _LOCK:
            return [r[0] for r in con.execute(q, [src]).fetchall()]

def fetch_cycles(
    battery: str,
    start: int | None,
    end: int | None,
    stride: int,
    cols: list[str] | None,
) -> pd.DataFrame:
    con = _conn()
    src = settings.DATA_SOURCE
    reader = "read_parquet" if _source_is_parquet(src) else "read_csv_auto"
    where = ["battery = ?"]
    params: list[object] = [battery]

    if start is not None:
        where.append("cycle_num >= ?")
        params.append(int(start))
    if end is not None:
        where.append("cycle_num <= ?")
        params.append(int(end))

    col_sql = ", ".join([f'"{c}"' for c in cols]) if cols else "*"

    # Use QUALIFY with row_number for stride without full materialization
    q = f"""
        WITH t AS (
          SELECT {col_sql},
                 ROW_NUMBER() OVER (ORDER BY cycle_num) AS rn
          FROM {reader}(?)
          WHERE {" AND ".join(where)}
          ORDER BY cycle_num
        )
        SELECT * EXCLUDE (rn)
        FROM t
        WHERE ((rn - 1) % ?) = 0
    """
    src2 = _normalize_parquet_path(src) if _source_is_parquet(src) else src
    params2 = [src2, *params, int(max(stride, 1))]
    return con.execute(q, params2).df()
