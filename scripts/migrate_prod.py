"""Run Alembic against the production (Supabase) database only.

The URL comes from `.env.prod` (or `--env-file`), never from `.env` or the
app settings, and is checked before anything that could open a connection
is imported. Refuses to run unless the URL points at the Supabase pooler and
not at a local database. Prints URLs with the password masked.

Usage:
    uv run python scripts/migrate_prod.py check     # who/where am I + vector
    uv run python scripts/migrate_prod.py current   # alembic current
    uv run python scripts/migrate_prod.py upgrade   # alembic upgrade head
    uv run python scripts/migrate_prod.py report    # tables + alembic_version
"""

import argparse
import re
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_HOST_FRAGMENT = "pooler.supabase.com"
FORBIDDEN_FRAGMENTS = ("localhost", "127.0.0.1", "@db:", "inboxpilot:")
PLACEHOLDER = "__FILL_PASSWORD__"


class UnsafeDatabaseUrl(RuntimeError):
    pass


def mask_url(url: str) -> str:
    return re.sub(r"(://[^:/@]+:)[^@]*@", r"\1****@", url)


def load_database_url(env_file: Path) -> str:
    if not env_file.is_file():
        raise UnsafeDatabaseUrl(f"{env_file} not found")
    url = dotenv_values(env_file).get("DATABASE_URL") or ""
    if not url:
        raise UnsafeDatabaseUrl(f"DATABASE_URL is missing from {env_file}")
    return url


def check_database_url(url: str) -> None:
    """Raise `UnsafeDatabaseUrl` unless `url` targets the Supabase pooler."""
    lowered = url.lower()
    for fragment in FORBIDDEN_FRAGMENTS:
        if fragment in lowered:
            raise UnsafeDatabaseUrl(
                f"refusing to run: DATABASE_URL looks local "
                f"(contains {fragment!r}): {mask_url(url)}"
            )
    if REQUIRED_HOST_FRAGMENT not in lowered:
        raise UnsafeDatabaseUrl(
            f"refusing to run: DATABASE_URL does not contain "
            f"{REQUIRED_HOST_FRAGMENT!r}: {mask_url(url)}"
        )
    if PLACEHOLDER in url:
        raise UnsafeDatabaseUrl(
            f"refusing to run: replace {PLACEHOLDER} in DATABASE_URL first"
        )


def to_sqlalchemy_url(url: str) -> str:
    # Only psycopg (v3) is installed; a bare postgresql:// would make
    # SQLAlchemy look for psycopg2.
    return re.sub(r"^postgres(ql)?://", "postgresql+psycopg://", url)


def run_check(url: str) -> None:
    import psycopg

    with psycopg.connect(url) as conn:
        user, database, addr = conn.execute(
            "select current_user, current_database(), inet_server_addr()"
        ).fetchone()
        vector = conn.execute(
            "select extname from pg_extension where extname = 'vector'"
        ).fetchone()
    print(f"current_user={user} current_database={database} server_addr={addr}")
    print(f"vector extension: {'present' if vector else 'MISSING'}")


def run_report(url: str) -> None:
    import psycopg

    with psycopg.connect(url) as conn:
        tables = conn.execute(
            "select tablename from pg_tables where schemaname = 'public' "
            "order by tablename"
        ).fetchall()
        versions = conn.execute("select version_num from alembic_version").fetchall()
    print("public tables:", ", ".join(name for (name,) in tables) or "(none)")
    print("alembic_version:", ", ".join(v for (v,) in versions) or "(none)")


def run_alembic(url: str, command_name: str) -> None:
    from alembic import command
    from alembic.config import Config

    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    # Set explicitly so alembic/env.py never falls back to app settings
    # (.env, which points at localhost). `%` must be escaped for configparser.
    config.set_main_option("sqlalchemy.url", to_sqlalchemy_url(url).replace("%", "%%"))
    if command_name == "current":
        command.current(config, verbose=True)
    else:
        command.upgrade(config, "head")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["check", "current", "upgrade", "report"])
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.prod")
    args = parser.parse_args(argv)

    try:
        url = load_database_url(args.env_file)
        check_database_url(url)
    except UnsafeDatabaseUrl as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"target: {mask_url(url)}")
    if args.action == "check":
        run_check(url)
    elif args.action == "report":
        run_report(url)
    else:
        run_alembic(url, args.action)
    return 0


if __name__ == "__main__":
    sys.exit(main())
