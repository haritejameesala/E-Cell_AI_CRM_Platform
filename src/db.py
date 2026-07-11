from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# Falls back to a local dev DB if DATABASE_URL isn't set - fine for local
# work, but make sure it's actually configured in prod.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:Hari%40NITcrm2026@localhost/crm_db",
)

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()