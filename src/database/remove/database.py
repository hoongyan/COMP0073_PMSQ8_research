from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

# Fetch database URL
URL_DATABASE = os.getenv("DATABASE_URL")
if not URL_DATABASE:
    raise ValueError("DATABASE_URL not found in environment variables.")

# Create SQLAlchemy engine
try:
    engine = create_engine(
        URL_DATABASE,
        echo=False  # Disable in production
    )
    print("Database engine created successfully!")
except Exception as e:
    print(f"Failed to create database engine: {e}")
    raise

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def test_db_connection():
    """
    Test connection to the PostgreSQL database.
    """
    try:
        with SessionLocal() as session:
            result = session.execute(text("SELECT NOW();"))
            print("Current Time:", result.fetchone()[0])
            print("Database connection test successful!")
    except Exception as e:
        print(f"Database connection test failed: {e}")
        raise

def test_pgvector_extension():
    """
    Test and enable pgvector extension if not already enabled.
    """
    try:
        with SessionLocal() as session:
            result = session.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector';"))
            if result.fetchone():
                print("pgvector extension is enabled!")
            else:
                print("pgvector extension is not enabled. Enabling now...")
                session.execute(text("CREATE EXTENSION vector;"))
                session.commit()
                print("pgvector extension enabled successfully!")
    except Exception as e:
        print(f"Failed to check or enable pgvector extension: {e}")
        raise

if __name__ == "__main__":
    test_db_connection()
    test_pgvector_extension()