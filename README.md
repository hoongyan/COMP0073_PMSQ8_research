<!-- COMP0073 Research Project
This repository contains a research project that uses a PostgreSQL database with the pgvector extension for vector-based data storage (e.g., embeddings) and Alembic for database migrations. The project includes a scam_reports table with a vector(384) column for storing embeddings. This README provides instructions for setting up the environment and applying migrations.
Prerequisites

Docker: Install Docker Desktop or Docker CLI (instructions).
Python 3.13: Install Python 3.13 for local Alembic migrations (instructions).
Git: Ensure Git is installed to clone the repository (instructions).

Project Structure

docker/docker-compose.yml: Defines the PostgreSQL service with pgvector.
scripts/setup.sh: Script to initialize the Docker environment and enable the pgvector extension.
alembic/: Contains database migration scripts (e.g., 9e4158d25d25_initial_migration.py for the scam_reports table).
alembic.ini: Configuration for Alembic migrations.
.env: Environment variables for database connection.

Setup Instructions
Follow these steps to set up the project after cloning the repository:

1. Clone the Repository
   Clone the repository and navigate to the project directory:
   git clone <repository-url>
   cd comp0073_research

2. Create the .env File
   Create a .env file in the project root (comp0073_research/) with the following content:
   POSTGRES_USER=comp0073_user
   POSTGRES_PASSWORD=comp0073
   POSTGRES_DB=researchdb
   POSTGRES_PORT=5434

Ensure these values match the sqlalchemy.url in alembic.ini:
sqlalchemy.url = postgresql+psycopg://comp0073_user:comp0073@localhost:5434/researchdb

3. Run the Setup Script
   The setup.sh script starts the PostgreSQL database with the pgvector extension enabled:
   ./scripts/setup.sh

This will:

Validate the .env file.
Start the Docker container for PostgreSQL (comp0073_research_pgvector).
Enable the pgvector extension in the researchdb database.

Expected output:
.env file validated successfully
Starting Docker services...
Docker containers started successfully.
Enabling pgvector extension in researchdb...
pgvector extension enabled successfully.
Verifying pgvector extension in researchdb...
pgvector extension is verified successfully.
Setup complete! You can now run your Alembic migration:
alembic upgrade head

4. Set Up Python Environment
   Create a Python virtual environment and install dependencies:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

If requirements.txt doesn’t exist, install the required packages:
pip install alembic==1.13.3 sqlalchemy==2.0.35 psycopg==3.2.1 pgvector==0.3.0

5. Apply Database Migrations
   Run Alembic migrations to create the scam_reports table:
   alembic upgrade head

Verify the migration:
docker exec -it comp0073_research_pgvector psql -U comp0073_user -d researchdb -c "SELECT \* FROM alembic_version;"

Expected output:
version_num

---

9e4158d25d25
(1 row)

Check the scam_reports table structure:
docker exec -it comp0073_research_pgvector psql -U comp0073_user -d researchdb -c "\d scam_reports"

Ensure the embedding column has type vector(384).
Troubleshooting

Setup Script Fails:

Check Docker logs for errors:docker logs comp0073_research_pgvector

Ensure the .env file matches alembic.ini:sqlalchemy.url = postgresql+psycopg://comp0073_user:comp0073@localhost:5434/researchdb

If the pgvector extension isn’t enabled, the research_data volume may contain an existing database. Clear it (warning: this deletes all database data):docker-compose -f docker/docker-compose.yml down
docker volume rm comp0073_research_research_data
./scripts/setup.sh

Alembic Migration Fails:

Ensure pgvector is installed:pip show pgvector

Install if missing:pip install pgvector

Verify the migration script (alembic/versions/9e4158d25d25_initial_migration.py) includes:from pgvector.sqlalchemy import VECTOR

Test database connectivity:psql -h localhost -p 5434 -U comp0073_user -d researchdb

Use password comp0073.

Next Steps

Add your application code to interact with the scam_reports table (e.g., for research on scam report embeddings).
Share the repository with collaborators, who can follow these steps to replicate the setup.
For production or advanced collaboration, consider containerizing Alembic (see Containerizing Alembic).

Containerizing Alembic (Optional)
To run Alembic in a Docker container for enhanced reproducibility (e.g., for collaborators or production), add an app service to docker/docker-compose.yml:
services:
pgvector: # Existing pgvector service
app:
build:
context: .
dockerfile: docker/Dockerfile
container_name: comp0073_research_app
volumes: - .:/app
working_dir: /app
command: sh -c "alembic upgrade head && python app.py"
depends_on: - pgvector
environment: - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD} - POSTGRES_DB=${POSTGRES_DB} - POSTGRES_HOST=pgvector - POSTGRES_PORT=5432
restart: unless-stopped

Create docker/Dockerfile:
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

Create requirements.txt:
alembic==1.13.3
sqlalchemy==2.0.35
psycopg==3.2.1
pgvector==0.3.0

Update alembic.ini for Docker:
sqlalchemy.url = postgresql+psycopg://comp0073_user:comp0073@pgvector:5432/researchdb

Run the Docker services:
docker-compose

Important notes:
#Setup instructions
The project requires Python 3.13 with Tkinter support and dependencies listed in requirements.txt. Follow platform-specific instructions to set up a virtual environment. This is to ensure that the GUI is available. (to be done by users)
certain python environments may not have tkinter support -->

## Setup Instructions

### Prerequisites
- Docker and Docker Compose
- Python 3.8+
- Git
- Make

### Steps

1. **Clone the Repository**:
   ```bash
   git clone <your-repo-url>
   cd <your-repo>
   ```

2. **Set Up Environment Variables**:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` to ensure all variables are set:
     ```
     POSTGRES_USER=comp0073_user
     POSTGRES_PASSWORD=comp0073
     POSTGRES_DB=researchdb
     POSTGRES_PORT=5434
     DATABASE_URL=postgresql+psycopg://comp0073_user:comp0073@localhost:5434/researchdb
     OLLAMA_BASE_URL=http://localhost:11434
     ```

3. **Install Dependencies**:
   ```bash
   make install
   ```

4. **Set Up Docker, Schema, and Data**:
   - Run the full setup:
     ```bash
     make setup
     ```
   - Alternatively, run individual steps:
     ```bash
     ./setup.sh
     python scripts/init_db.py
     python scripts/load_data.py
     ```

5. **Test Retrieval**:
   ```bash
   make test
   ```
   - Or manually:
     ```bash
     python scripts/test_retrieval.py
     ```

## Notes
- Ensure Docker is running before executing scripts.
- Ensure `data/scam_report/dataset/scam_details.csv` exists.
- Logs are written to `application.log`, `database_operations.log`, `vector_operations.log`, `preprocessor.log`, and `data_load.log`.
- For development, set `DATABASE_ECHO=True` in `.env` to enable SQLAlchemy query logging.
```