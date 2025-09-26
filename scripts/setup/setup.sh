#!/bin/bash

# Exit on any error
set -e

# Define paths
COMPOSE_FILE="docker/docker-compose.yml"
ENV_FILE=".env"
LOG_DIR="logs/setup"
LOG_FILE="$LOG_DIR/setup.log"

# Setup logging
mkdir -p "$LOG_DIR"
echo "$(date '+%Y-%m-%d %H:%M:%S') - setup - INFO - Initializing setup script" >> "$LOG_FILE"

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found in the root directory."
    echo "Please create $ENV_FILE with POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, and POSTGRES_PORT."
    exit 1
fi

# Validate required .env variables
source "$ENV_FILE"
for var in POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DB POSTGRES_PORT; do
    if [ -z "${!var}" ]; then
        echo "Error: $var is not set in $ENV_FILE."
        exit 1
    fi
done
echo ".env file validated successfully"

#Check if docker-compose.yml exists
if [ ! -f "$COMPOSE_FILE" ]; then
    echo "Error: $COMPOSE_FILE not found."
    exit 1
fi

# Build and start Docker services
echo "Starting Docker services..."
docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up --build -d
if [ $? -eq 0 ]; then
    echo "Docker containers started successfully."
else
    echo "Failed to start Docker containers. Check logs:"
    docker logs comp0073_research_pgvector
    exit 1
fi

# Enable pgvector extension
echo "Enabling pgvector extension in $POSTGRES_DB..."
sleep 5 # Wait for PostgreSQL to be ready
docker exec comp0073_research_pgvector psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "CREATE EXTENSION IF NOT EXISTS vector;"
if [ $? -eq 0 ]; then
    echo "pgvector extension enabled successfully."
else
    echo "Error: Failed to enable pgvector extension. Check logs:"
    docker logs comp0073_research_pgvector
    exit 1
fi

# Verify pgvector extension is enabled
echo "Verifying pgvector extension in $POSTGRES_DB..."
docker exec comp0073_research_pgvector psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT * FROM pg_extension WHERE extname = 'vector';" | grep -q "vector" && {
    echo "pgvector extension is verified successfully."
} || {
    echo "Error: pgvector extension not enabled. Check logs:"
    docker logs comp0073_research_pgvector
    exit 1
}

# Apply Alembic migrations
echo "Applying Alembic migrations..."
alembic upgrade head
if [ $? -eq 0 ]; then
    echo "Alembic migrations applied successfully."
else
    echo "Error: Failed to apply Alembic migrations."
    exit 1
fi

echo "Setup complete. Run 'python scripts/init_db.py' to initialize the database schema, then 'python scripts/load_data.py' to load data."
echo "Alternatively, use 'make setup' to run all steps."
