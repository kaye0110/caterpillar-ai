import os

import pandas as pd

from src.config.AppConfig import AppConfig


def main():
    # Initialize configuration
    config = AppConfig()
    logger = config.logger

    # Validate connections
    if not config.validate_mysql_connection():
        return
    if not config.validate_redis_connection():
        return
    if not config.validate_es_connection():
        return

    # Connect to MySQL
    try:
        conn = config.get_mysql_connection()
        logger.info("Connected to MySQL database.")
    except Exception as e:
        logger.error(f"Error connecting to MySQL: {e}")
        return

    # Read data from MySQL
    query = "SELECT * FROM ts_stock_basic"
    try:
        df = pd.read_sql(query, conn)
        logger.info("Data read from MySQL successfully.")
    except Exception as e:
        logger.error(f"Error reading data from MySQL: {e}")
        return
    finally:
        conn.close()

    # Define the output path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    output_path = os.path.join(project_root, 'resources', 'data', 'aaa.parquet')

    # Write data to Parquet file
    try:
        df.to_parquet(output_path, index=False)
        logger.info(f"Data written to {output_path} successfully.")
    except Exception as e:
        logger.error(f"Error writing data to Parquet: {e}")


if __name__ == "__main__":
    main()
