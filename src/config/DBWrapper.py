import logging

import mysql.connector

from src.config.AppConfig import AppConfig


class DBWrapper:
    def __init__(self):
        self.config = AppConfig()
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        """Enter the runtime context related to this object."""
        self.conn = self.config.get_mysql_connection()
        self.cursor = self.conn.cursor(dictionary=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context related to this object."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def execute_query(self, query, params=None):
        """Execute a query and return the results."""
        try:
            # self.logger.info("Executing query: %s %s", query, params)
            self.cursor.execute(query, params or ())
            return self.cursor.fetchall()
        except mysql.connector.Error as err:
            self.logger.error(f"Error executing query: {err}")
            return None
