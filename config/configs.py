import psycopg2
import psycopg2.extras as extras
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import os


load_dotenv(find_dotenv('config/.env'))

class MyDatabase:
    def __init__(self):
        self.db = os.getenv('POSTGRES_DATABASE')
        self.postgres_config = {
            "database": self.db,
            "user": os.getenv('POSTGRES_USER'),
            "password": os.getenv('POSTGRES_PASSWORD'),
            "host": os.getenv('POSTGRES_HOST'),
            "port": os.getenv('POSTGRES_PORT')
        }
        self.conn = psycopg2.connect(
            database=self.postgres_config['database'],
            user=self.postgres_config['user'],
            password=self.postgres_config['password'],
            host=self.postgres_config['host'],
            port=self.postgres_config['port'],
        )
        self.cur = self.conn.cursor()
        self.schema = self.db

    def create_schema(self, schema):
        self.cur = self.conn.cursor()
        query = f"""CREATE SCHEMA IF NOT EXISTS {schema} AUTHORIZATION postgres;"""
        self.cur.execute(query)
        self.conn.commit()
        self.schema = schema

    def drop_schema(self, schema):
        self.cur = self.conn.cursor()
        query = f"""DROP SCHEMA IF EXISTS {schema} CASCADE;"""
        self.cur.execute(query)
        self.conn.commit()

    def query(self, query):
        self.cur = self.conn.cursor()
        self.cur.execute(query)
        self.conn.commit()
        data = pd.read_sql_query(query, self.conn)
        return data

    def create_table(self, query):
        self.cur = self.conn.cursor()
        self.cur.execute(query)
        self.conn.commit()
        print("the table has been created")


    def insert(self, df, table):
        self.cur = self.conn.cursor()
        tuples = [tuple(x) for x in df.to_numpy()]

        cols = ','.join(list(df.columns))
        # SQL query to execute
        query = "INSERT INTO %s.%s(%s) VALUES %%s" % (self.schema,table, cols)
        try:
            extras.execute_values(self.cur, query, tuples)
            self.conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            self.conn.rollback()
            self.cur.close()
            return 1
        print("the dataframe is inserted")
        self.cur.close()

    def drop_table(self, table):
        self.cur = self.conn.cursor()
        query = f'''DROP TABLE IF EXISTS {table}'''
        self.cur.execute(query)
        self.conn.commit()
        print("the table has been dropped")

    def close(self):
        self.cur.close()
        self.conn.close()
