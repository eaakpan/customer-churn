import psycopg2
import psycopg2.extras as extras
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import os
from pyspark.sql import SparkSession


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

    def update(self, df, table):
        """
        Updates an existing table with values from a DataFrame, dynamically detecting key columns.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data to update.
            table (str): The name of the table to update.

        Returns:
            None
        """
        self.cur = self.conn.cursor()

        # Dynamically retrieve primary key columns for the table
        key_query = f"""
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = '{self.schema}.{table}'::regclass AND i.indisprimary;
        """
        self.cur.execute(key_query)
        key_columns = [row[0] for row in self.cur.fetchall()]

        if not key_columns:
            raise ValueError(f"Table '{table}' does not have a primary key. Key columns must be specified explicitly.")

        # Generate the SET clause dynamically from DataFrame columns
        update_columns = [col for col in df.columns if col not in key_columns]
        set_clause = ', '.join([f"{col} = EXCLUDED.{col}" for col in update_columns])

        # Generate the ON CONFLICT clause dynamically
        conflict_clause = ', '.join(key_columns)

        # Create a temporary table to stage data
        temp_table = f"{table}_temp"
        cols = ','.join(df.columns)
        query_create_temp = f"CREATE TEMP TABLE {temp_table} (LIKE {self.schema}.{table} INCLUDING ALL)"
        self.cur.execute(query_create_temp)

        # Insert data into the temporary table
        tuples = [tuple(x) for x in df.to_numpy()]
        query_insert_temp = f"INSERT INTO {temp_table} ({cols}) VALUES %s"
        extras.execute_values(self.cur, query_insert_temp, tuples)

        # Perform the update using the temporary table
        query_update = f"""
            INSERT INTO {self.schema}.{table} ({cols})
            SELECT {cols} FROM {temp_table}
            ON CONFLICT ({conflict_clause}) DO UPDATE
            SET {set_clause}
        """
        try:
            self.cur.execute(query_update)
            self.conn.commit()
            print("Table updated successfully.")
        except (Exception, psycopg2.DatabaseError) as error:
            print(f"Error: {error}")
            self.conn.rollback()
        finally:
            # Drop the temporary table
            query_drop_temp = f"DROP TABLE IF EXISTS {temp_table}"
            self.cur.execute(query_drop_temp)
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


################################### Synthetic Data Configs #############################################

churn_rates = { 'No':0.73463013,
                'Yes':0.26536987}


#########################################################################################################

########################################### PySpark #####################################################

class PySparkConnection:
    def __init__(self):
        os.environ['SPARK_HOME'] = os.path.join(os.getcwd(), 'venv', 'Lib', 'site-packages', 'pyspark')
        os.environ['HADOOP_HOME'] = os.path.join(os.getcwd(), 'venv', 'Lib', 'site-packages', 'pyspark')
        self.spark = SparkSession.builder\
            .master(os.getenv('PYSPARK_MASTER_URL'))\
            .appName(os.getenv('PYSPARK_APPNAME'))\
            .config(os.getenv('PYSPARK_CONFIG_KEY'), os.getenv('PYSPARK_PORT'))\
            .getOrCreate()
        self.postgres_url = f"jdbc:postgresql://{os.getenv('POSTGRES_HOST')}"\
                            f":{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DATABASE')}"
        self.properties = {
                "user": os.getenv('POSTGRES_USER'),
                "password": os.getenv('POSTGRES_PASSWORD'),
                "driver": "org.postgresql.Driver"
            }

    def read(self, table):
        return self.spark.read.jdbc(self.postgres_url, table, properties=self.properties)


#########################################################################################################