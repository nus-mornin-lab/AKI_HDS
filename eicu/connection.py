from sqlalchemy import create_engine
import pandas as pd
import psycopg2


def getEngine():
	try:
		connection = psycopg2.connect(dbname='eicu', user='', password='')
		# sorry for this, we need to set schema this way
		pd.read_sql_query("""
		    set search_path to eicu;
		    SELECT COUNT(*) FROM patient;
		    """, con=connection)
		return connection
	except:
		print("Unable to connect to the database")
		return None

def getConnection():
	try:
		connection = getEngine()
		return connection
	except:
		print("Unable to connect to the database")
		return None
