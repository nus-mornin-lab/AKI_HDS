from sqlalchemy import create_engine
import pandas as pd


def getEngine():
	try:
		engine = create_engine('postgresql://maxpoon:@localhost:5432/mimic')
		connection = engine.connect()
		# sorry for this, we need to set schema this way
		pd.read_sql_query("""
		    set search_path to mimiciii;
		    SELECT COUNT(*) FROM patients;
		    """, con=connection)
		return connection
	except:
		print("Unable to connect to the database")
		return None

def getConnection():
	try:
		engine = getEngine()
		connection = engine.connect()
		return connection
	except:
		print("Unable to connect to the database")
		return None
