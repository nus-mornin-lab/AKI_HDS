from sqlalchemy import create_engine
import pandas as pd

engine = None

def getEngine():
	global engine
	if engine:
		return engine
	try:
		engine = create_engine('postgresql://maxpoon:@localhost:5432/mimic')
		connection = engine.connect()
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
