from sqlalchemy import create_engine
import pandas as pd

user = input("DB user name: ")
address = input("DB address: ")
port = input("DB port number: ")


def getEngine():
    try:
        engine = create_engine('postgresql://{user}:@{address}:{port}/mimic'.format(
            user=user,
            address=address,
            port=port
        ))
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
