"""Utils for interacting with the database"""

from typing import List, Optional
from urllib.parse import quote_plus as urlquote

import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, update

from src.constants_stream import DB_CONFIG
from src.models.db_models import Raw
from ossr_utils.misc_utils import print_df_full, convert_utc_to_dt
from src.constants_stream import DATA_DIR





### DATA MANIPULATION ###
def insert_or_update_db(tablename: str,
                        df: pd.DataFrame,
                        op: str):
    """Insert new data to a table"""
    assert type(df) is pd.DataFrame
    assert op in ['insert', 'update']

    num_rows = len(df)

    session, engine, conn = get_db_session()

    if op == 'insert':
        print('insert_or_update_db() -> Inserting {} records to the {} table'.format(num_rows, tablename))

        df.to_sql(tablename, con=engine, if_exists='append', index=False)
    elif op == 'update':
        assert tablename in ['raw']

        print('insert_or_update_db() -> Updating {} records in the {} table'.format(num_rows, tablename))

        model = get_table_from_name(tablename)
        for _, row in df.iterrows():
            if tablename == 'raw':
                col = model.datetime
                key = 'datetime'
            stmt = update(model).where(col == row[key]).values(row.to_dict())
            conn.execute(stmt)

    close_db_sess(session, engine)

def close_db_sess(session, engine):
    session.close()
    engine.dispose()

def execute_pure_sql(statement: str) -> pd.DataFrame:
    """Execute pure SQL query"""
    session, engine, conn = get_db_session()
    df = pd.read_sql_query(statement, engine)
    close_db_sess(session, engine)
    return df

def execute_pure_sql_no_return(statement: str):
    session, engine, conn = get_db_session()
    conn.execute(statement)
    close_db_sess(session, engine)

def get_records(tablename: str,
                limit: Optional[int] = None) \
        -> pd.DataFrame:
    """Get the records in a database table associated with the specified session ID"""
    assert tablename in ['raw']

    model = get_table_from_name(tablename)
    session, engine, conn = get_db_session()

    query = session.query(model)

    if limit is not None:
        query = query.limit(limit)

    df = pd.read_sql(query.statement, query.session.bind)

    close_db_sess(session, engine)

    return df




### CONFIG, SESSION ###
def get_db_session():
    """Initialize DB session"""
    config = DB_CONFIG

    db_uri = '{}://{}:{}@{}/{}'.format(
        config['dialect'],
        config['user'],
        urlquote(config['password']),
        config['host'],
        config['database']
    )

    engine = create_engine(db_uri)
    conn = engine.connect()
    Session = sessionmaker(bind=engine)
    session = Session()

    return session, engine, conn




### MISC ###
def get_table_from_name(tablename: str):
    if tablename == 'raw':
        return Raw

def get_table_col_names(tablename: str) -> List[str]:
    table = get_table_from_name(tablename)
    col_names = [c.name for c in table.__table__.columns]
    return col_names









if __name__ == '__main__':
    tablename = 'raw'
    table = get_table_from_name(tablename)

    import os

    sess_ids = [1671571740613, 1671573697960]

    for sess_id in sess_ids:
        dpath = os.path.join(DATA_DIR, str(sess_id))
        ts = [int(f[:-4]) / 1e3 for f in sorted(os.listdir(dpath)) if '.wav' in f]
        df = pd.DataFrame({'datetime': [convert_utc_to_dt(ts_) for ts_ in ts]})
        # print_df_full(df)
        insert_or_update_db('raw', df, 'insert')
