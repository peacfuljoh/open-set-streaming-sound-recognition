"""Database table definitions"""

from sqlalchemy import Column
from sqlalchemy import Integer, TIMESTAMP, Text
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class Raw(Base):
    __tablename__ = "raw"

    datetime = Column(TIMESTAMP, primary_key=True)
    max_amp = Column(Integer) # smallint
    filepath = Column(Text)

