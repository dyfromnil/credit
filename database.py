import pymysql
import pandas as pd

from sqlalchemy import create_engine


class DataBase:
    def __init__(self, host, database, user, password, port, creditTable) -> None:
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.creditTable = creditTable

    def readDataFromMysql(self, sql_order):
        '''
        查询
        '''
        db = pymysql.connect(self.host, self.user,
                             self.password, self.database)
        cursor = db.cursor()
        try:
            cursor.execute(sql_order)
            data = cursor.fetchall()
            data = pd.DataFrame(list(data))
            print("Retrieve success!\n")
        except:
            data = pd.DataFrame()
        db.close()
        return data

    # 清空分数表
    def deleteCreditScoreTable(self):
        db = pymysql.connect(self.host, self.user,
                             self.password, self.database)
        cursor = db.cursor()
        sql = 'delete from '+self.creditTable
        cursor.execute(sql)
        sql = 'ALTER TABLE '+self.creditTable+' AUTO_INCREMENT=1'
        cursor.execute(sql)
        db.commit()
        db.close()

    # 提交
    def commitScore(self, creditScore):
        # con=create_engine("mysql+pymysql://username:password@host(:port)/database",encoding='utf-8')
        info = 'mysql+pymysql://'+self.user+':'+self.password + \
            '@'+self.host+':'+self.port+'/'+self.database
        con = create_engine(info, encoding='utf-8')

        creditScore.to_sql(name=self.creditTable, con=con,
                           if_exists='append', index=False)
