import os
import sys
from sys import argv
import pickle
import yaml

import numpy as np
import pandas as pd
import joblib
import woe.feature_process as fp
import woe.eval as eval

from database import DataBase
from ahp_score import AHP

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
    if(len(argv) < 2):
        raise("请输入是否是首次运行（首次运行会重新训练lr模型） 0:首次 1:非首次")
    init = True if sys.argv[1] == '0' else False

    with open('cfg.yaml', 'r') as f:
        cfg = yaml.load(f)

    # 配置训练样本数，正负样本各sampleNums个
    sampleNums = cfg['sampleNums']

    # databaseOperation
    databaseOperation = DataBase(cfg['db']['host'], cfg['db']['database'], cfg['db']['user'],
                                 cfg['db']['password'], cfg['db']['port'], cfg['db']['creditTable'])

    # 所有维度的查询
    sql = '''select s.supplier_id,s.name,s.fund,IF(s.createtime,TIMESTAMPDIFF(DAY ,s.createtime,CURRENT_DATE())/365,TIMESTAMPDIFF(DAY ,'2015-04-08 00:00:00',CURRENT_DATE())/365) as time,IF(s.supplier_area='["999999"]',34,LENGTH(s.supplier_area) - LENGTH(REPLACE(s.supplier_area,',',''))+1) as supplier_area,LENGTH(s.mian_materialclass) - LENGTH(REPLACE(s.mian_materialclass,',',''))+1 as mian_materialclass,LENGTH(s.sub_materialclass) - LENGTH(REPLACE(s.sub_materialclass,',',''))+1 as sub_materialclass,s.account_proportion,4-s.business_model as business_model,s.operationperiod > CURRENT_DATE() as operationperiod,4 - s.state as state,p.tender_purchaser_count,p.year_tender_count,p.tender_count,p.public_tender_ratio,p.invite_tender_ratio,p.cashfund_avg,p.iswin_count,p.year_iswin_count,p.failure_bid_count,p.year_failurebid_count,p.win_bid_ratio,p.win_bid_money,p.year_winbid_money,p.failure_bid_ratio,p.contract_count,p.end_contract_count,p.order_count,p.confirm_order_ratio,p.return_order_ratio,p.instock_count,p.invoice_count,p.pay_invoice_count,p.instock_honesty_average,p.instock_product_average,p.instock_service_average,p.instock_deliverspeed_average,p.service_level_average,p.price_level_average,p.company_level_average,p.bid_eval_count,p.operate_level,p.vip_order_money,p.mvp_order_money,p.market_order_money,risk1.repea as phone_repeat,risk2.repea as card_repeat,p.renew_click_count,p.project_click_count,p.like_click_count,p.lighthouse_click_count,finan_1.cnt as year_apply_cnt,finan_2.cnt as apply_cnt,finan_3.amount as year_apply_amount,finan_4.amount as apply_amount,finan_5.amount as weiyedai_apply_amount from zjc_intrec_supplier s left join zjc_intrec_supplier_param p on s.supplier_id=p.supplier_id LEFT JOIN (select linkmanphone,count(*) as repea from zjc_intrec_supplier where linkmanphone is true GROUP BY linkmanphone) risk1 on s.linkmanphone=risk1.linkmanphone LEFT JOIN (select card_number,count(*) as repea from zjc_intrec_supplier where card_number is true GROUP BY card_number) risk2 on s.card_number=risk2.card_number LEFT JOIN (SELECT company_name,count(*) as cnt from zjc_intrec_zlzrfinancial_data WHERE due_date is TRUE and DATE(due_date)>= DATE_SUB(CURDATE(), INTERVAL 1 YEAR) GROUP BY company_name) finan_1 on s.name=finan_1.company_name LEFT JOIN (SELECT company_name,count(*) as cnt from zjc_intrec_zlzrfinancial_data WHERE due_date is TRUE GROUP BY company_name) finan_2 on s.name=finan_2.company_name LEFT JOIN (SELECT company_name,SUM(money) as amount from zjc_intrec_zlzrfinancial_data WHERE due_date is TRUE and DATE(due_date)>= DATE_SUB(CURDATE(), INTERVAL 1 YEAR) GROUP BY company_name) finan_3 on s.name=finan_3.company_name LEFT JOIN (SELECT company_name,SUM(money) as amount from zjc_intrec_zlzrfinancial_data WHERE due_date is TRUE GROUP BY company_name) finan_4 on s.name=finan_4.company_name LEFT JOIN (SELECT company_name,SUM(money) as amount from zjc_intrec_financial_data GROUP BY company_name) finan_5 on s.name=finan_5.company_name where s.state=2 or s.state=4'''
    res = databaseOperation.readDataFromMysql(sql)
    res.fillna(0, inplace=True)
    columnName = ['supplier_id', 'company_name', 'fund', 'time', 'supplier_area', 'mian_materialclass', 'sub_materialclass', 'account_proportion', 'business_model', 'operationperiod', 'state', 'tender_purchaser_count', 'year_tender_count', 'tender_count', 'public_tender_ratio', 'invite_tender_ratio', 'cashfund_avg', 'iswin_count', 'year_iswin_count', 'failure_bid_count', 'year_failurebid_count', 'win_bid_ratio', 'win_bid_money', 'year_winbid_money', 'failure_bid_ratio', 'contract_count', 'end_contract_count', 'order_count', 'confirm_order_ratio',
                  'return_order_ratio', 'instock_count', 'invoice_count', 'pay_invoice_count', 'instock_honesty_average', 'instock_product_average', 'instock_service_average', 'instock_deliverspeed_average', 'service_level_average', 'price_level_average', 'company_level_average', 'bid_eval_count', 'operate_level', 'vip_order_money', 'mvp_order_money', 'market_order_money', 'phone_repeat', 'card_repeat', 'renew_click_count', 'project_click_count', 'like_click_count', 'lighthouse_click_count', 'year_apply_cnt', 'apply_cnt', 'year_apply_amount', 'apply_amount', 'weiyedai_apply_amount']
    res.columns = columnName

    if init:

        # 归一化
        param = np.array(res.iloc[:, 2:]).astype(np.float)
        uniform = MinMaxScaler()
        param = uniform.fit_transform(param)

        # 计算ahp分数
        total = np.array(cfg['weight']['totalWeight'])
        subList = [np.array(li) for li in cfg['weight']['subWeight'].values()]
        ahp = AHP(total, subList, param)

        score = ahp.compute()
        scoreDf = pd.concat([pd.DataFrame(score), res], axis=1)
        column = columnName.copy()
        column.insert(0, 'scores')
        scoreDf.columns = column
        scoreDf.sort_values('scores', ascending=False, inplace=True)

        # 插入label
        col_name = scoreDf.columns.tolist()
        col_name.insert(3, 'label')
        scoreDf = scoreDf.reindex(columns=col_name)
        scoreDf.reset_index(inplace=True, drop=True)
        num = scoreDf.shape[0]

        scoreDf.loc[:int(num/2), 'label'] = 1
        scoreDf.loc[int(num/2):, 'label'] = 0

        # 读取数据
        data = scoreDf.iloc[:, 3:]
        data.rename(columns={'label': 'target'}, inplace=True)
        data = data.astype('float')

        ''' woe分箱, iv and transform '''
        print("woe....")
        data_woe = data  # 用于存储所有数据的woe值
        info_value_list = []
        n_positive = sum(data['target'])
        n_negtive = len(data) - n_positive
        for column in list(data.columns[1:]):
            # if data[column].dtypes == 'object':
            #     info_value = fp.proc_woe_discrete(
            #         data, column, n_positive, n_negtive, 0.05*len(data), alpha=0.05)
            # else:

            info_value = fp.proc_woe_continuous(
                data, column, n_positive, n_negtive, 0.05*len(data), alpha=0.05)
            info_value_list.append(info_value)
            data_woe[column] = fp.woe_trans(data[column], info_value)

        folder = os.path.exists('./dataDump/')
        if not folder:
            os.makedirs('./dataDump/')
        info_df = eval.eval_feature_detail(
            info_value_list, './dataDump/woe_info.csv')

        folder = os.path.exists('./model/')
        if not folder:
            os.makedirs('./model/')

        with open('./model/woe_info.pkl', 'wb') as fw:
            pickle.dump(info_value_list, fw)

        # 删除iv值过小的变量
        iv_threshold = 0.001
        iv = info_df[['var_name', 'iv']].drop_duplicates()
        x_columns = list(iv.var_name[iv.iv > iv_threshold])

        with open('./model/woe_info_column.pkl', 'wb') as fw:
            pickle.dump(x_columns, fw)

        data_woe = data_woe[x_columns]
        data_woe.to_csv('./dataDump/data_woe.csv')

        # 训练数据&标签
        labels = np.append(np.array(data.iloc[:sampleNums, 0]), np.array(
            data.iloc[-sampleNums:, 0]))
        data_train = np.array(
            pd.concat([data_woe.iloc[:sampleNums, :], data_woe.iloc[-sampleNums:, :]]))

        # lr固定参数
        lr_other_params = {
            'class_weight': 'balanced',
            'n_jobs': -1
        }
        LR = LogisticRegression(**lr_other_params)

        # lr超参搜索
        lr_cv_params = {
            # 'penalty': ['l1', 'l2'],
            'penalty': ['l2'],
            'C': np.linspace(0.1, 6, 60),
            'max_iter': np.linspace(50, 1000, 20),
        }

        scoring = {
            'acc': make_scorer(accuracy_score),
            'roc_auc': make_scorer(roc_auc_score)
        }

        # Begin training
        print("training...")

        grid_lr = RandomizedSearchCV(LR, lr_cv_params, cv=5, refit='acc', n_iter=10,
                                     scoring=scoring, n_jobs=-1)
        grid_lr.fit(data_train, labels)

        lr_best_estimator = grid_lr.best_estimator_
        joblib.dump(lr_best_estimator, './model/lr_best_estimator.pkl')

    # 计算供应商分数
    lr_best_estimator = joblib.load('./model/lr_best_estimator.pkl')
    with open('./model/woe_info.pkl', 'rb') as fr:
        woe_info = pickle.load(fr)
    with open('./model/woe_info_column.pkl', 'rb') as fr:
        woe_info_column = pickle.load(fr)
    data_woe = res.iloc[:, 2:]

    for i in range(data_woe.shape[1]):
        data_woe.iloc[:, i] = fp.woe_trans(data_woe.iloc[:, i], woe_info[i])

    data_woe = data_woe[woe_info_column]
    pre = lr_best_estimator.predict_proba(data_woe)

    credit = 487.122+28.8539*np.log(pre[:, 1]/pre[:, 0])
    credit = pd.DataFrame(credit)
    credit = pd.concat([res['supplier_id'], credit], axis=1)
    credit.columns = ['supplier_id', 'score']

    # 情况供应商分数表，更新分数
    databaseOperation.deleteCreditScoreTable()
    databaseOperation.commitScore(credit)

    print('Update scores success!')
