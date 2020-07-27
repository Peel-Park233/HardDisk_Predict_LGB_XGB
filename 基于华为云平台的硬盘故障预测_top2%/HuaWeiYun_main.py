#   基于华为云平台的硬盘故障预测
#   数据清洗已完成（空值数据，异常数据都删除或填充）
#   从数据库中传入数据为dataflow形式
from naie.datasets import samples
from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd
from naie.datasets import get_data_reference
from naie.context import Context as context
from naie.feature_processing import data_flow
import moxing as mox
import os
#   还需要导入华为云数据处理部分类似于jupterlab平台的库
import os
os.chdir("/home/ma-user/work/demo_0727")
from naie.context import Context as context
from naie.datasets import data_reference
from naie.feature_processing import data_flow
from naie.feature_analysis import data_analysis
from naie.feature_processing.expression import col, cols, cond, f_and, f_not, f_or
from naie.common.data.typedefinition import StepType, ColumnRelationship, JoinType, ColumnSelector,DynamicColumnsSelectorDetails, StaticColumnsSelectorDetails, ColumnsSelectorDetails, DataProcessMode


def timedata_processing(dataflow, time_length):
    #   训练集数据时间序列长度不一致，可以通过这个控制负样本的时间序列至少有多长
    #   时间序列长度相同
    #   本题提分最重要的部分，因为在正负样本比例差很大的情况下，负样本的数据对于lgb太重要了
    train_data = dataflow.to_pandas_dataframe()
    print(train_data.shape) #  时间序列优化前的长度
    train_data['value_count'] =train_data['serial_number'].map(train_data['serial_number'].value_counts())
    train_data = train_data[train_data['value_count'] >= time_length]
    train_data.drop('value_count', axis=1, inplace=True)
    train_data = train_data.sort_values('date_g', ascending=False)
    train_data = train_data.groupby('serial_number').head(time_length).reset_index(drop=True)
    # train_data.head(60)
    print(train_data.shape)  # 时间序列优化后的长度
    # dataflow = dataflow.create_dataflow_from_df(train_data)
    # dataflow.show_head()
    return train_data


#   考虑到normalized特征是对raw特征的某种归一化操作的结果，那么两者之间应该有潜在关系可以挖掘，
#   可以尝试将两者拿来做除法，产生新的特征
def create_new_features(train_data):
    nums = [1, 4, 5, 7, 9, 12, 184, 187, 188, 189, 190, 191, 192, 193, 194, 197, 198, 199]
    original_features = []

    for val in nums:
        train_data[str(val)] = train_data['smart_' + str(val) + '_raw'] / (
                    train_data['smart_' + str(val) + '_normalized'] + 0.2)
        original_features.append(str(val))
    train_data.head(10)

    dataflow = data_flow.create_dataflow_from_df(train_data)
    dataflow.show_head()

    return dataflow


#   删除时间强相关特征，防止过拟合
def dete_features(dataflow):
    use_regex = False
    columns = ["smart_9_normalized", "smart_9_raw", "smart_241_normalized", "smart_241_raw", "smart_242_normalized",
               "smart_242_raw"]
    term = '.*'
    columns = ColumnSelector(term, use_regex=True) if use_regex == True else columns
    current_dataflow = dataflow  # @param {"id":"default.current_dataflow"}
    dataflow = current_dataflow.drop_columns(columns)  # @return {"id":"default.dataflow"}
    dataflow.show_head()
    train_data = dataflow.to_pandas_dataframe()
    return train_data


#   产生有统计意义的特征
def statistical_features(train_data):
    nums = [1, 4, 5, 7, 9, 12, 184, 187, 188, 189, 190, 191, 192, 193, 194, 197, 198, 199]
    original_features = []

    for val in nums:
        train_data[str(val)] = train_data['smart_' + str(val) + '_raw'] / (
                    train_data['smart_' + str(val) + '_normalized'] + 0.1)
        original_features.append(str(val))
    train_data.head(10)

    new_data = pd.DataFrame()
    for col in original_features:
        print(col)
        new_data[col + '_mean'] = data.groupby('serial_number')[col].mean().values
        new_data[col + '_min'] = data.groupby('serial_number')[col].min().values
        new_data[col + '_max'] = data.groupby('serial_number')[col].max().values

        new_data[col + '_skew'] = data.groupby('serial_number')[col].skew().values
        new_data[col + '_nunique'] = data.groupby('serial_number')[col].nunique().values
        new_data[col + '_min_max'] = new_data[col + '_max'] - new_data[col + '_min']
        new_data[col + '_mean_max'] = new_data[col + '_max'] - new_data[col + '_mean']
        new_data[col + '_min_mean'] = new_data[col + '_mean'] - new_data[col + '_min']

    return new_data


def main_study(dataflow_train, test_data):
    id_column = "serial_number"  # 标志哪一块盘
    time_column = "date_g"  # 时间列
    target_column = "failure"  # 目标列

    from naie.automl import VegaAutoML
    automl = VegaAutoML(id_column=id_column,
                        time_column=time_column,
                        target_column=target_column,
                        model_type="time_series_classifier",
                        train_data_reference=dataflow_train,
                        included_models=["lightgbm", "xgbm"],   #   基于lgb和xgb做模型融合
                        metric="f1",
                        max_trial_number=2,
                        random_state=1)
    automl.train()

    # print model ranking list
    lb = automl.get_leader_board()
    print(lb)

    # get the best model
    model_best = automl.get_best_model()

    # predict the test data and evaluate the result
    feature_test = test_data
    y_predict_prob = model_best.predict_proba(feature_test)[:, 1]  # 输出正样本的概率
    threshold = 0.9965
    y_predict = (y_predict_prob > threshold).astype(int)

    id_index = feature_test[id_column].drop_duplicates().values.tolist()
    data_predict = pd.DataFrame({"SN": id_index, "label": y_predict})

    data_predict.to_csv("/cache/submit.csv", index=False)

    mox.file.copy("/cache/submit.csv", os.path.join(context.get_output_path(), "submit.csv"))


dr_train = get_data_reference("Default", "train")
dr_test = get_data_reference("Default", "test")
dataflow = get_data_reference("Default", "train")
#   使用DataReference对象全量获取pandas DataFrame
# train_data = dr_train.to_pandas_dataframe()
# test_data = dr_test.to_pandas_dataframe()

#   对训练数据进行时间序列长度统一，测试数据转化为pandas DataFrame
time_length = 26
train_data = timedata_processing(dr_train, time_length)
test_data = dr_test.to_pandas_dataframe()

#   生成新的特征列
dr_train = create_new_features(train_data)
dr_test = create_new_features(test_data)

#   删除时间强相关特征
train_data = dete_features(dr_train)
test_data = dete_features(dr_test)

#   主程序，训练及预测, 将训练数据转化为数据流类型，预测数据pandas DataFrame类型导入主函数
dataflow_train = data_flow.create_dataflow_from_df(train_data)
main_study(dataflow_train, test_data)