# 0. 사용할 패키지 불러오기
import numpy as np
import yaml
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree


def make_model():

    # 1. 데이터셋 불러오기
    df_origin = pd.read_csv('moodang_dataset.csv')

    target = (df_origin.iloc[:, 0].to_list())
    print(target)

    # # 데이터셋 object name yaml 파일에서 불러오기
    # with open('yolov5/customDataset/gachon_road.yaml') as f:
    #     object_name = yaml.load(f, Loader=yaml.FullLoader)
    #     data = list(object_name.values())
    #     df_name = data[5]
    #     df_name.insert(0, 'location')
    # df_origin = pd.read_csv('dataset.csv', names=df_name)

    # Convert categorical features to numeric values using oneHotEncoder
    encoder = LabelEncoder()
    df_origin['location'] = encoder.fit_transform(df_origin['location'])

    # X, Y 분류
    X = df_origin.drop(columns=['location'])
    y = df_origin[['location']]

    # decision tree model
    # model = DecisionTreeClassifier(random_state=42)
    model = DecisionTreeRegressor(min_samples_split=4, min_impurity_decrease=0.1, random_state=42)
    model.fit(X, y)

    # visualization
    fig = plt.figure(figsize=(30,40))
    fig.show(plot_tree(model, feature_names=X.columns, class_names=encoder.classes_, filled=True))

    # save model train result
    joblib.dump(model, './decision_model.pkl')

    # print
    print('Finish')

make_model()
