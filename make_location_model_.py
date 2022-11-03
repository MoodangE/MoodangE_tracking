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
    target = ['Rotary1', 'Rotary2', 'Rotary3', 'Rotary4', 'Rotary5', 'Rotary6', 'Rotary7', 'Rotary8', 'Rotary9',
              'Rotary10', 'Rotary11', 'Rotary12', 'IT_MainGate1', 'IT_MainGate2', 'IT_MainGate3', 'IT_MainGate4',
              'IT_MainGate5', 'IT_MainGate6', 'AI1', 'AI2', 'AI3', 'AI4', 'AI5', 'AI6', 'AI7', 'AI8']

    # 1. 데이터셋 불러오기
    df_origin = pd.read_csv('moodang_dataset - 시트1.csv')

    # print(df_origin[['location']])

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
    print(X.columns)
    print(y)
    # decision tree model
    # model = DecisionTreeClassifier(random_state=0)
    model = DecisionTreeRegressor(min_samples_split=4, min_impurity_decrease=0.1, random_state=0)
    model.fit(X, y)

    # visualization
    fig = plt.figure(figsize=(15, 20))
    fig.show(plot_tree(model, feature_names=X.columns, class_names=encoder.__class__, filled=True))

    # save model train result
    joblib.dump(model, 'decision_model.pkl')


make_model()
