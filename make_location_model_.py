def make_model():
    # 0. 사용할 패키지 불러오기
    import numpy as np
    import yaml
    import joblib
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.tree import plot_tree

    # 1. 데이터셋 불러오기
    df_origin = pd.read_csv('moodang_dataset - 시트1.csv')

    print(df_origin[['location']])
    # # 데이터셋 object name yaml 파일에서 불러오기
    # with open('yolov5/customDataset/gachon_road.yaml') as f:
    #     object_name = yaml.load(f, Loader=yaml.FullLoader)
    #     data = list(object_name.values())
    #     df_name = data[5]
    #     df_name.insert(0, 'location')
    # df_origin = pd.read_csv('dataset.csv', names=df_name)


    # Convert categorical features to numeric values using oneHotEncoder
    encoder = OrdinalEncoder()
    df_origin[['location']] = encoder.fit_transform(df_origin[['location']])

    # X, Y 분류
    X = df_origin.drop(columns=['location'])
    y = df_origin[['location']]

    # decision tree model
    model = DecisionTreeRegressor(min_samples_split=4, min_impurity_decrease=0.1, random_state=0)
    model.fit(X, y)

    # visualization
    fig = plt.figure(figsize=(15, 8))
    fig.show(plot_tree(model, feature_names=X.columns))
    print(X.columns)

    # save model train result
    joblib.dump(model, 'decision_model.pkl')