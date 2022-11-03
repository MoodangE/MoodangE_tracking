import numpy as np
import joblib
import pandas as pd

column = ['20kmSign', 'arrow', 'bicycleSign', 'buildingSign', 'bump', 'bumpSign',
          'campusMap', 'campusSign', 'crossSign', 'crosswalk', 'diamond',
          'emart24', 'gachon', 'handicap', 'infinity', 'lifelongPlanCard',
          'mirror', 'moodangSign', 'moodangStation', 'noBicycleSign', 'noUturn',
          'rotation', 'rotationRed', 'slowSign', 'snowRemoval', 'straight',
          'straightLeft', 'straightRight', 'streetlamp_M', 'streetlamp_T',
          'streetlamp_hat', 'streetlamp_r', 'turnLeft', 'turnLeftRight',
          'turnRight', 'vehicleBreaker', 'yield']

index = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

target = ['로터리1', '로터리2', '로터리3', '로터리4', '로터리5', '로터리6', '로터리7', '로터리8', '로터리9', '로터리10', '로터리11', '로터리12', 'IT대학_정문1',
          'IT대학_정문2', 'IT대학_정문3', 'IT대학_정문4', 'IT대학_정문5', 'IT대학_정문6', 'AI관정류장1', 'AI관정류장2', 'AI관정류장3', 'AI관정류장4',
          'AI관정류장5', 'AI관정류장6', 'AI관정류장7', 'AI관정류장8']


def location_predict(categories, names):
    df = pd.DataFrame(data=index, columns=column)

    for j in range(len(categories)):
        ca = names[int(j)]
        df[ca] = 1

    model_from_joblib = joblib.load('decision_model.pkl')
    model_predict = model_from_joblib.predict(df)

    predict_location = target[int(model_predict[0])]
    return predict_location
