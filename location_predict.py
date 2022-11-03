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

target = ['Rotary1', 'Rotary2', 'Rotary3', 'Rotary4', 'Rotary5', 'Rotary6', 'Rotary7', 'Rotary8', 'Rotary9', 'Rotary10',
          'Rotary11', 'Rotary12', 'IT_MainGate1', 'IT_MainGate2', 'IT_MainGate3', 'IT_MainGate4', 'IT_MainGate5',
          'IT_MainGate6', 'AI1', 'AI2', 'AI3', 'AI4', 'AI5', 'AI6', 'AI7', 'AI8']
target.sort()

# load model
model_from_joblib = joblib.load('decision_model.pkl')


def location_predict(categories, names):
    df = pd.DataFrame(data=index, columns=column)

    for j in range(len(categories)):
        cat = int(categories[j]) if categories is not None else 0
        label = names[cat]
        df[label] = 1

    model_predict = model_from_joblib.predict(df)
    model_predict = int(np.round(model_predict[0]))
    predict_location = target[model_predict]

    return predict_location
