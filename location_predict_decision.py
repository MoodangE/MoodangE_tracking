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
          'IT_MainGate6', 'AI1', 'AI2', 'AI3', 'AI4', 'AI5', 'AI6', 'AI7', 'AI8', 'MainGate1', 'MainGate2', 'MainGate3',
          'MainGate4', 'MainGate5', 'Tunnel1', 'Tunnel2', 'Tunnel3', 'Tunnel4', 'Tunnel5', 'Tunnel6', 'Tunnel7',
          'Tunnel8', 'Tunnel9', 'Education1', 'Education2', 'Education3', 'Education4', 'Education5', 'Education6',
          'Education7', 'Education8', 'student1', 'student2', 'student3', 'student4', 'student5', 'student6',
          'student7', 'art1', 'art2', 'art3', 'art4', 'art5', 'art6', 'art7', 'art8', 'art9', 'art10', 'MainLib1',
          'MainLib2', 'MainLib3', 'MainLib4', 'MainLib5', 'MainLib6']
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
