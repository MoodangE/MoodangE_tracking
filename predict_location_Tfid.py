import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from predict_location_corpora_Tfid import make_corpora_tfid

# Connect firebase Realtime DB
import json
import firebase_admin
from firebase_admin import credentials, db

# Load the database URL from the config file
with open('serviceDatabaseUrl.json') as f:
    config = json.load(f)
database_url = config['databaseURL']

cred = credentials.Certificate("./serviceAccountKey.json")
firebase_admin.initialize_app(cred, {'databaseURL': database_url})

# Predict location
corpora = make_corpora_tfid()
vector = TfidfVectorizer(max_df=500)

target = ['AI', 'Art', 'Education', 'EduMainLib', 'MainGate', 'MainLib', 'Rotary', 'Student', 'Tunnel']
map_sequence = ['MainGate', 'Tunnel', 'Education', 'EduMainLib', 'Student', 'AI', 'MainLib', 'Rotary', 'Art']
sequence_count = len(map_sequence)

standard_value = 0.5


def location_predict_vector(datas, previous_location, bus_id, bus_power):
    corpora_tmp = corpora.copy()
    corpora_tmp.append(datas)

    transformed_weights = vector.fit_transform(corpora_tmp)
    similarity = cosine_similarity(transformed_weights[-1], transformed_weights[:-1])
    similarity = pd.DataFrame(similarity, index=['Similarity'], columns=target)

    predict = similarity.T.sort_values(by='Similarity', ascending=False).head(3)
    print('\nprevious:', previous_location, '\npredict\n', predict, '\n')

    predict_location = predict.index.tolist()
    predict_percent = sum(predict.values.tolist(), [])
    result = predict_location[0]

    if previous_location != 'None':
        previous_code = map_sequence.index(previous_location)
        predict_code = (previous_code + 1) % sequence_count
        for i, percentage in enumerate(zip(predict_location, predict_percent)):
            # If the current and previous locations are the same
            if percentage[0] == previous_location and standard_value < percentage[1]:
                result = percentage[0]
                break

            percentage_code = map_sequence.index(percentage[0])
            # If the current and previous locations are the different
            if percentage_code == predict_code and standard_value < percentage[1]:
                result = percentage[0]
                break
            elif i == len(predict_location) - 1:
                result = previous_location

        # Save predict result to firebase Realtime Database
        # Classification drive route state
        result_code = map_sequence.index(result)
        if 0 < result_code < 5:
            up_state = True
            down_state = False
        elif 0 == result_code or 5 == result_code:
            up_state = True
            down_state = True
        else:
            up_state = False
            down_state = True

        data_path = 'dataList/Bus/' + str(bus_id)
        ref = db.reference(data_path)
        ref.update({'location': result,
                    'up': up_state,
                    'down': down_state,
                    'power': bus_power,
                    })

    return result
