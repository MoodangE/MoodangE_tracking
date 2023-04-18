import collections

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


def calculate_congestion(datas, frame, filming_location):
    # 80 percent of summary_frame
    criterion = int(0.8 * frame)

    # Count id values
    count_person = dict(collections.Counter(datas))
    count_standing = 0

    for i in count_person.values():
        if i > criterion:
            count_standing += 1

    if count_standing < 14:
        level = 'Spare'
    elif count_standing < 18:
        level = 'General'
    elif count_standing < 23:
        level = 'Caution'
    else:
        level = 'Congestion'

    # Save predict result to firebase Realtime Database
    data_path = 'dataList/Congestion/' + str(filming_location)
    ref = db.reference(data_path)
    ref.update({'level': level,
                'person': count_standing,
                })

    return level, count_standing
