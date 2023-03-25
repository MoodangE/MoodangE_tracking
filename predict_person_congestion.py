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
        value = '여유'
    elif count_standing < 18:
        value = '보통'
    elif count_standing < 23:
        value = '주의'
    else:
        value = '혼잡'
    # value += ', standing person: ' + str(count_standing)

    # Save predict result to firebase Realtime Database
    data_path = 'dataList/Congestion'
    ref = db.reference(data_path)
    ref.update({filming_location: value})

    return value
