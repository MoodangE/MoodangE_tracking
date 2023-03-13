import collections


def calculate_congestion(datas, frame):
    # 80 percent of summary_frame
    criterion = int(0.8 * frame)

    # Count id values
    count_person = dict(collections.Counter(datas))
    count_standing = 0

    for i in count_person.values():
        if i > criterion:
            count_standing += 1

    if count_standing < 14:
        value = 'Spare'
    elif count_standing < 18:
        value = 'General'
    elif count_standing < 23:
        value = 'Caution'
    else:
        value = 'Congestion'
    value += ', standing person: ' + str(count_standing)

    return value
