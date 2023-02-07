import collections


def calculate_congestion(datas):
    # Count id values
    count_person = collections.Counter(datas)

    # If FPS = 30 and sum_time = 5, n = FPS * 3
    count_standing = len(count_person.most_common(n=5))

    if count_standing < 10:
        result = 'Spare'
    elif count_standing < 18:
        result = 'General'
    elif count_standing < 23:
        result = 'Caution'
    else:
        result = 'Congestion'

    return result
