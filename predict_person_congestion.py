import collections


def calculate_congestion(datas):
    counts = collections.Counter(datas)
    counts = counts.sort()
    if datas < 10:
        result = 'Spare'
    elif datas < 18:
        result = 'General'
    elif datas < 23:
        result = 'Caution'
    else:
        result = 'Congestion'
    return result
