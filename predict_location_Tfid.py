import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from predict_location_corpora_Tfid import make_corpora_tfid

corpora = make_corpora_tfid()
vector = TfidfVectorizer(max_df=500)

target = ['AI', 'Art', 'Education', 'EduMainLib', 'MainGate', 'MainLib', 'Rotary', 'Student', 'Tunnel']
map_sequence = ['MainGate', 'Tunnel', 'Education', 'EduMainLib', 'Student', 'AI', 'MainLib', 'Rotary', 'Art']
sequence_count = len(map_sequence)


def location_predict_vector(datas, previous_location):
    corpora_tmp = corpora.copy()
    corpora_tmp.append(datas)

    transformed_weights = vector.fit_transform(corpora_tmp)
    similarity = cosine_similarity(transformed_weights[-1], transformed_weights[:-1])
    similarity = pd.DataFrame(similarity, index=['Similarity'], columns=target)

    predict = similarity.T.sort_values(by='Similarity', ascending=False).head(3)
    print('\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tprevious:', previous_location, '\t predict:', predict)
    predict = predict.index.tolist()
    result = predict[0]

    if previous_location != 'None':
        previous_code = map_sequence.index(previous_location)
        predict_code = (previous_code + 1) % sequence_count
        for i, percentage in enumerate(predict):
            if percentage == previous_location:
                result = percentage
                break
            percentage_code = map_sequence.index(percentage)
            if percentage_code == predict_code:
                result = percentage
                break
            elif i == len(predict) - 1:
                result = previous_location

    return result
