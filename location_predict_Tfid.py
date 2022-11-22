import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from make_location_model_Tfid import make_model_tfid

# load model
# vector = pickle.load(open("vectorized_model.pkl", "rb"))
# vector_data = pickle.load(open("vectorized_data.pkl", "rb"))

corpora = make_model_tfid()
vector = TfidfVectorizer()

target = ['AI', 'Art', 'Education', 'EduMainLib', 'IT', 'MainGate', 'MainLib', 'Rotary', 'Student', 'Tunnel']


def location_predict_vector(datas):
    corpora_tmp = corpora.copy()
    corpora_tmp.append(datas)
    print(corpora_tmp)
    transformed_weights = vector.fit_transform(corpora_tmp)
    print(transformed_weights)
    # datas_weight = vector.transform(datas)
    # similarity = cosine_similarity(datas_weight, vector_data)
    similarity = cosine_similarity(transformed_weights[-1], transformed_weights[:-1])
    similarity = pd.DataFrame(similarity, index=['Similarity'],
                              columns=target)
    predict = similarity.T.sort_values(by='Similarity', ascending=False).head(1)

    return predict.index.tolist()

# test = ['vehicleBreaker vehicleBreaker vehicleBreaker vehicleBreaker vehicleBreaker vehicleBreaker vehicleBreaker vehicleBreaker vehicleBreaker gachon vehicleBreaker']
# tf = TfidfTransformer()
# load_vec = TfidfVectorizer
# datas_weight = vector.transform(test)
# similarity = cosine_similarity(datas_weight, vector_data)
# similarity = pd.DataFrame(similarity, index=['Similarity'],
#                           columns=target)
# predict = similarity.T.sort_values(by='Similarity', ascending=False).head(1)
# print(predict.index.tolist())
