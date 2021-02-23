import nlp_analysis as nlp
from sklearn.cluster import KMeans

import pandas as pd
import joblib


###############################################
# k-mean clustering
###############################################

num_clusters = 5

model = KMeans(num_clusters, random_state=123)

model.fit(nlp.tfidf_matrix )


#k_rng = range(2,8)
#em.silhouette_score_(k_rng, tfidf_matrix)
#em.sse_scaler_(k_rng, tfidf_matrix)


joblib.dump(model,  'doc_cluster.pkl')



