from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import scipy as sp
import sklearn as sk
import matplotlib.pyplot as plt
import requests
import urllib.request
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA



def get_seasons():

    link = requests.get("https://www.premierleague.com/history/season-reviews")
    soup = BeautifulSoup(link.text, "html.parser")

    seasons = soup.find_all("a", {"class" : "btn-history"})
    season_links = []
    season_names = []
    season_descripts = []
    for any in seasons:
        season_links.append("https://www.premierleague.com" + str(any.get('href')) + "/report")
    for links in season_links:
        seasonlink = requests.get(links)
        seasonsoup = BeautifulSoup(seasonlink.text, "html.parser")
        season_names.append(seasonsoup.find("h2").text)
        paras2 = seasonsoup.find("p")
        inner_para2 = paras2.find_all("p")
        interim = ""
        for this in inner_para2:
            interim = interim + str(this.text)
        season_descripts.append(interim)


    season_name_frame = pd.DataFrame(season_names, columns = ["Season"])
    season_desc_frame = pd.DataFrame(season_descripts, columns = ["Review"])
    season_frame = pd.concat([season_name_frame, season_desc_frame], axis = 1)


    return season_frame


get_seasons()


def use_seasons():
    season_frame = get_seasons()
    reviews = season_frame['Review'].tolist()

    vectorizer = TfidfVectorizer()
    tf_idf = vectorizer.fit_transform(reviews)
    print (vectorizer.get_feature_names())
    HClusters = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
    HClusters.fit(tf_idf.todense())
    cluster_frame = pd.DataFrame(HClusters.labels_, columns = ["Clusters"])
    cluster_frame = pd.concat([cluster_frame, season_frame['Season']], axis = 1)
    print (cluster_frame)


    pca = PCA(n_components = 2)
    pca_cont = pca.fit_transform(tf_idf.todense())
    pca_cont_DF = pd.DataFrame(pca_cont)
    pca_cont_DF.columns = ["PC1", "PC2"]

    plt.scatter(pca_cont_DF['PC1'], pca_cont_DF['PC2'], c = HClusters.labels_)



use_seasons()

x = ["one", "two", "three"]
y = "twom"

if y in x:
    print ("yeah")
else:
    print ("nah")
