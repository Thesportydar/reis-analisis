import sqlite3
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from statistics import mode
import os
from dotenv import load_dotenv


load_dotenv()
DB_PATH = os.getenv("DATABASE_URL", "sqlite.db")

class QuestionClusterer:

    def __init__(self, num_clusters, embedder_model="hiiamsid/sentence_similarity_spanish_es"):
        self.embedder = SentenceTransformer(embedder_model)
        self.clustering_model = None
        self.num_clusters = num_clusters if num_clusters else None

    @staticmethod
    def optimalK(data, nrefs=3, maxClusters=15):
        """
            Gap Statistic for K means
        """
        gaps = np.zeros((len(range(1, maxClusters)),))
        resultsdf = []

        for gap_index, k in enumerate(range(1, maxClusters)):
            # Holder for reference dispersion results
            refDisps = np.zeros(nrefs)# For n references, generate random sample and perform kmeans getting resulting dispersion of each loop

            for i in range(nrefs):
                # Create new random reference set
                randomReference = np.random.random_sample(size=data.shape)

                # Fit to it
                km = KMeans(k)
                km.fit(randomReference)

                refDisp = km.inertia_
                refDisps[i] = refDisp

                # Fit cluster to original data and create dispersion
                km = KMeans(k)
                km.fit(data)

                origDisp = km.inertia_# Calculate gap statistic
                gap = np.log(np.mean(refDisps)) - np.log(origDisp)# Assign this loop's gap statistic to gaps
                gaps[gap_index] = gap

                resultsdf.append([k,gap])

                df_extended = pd.DataFrame(resultsdf, columns=['clusterCount', 'gap'])

        return (gaps.argmax() + 1, df_extended)


    def fetch_questions(self, table_name, rei_id, period):
        query = f"""
            SELECT question, group_id
            FROM {table_name}
            LEFT JOIN main.class c on c.id = {table_name}.class_id
            WHERE class_id = ? AND recorrido_id = ?;
        """

        with sqlite3.connect(DB_PATH) as con:
            cursor = con.cursor()
            cursor.execute(query, (period, rei_id))
            return cursor.fetchall()


    def preprocess_questions(self, questions):
        if not questions:
            return pd.DataFrame(columns=["pregunta", "grupo", "votos"])

        dfcorpus = pd.DataFrame(questions)
        dfcorpus = dfcorpus.rename(columns={0: "pregunta"})
        dfcorpus = dfcorpus.rename(columns={1: "grupo"})

        dfcorpus["votos"] = 1
        corpus = dfcorpus["pregunta"].tolist()
        corpus_embeddings = self.embedder.encode(corpus)

        # Compute cosine similarities
        similarities = self.embedder.similarity(corpus_embeddings, corpus_embeddings)
        to_remove = []

        for idx_i, sentence1 in enumerate(corpus):
            count_matches = 0
            match_groups = dfcorpus['grupo'].iloc[idx_i]

            for idx_j, sentence2 in enumerate(corpus):
                sim = similarities[idx_i][idx_j]

                if (sim > 0.9) and (idx_i != idx_j):
                    count_matches += 1
                    gxj = dfcorpus['grupo'].iloc[idx_j]
                    match_groups = match_groups + "," + gxj

                    if (idx_j>idx_i):
                        to_remove.append(idx_j)

            dfcorpus.iloc[idx_i, dfcorpus.columns.get_loc('votos')] = count_matches + 1
            dfcorpus.iloc[idx_i, dfcorpus.columns.get_loc('grupo')] = str(match_groups)

        if len(to_remove)>0:
            rm_indexes = np.array(to_remove)
            dfcorpus = dfcorpus.drop(dfcorpus.index[rm_indexes])

        dfcorpus = dfcorpus.reset_index()
        del dfcorpus['index']

        return dfcorpus


    def train_cluster_model(self, corpus_embeddings):

        if not self.num_clusters:
            if len(corpus_embeddings)<15:
                maxC=len(corpus_embeddings)
            else:
                maxC=15

            scores = []
            for i in range(0,5):
                score_g, df = QuestionClusterer.optimalK(corpus_embeddings, maxClusters=maxC)
                scores.append(score_g)

            score_g = mode(scores)
            self.num_clusters = score_g

        self.clustering_model = KMeans(n_clusters=self.num_clusters, random_state=0)
        self.clustering_model.fit(corpus_embeddings)


    def clusterize_dx_questions(self, rei_id, period):
        questions = self.fetch_questions("dx_question", rei_id, period)
        dfcorpus = self.preprocess_questions(questions)

        if dfcorpus.empty:
            # add empty cluster column and return
            dfcorpus['cluster'] = None
            return dfcorpus

        corpus = dfcorpus["pregunta"].tolist()
        corpus_embeddings = self.embedder.encode(corpus)

        self.train_cluster_model(corpus_embeddings)

        cluster_assignment = self.clustering_model.labels_
        cluster_assignment = [x + 1 for x in cluster_assignment]

        clustered_sentences = [[] for i in range(self.num_clusters)]
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[cluster_id - 1].append(corpus[sentence_id])

        dfcorpus['cluster'] = cluster_assignment

        df_info_reis_pregs = dfcorpus

        return df_info_reis_pregs


    def clusterize_new_questions(self, rei_id, period):

        questions = self.fetch_questions("new_question", rei_id, period)
        if not questions:
            return pd.DataFrame(columns=["pregunta", "grupo", "cluster"])

        if not self.clustering_model:
            raise ValueError("The model has not been trained yet. Call clusterize_dx_questions first.")

        df_info_reis_pregnuevas = pd.DataFrame(questions)
        df_info_reis_pregnuevas = df_info_reis_pregnuevas.rename(columns={0: "pregunta"})
        df_info_reis_pregnuevas = df_info_reis_pregnuevas.rename(columns={1: "grupo"})

        nuevas_preg = df_info_reis_pregnuevas['pregunta'].tolist()
        nuevas_preg_grupo = df_info_reis_pregnuevas['grupo'].tolist()

        asign_clusters = []
        for p in nuevas_preg:
            frase_embeddings = self.embedder.encode(p)

            sims = []
            for i in range(0, self.num_clusters):
                sim = self.embedder.similarity(self.clustering_model.cluster_centers_[i], frase_embeddings)
                sims.append(sim)

            val_max = max(sims)
            pos_max = sims.index(max(sims[::-1]))
            asign_clusters.append(pos_max)

        asign_clusters = [x + 1 for x in asign_clusters]
        df_info_reis_pregnuevas['cluster'] = asign_clusters

        return df_info_reis_pregnuevas
