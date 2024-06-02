import glob
import os
import numpy as np
from models.document import Document
from collections import defaultdict


class DocumentsFolder:

    def __init__(self, folder_location: str, folder_name: str,
                 queries_dictionary: dict,
                 relevance_dictionary: dict):

        self.folder_location = folder_location

        self.folder_number = int(folder_name.replace('Data_C', ''))

        self.documents = self.load_documents_in_location()

        self.relevance_for_folder = relevance_dictionary[self.folder_number]

        self.avg_document_length = self.calculate_average_length()

        self.document_frequencies, self.relevant_document_frequencies = self.calculate_document_frequencies()

        self.corresponding_query = queries_dictionary[self.folder_number]

        # ranking results
        self.bm25_ranking_result = {}

        self.jm_ranking_result = {}

        self.prm_ranking_result = {}

    def load_documents_in_location(self):
        documents_dictionary = {}

        for file_loc in glob.glob(os.path.join(self.folder_location, "*.xml")):
            document = Document(file_loc)

            documents_dictionary[int(document.get_document_id())] = document

        return documents_dictionary

    def get_folder_number(self):
        return self.folder_number

    def calculate_average_length(self):

        # get the length of all documents
        sum_document_lengths = sum(doc.get_document_length()
                                   for doc in self.documents.values())

        # calculate the average
        avg_doc_len = sum_document_lengths / len(self.documents)

        return avg_doc_len

    def calculate_document_frequencies(self):

        document_frequencies = {}
        relevant_document_frequencies = {}

        for document_id, doc in self.documents.items():
            for term in doc.terms.keys():

                if term in document_frequencies:
                    document_frequencies[term] += 1
                    relevant_document_frequencies[term] += 1 * self.relevance_for_folder[document_id]
                else:
                    document_frequencies[term] = 1
                    relevant_document_frequencies[term] = 1 * self.relevance_for_folder[document_id]

        return document_frequencies, relevant_document_frequencies

    def bm25_ranking(self, using_relevance: bool = True,
                     K1: float = 1.2, K2: float = 500.0,
                     B: float = 0.75):

        # initialise the dictionary
        document_weighting = {}

        for document_id, document in self.documents.items():

            # initialise the weighting and the value of K
            weighting = 0
            K = K1 * ((1 - B) + B * document.get_document_length() / self.avg_document_length)
            # R is the number of relevant documents
            R = sum(self.relevance_for_folder.values())
            # N is the number of documents in the folder
            N = len(self.documents)

            """
            The BM25 function is comprised of several key components.
            
            The first (inverse document frequency) comprises term 1.  It is designed to measure the amount
            of useful information that each term provides.
            
            There is also term frequency components measuring the frequency of terms in both the query &
            document.  The document frequency has length normalisation K that ensures longer documents do not
            benefit from a larger score due to having more words.
            """

            # for query_term, query_frequency in self.corresponding_query.parsed_query_text.items():
            for query_term, query_frequency in self.corresponding_query.parsed_long_query.items():
                if query_term in document.terms:
                    fi = document.terms[query_term]
                else:
                    fi = 0

                # ri is the appearance of the term in relevant documents
                if query_term in self.relevant_document_frequencies:
                    ri = self.relevant_document_frequencies[query_term]
                else:
                    ri = 0

                # ni is document frequency of the term
                if query_term in self.document_frequencies:
                    ni = self.document_frequencies[query_term]
                else:
                    ni = 0

                if using_relevance:
                    term_1_numerator = (ri + 0.5) / (R - ri + 0.5)
                    term_1_denominator = (ni - ri + 0.5) / (N - ni - R + ri + 0.5)
                else:
                    term_1_numerator = 1
                    term_1_denominator = (ni - 0 + 0.5) / (N - ni - 0 + 0.5)

                '''
                We are seeking to make sure the fraction is greater than 1 in order to guarantee
                a positive value after taking the log.
                
                We can do this by considering the smallest possible value the fraction can take
                and then multiplying by a constant to ensure it is greater than 1.
                
                This is simply the lower bound of the numerator and upper bound of the denominator.
                
                The numerator lower bound is found when r_i is minimized (r_i=0). Therefore the
                lower bound is 0.5/(R+0.5)
                
                The denominator upper bound is found when n_i is maximized but r_i is minimized.
                Intuitively this is n_i = N and r_i = 0.  But n_i could not equal N when r_i is 0
                as this would be contradictory as some relevant documents must have i appear.
                Therefore n_i = N-R.  The upper bound is then (N-R+0.5)/0.5 = 2(N-R)+1.
                
                With these bounds we can guarantee a minimum value for term 1 of 1 by multiply by a constant
                that is the inversion of these 2.
                
                Therefore the constant guaranteeing positivity is:
                
                2*(N-R+1)/(0.5/(R+0.5))
                
                Alternatively if no relevance is used then we simply need to ensure the denominator is less
                than 1 or that the numerator (1) is multiplied by a constant larger than the max value the
                denominator can take.  The greatest value is when ni = N, therefore the upper bound is:
                2N+1.  Therefore if we multiply the numerator by a constant 2N+1, the smallest value that 
                term 1 can take is 1.
                '''

                if using_relevance:
                    term_1 = term_1_numerator / term_1_denominator * 2 * (N - R + 0.5) / (0.5 / (R + 0.5))
                else:
                    term_1 = term_1_numerator / term_1_denominator * 2 * (N + 1)

                term_2 = (K1 + 1) * fi / (K + fi)

                term_3 = (K2 + 1) * query_frequency / (K2 + query_frequency)

                # put it all together and add to the current weighting
                weighting += np.log10(term_1) * term_2 * term_3

            document_weighting[document_id] = weighting

        ranked_results = {k: v for k, v in sorted(document_weighting.items(),
                                                  key=lambda item: item[1],
                                                  reverse=True)}

        return ranked_results

    def jm_lm_ranking(self):
        """
        P(Rx | D ) is the probability of document D relevant to long query Rx
        We use the log-sum version of the formula to compute the scores instead of multiplication.
        This helps avoid the issue of scores becoming very close to 0 when multiplying small probabilities


        The formula consists of two parts that are combined using log summation:
        The first part is (1 - λ) * fqi, D / |D|
        - fqi, D is the frequency of term qi in document D
        - ∣D∣ is the length of the document

        The second part is λ * Cqi / |data_Cx
        - Cqi is the frequency of term qi across the entire collection of documents
        - |Data_Cx| is the total terms in a collection

        The two parts are combined using log summation:
        log10(first_part + second_part + 1)

        The addition of 1 is done after summing the two parts to avoid a situation where a query term
        is not present in a document or the collection, which would result in a zero probability
        Adding 1 avoids taking the log of 0

        λ = 0.4
        """
        # Lambda
        alpha = 0.4

        # initialise dictionaries to store total terms and Cqi per collection
        collection_length = {}
        Cqi_per_collection = {}

        # calculate C
        for document in self.documents.values():
            collection_id = os.path.basename(os.path.dirname(document.document_location))
            if collection_id not in collection_length:
                collection_length[collection_id] = 0
                Cqi_per_collection[collection_id] = {query_term: 0 for query_term in
                                                     self.corresponding_query.parsed_long_query}
            collection_length[collection_id] += document.get_document_length()

            # calculate Cqi
            for term, freq in document.terms.items():
                if term in Cqi_per_collection[collection_id]:
                    Cqi_per_collection[collection_id][term] += freq

        # initialise dictionary to store the scores
        document_weighting = {}

        # score calculation for each document
        for document_id, document in self.documents.items():
            document_score = 0
            collection_id = os.path.basename(os.path.dirname(document.document_location))

            # C and cqi
            C = collection_length[collection_id]
            Cqi = Cqi_per_collection[collection_id]

            # calculate fqi
            for query_term in self.corresponding_query.parsed_long_query:
                fqi = document.terms.get(query_term, 0)
                term_frequency_in_collection = Cqi.get(query_term, 0)

                # combine the first part and the second part
                first_part = (1 - alpha) * (fqi / document.get_document_length())
                second_part = alpha * (term_frequency_in_collection / C)
                document_score += np.log10(first_part + second_part + 1)

            document_weighting[document_id] = document_score

        self.jm_ranking_result = {k: v for k, v in
                                  sorted(document_weighting.items(), key=lambda item: item[1], reverse=True)}
        
    
    def prm_ranking(self, K1: float = 1.2, K2: float = 500, B: float = 0.75, top_k_ratio: float = 0.1, size: float = 150, w5 = 0.5):
        
        # Pseudo-Feedback Algorithm

        # original_query = self.corresponding_query.parsed_query_text

        # Initialize the dictionary for document weights
        document_weighting = self.bm25_ranking(using_relevance=False, K1=K1, K2=K2, B=B)
        
        # 2. Select some number of the top-ranked documents to be the set C (top k documents)

        sorted_document_weighting = sorted(document_weighting.items(), key=lambda x: x[1], reverse=True)

        #select top k of documents based on top_k_ratio
        total_document = len(sorted_document_weighting)
        n_top_docs = int(total_document * top_k_ratio)
        if n_top_docs < 1:
            n_top_docs = 1

        top_k_documents = [doc_id for doc_id, _ in sorted_document_weighting[:n_top_docs]]


        # 3. Calculate the relevance model probabilities P(w|R) using the estimate for P(w,q1...qn) (dirichlet smoothing)
        term_relevance_probability = defaultdict(float)
        collection_term_count = defaultdict(float)
        total_terms_in_C = 0

        total_collection_terms = 0
        for doc_val in self.documents.values():
            total_collection_terms += sum(doc_val.terms.values())

        for document_id in top_k_documents:
            document = self.documents[document_id]
            total_terms_in_C += sum(document.terms.values())
            for term, freq in document.terms.items():
                term_relevance_probability[term] += freq
                collection_term_count[term] += freq

        for term in term_relevance_probability:
            term_relevance_probability[term] /= total_terms_in_C


        # 4. Rank documents again using the KL-divergence score: sum( P(w|R) log P(w|D) )
        for document_id, document in self.documents.items():
            kl_divergence = 0
            doc_length = document.get_document_length()
            for term, p_w_R in term_relevance_probability.items():
                p_w_C = collection_term_count[term] / total_collection_terms
                p_w_D = (document.terms.get(term, 0) + size * p_w_C) / (doc_length + size)
                if p_w_D > 0:
                    kl_divergence += p_w_R * np.log(p_w_R / p_w_D)

            document_weighting[document_id] = -kl_divergence

        # combine new prm score with the previous bm25 score
        combined_weighting = {}
        for doc_id in document_weighting:
            #compared_weighting[doc_id] = max(document_weighting[doc_id], self.bm25_ranking_result[doc_id])
            combined_weighting[doc_id] = w5 * document_weighting[doc_id] + (1 - w5) * self.bm25_ranking_result[doc_id]
            

        self.prm_ranking_result = {k: v for k, v
                                   in sorted(combined_weighting.items(),
                                             key=lambda item: item[1], reverse=True)}
        
    



    def prm_ranking_2(self, K1: float = 1.2, K2: float = 500, B: float = 0.75, n_top_docs: int = 3):

        # Initialize the dictionary for document weights
        document_weighting = self.bm25_ranking(using_relevance=False, K1=K1,
                                               K2=K2, B=B)

        # 2. Select some number of the top-ranked documents to be the set C (top n documents)
        sorted_document_weighting = sorted(document_weighting.items(), key=lambda x: x[1], reverse=True)
        # top_k_documents = [doc_id for doc_id, _ in sorted_document_weighting[:n_top_docs]]

        percentile_95 = np.percentile([sorted_document_weighting[i][1] for i in range(len(sorted_document_weighting))],
                                      95)

        # mean_weight = np.mean([sorted_document_weighting[i][1] for i in range(len(sorted_document_weighting))])
        # sd_weight = np.std([sorted_document_weighting[i][1] for i in range(len(sorted_document_weighting))])

        top_k_documents = [doc_id for doc_id, weight in sorted_document_weighting if weight >= percentile_95]


        # extract the important features using TF*IDF
        features = defaultdict(float)
        total_relevant_documents = len(self.documents)

        for doc_id in top_k_documents:
            document = self.documents[doc_id]
            doc_length = document.get_document_length()

            for term, freq in document.terms.items():
                tf_component = freq / doc_length
                df = self.document_frequencies[term]
                idf_component = np.log(total_relevant_documents/df) # no worry about div error as term has to exist
                tf_idf = tf_component * idf_component
                features[term] += tf_idf

        sorted_feature_weighting = sorted(features.items(), key=lambda x: x[1], reverse=True)

        # percentile_features = np.percentile([sorted_feature_weighting[i][1] for i in range(len(sorted_feature_weighting))],
        #                               95)
        mean_feature_weight = np.mean([sorted_feature_weighting[i][1] for i in range(len(sorted_feature_weighting))])
        sd_feature_weight = np.std([sorted_feature_weighting[i][1] for i in range(len(sorted_feature_weighting))])

        top_k_feature = [term for term, tf_idf_weight in sorted_feature_weighting if tf_idf_weight >= mean_feature_weight + sd_feature_weight]

        print(top_k_feature)


    def calculate_average_precision(self):

        # Total number of relevant documents in the folder
        num_relevant_documents = sum(self.relevance_for_folder.values())

        # Calculate BM25 average precision
        bm25_running_relevant_docs = 0
        current_iteration = 1
        bm25_rolling_average_precision = 0
        for document_id in self.bm25_ranking_result.keys():
            if self.relevance_for_folder[document_id]:  # Check if the document is relevant
                bm25_running_relevant_docs += 1
                bm25_current_precision = bm25_running_relevant_docs / current_iteration
                bm25_rolling_average_precision += bm25_current_precision
            current_iteration += 1

        bm25_average_precision = bm25_rolling_average_precision / num_relevant_documents

        # Calculate JM average precision
        jm_running_relevant_docs = 0
        current_iteration = 1
        jm_rolling_average_precision = 0
        for document_id in self.jm_ranking_result.keys():
            if self.relevance_for_folder[document_id]:  # Check if the document is relevant
                jm_running_relevant_docs += 1
                jm_current_precision = jm_running_relevant_docs / current_iteration
                jm_rolling_average_precision += jm_current_precision  # Accumulate the precision
            current_iteration += 1

        jm_average_precision = jm_rolling_average_precision / num_relevant_documents

        # Calculate PRM average precision (note: this loop was using jm_ranking_result, which may be incorrect)
        prm_running_relevant_docs = 0
        current_iteration = 1
        prm_rolling_average_precision = 0
        for document_id in self.prm_ranking_result.keys():  # Assuming it should be prm_ranking_result
            if self.relevance_for_folder[document_id]:  # Check if the document is relevant
                prm_running_relevant_docs += 1
                prm_current_precision = prm_running_relevant_docs / current_iteration
                prm_rolling_average_precision += prm_current_precision  # Accumulate the precision
            current_iteration += 1

        prm_average_precision = prm_rolling_average_precision / num_relevant_documents

        # Create a tuple
        self.folder_average_precision = (self.folder_number,
                                         bm25_average_precision,
                                         jm_average_precision,
                                         prm_average_precision)

    def calculate_precision_at_k(self, k=10):

        bm25_predicted_relevance = 0
        jm_predicted_relevance = 0
        prm_predicted_relevance = 0

        for document_id in list(self.bm25_ranking_result.keys())[:k]:

            # check if above threshold for bm25
            if self.relevance_for_folder[document_id]:
                bm25_predicted_relevance += 1

        for document_id in list(self.jm_ranking_result.keys())[:k]:

            # check if above threshold for bm25
            if self.relevance_for_folder[document_id]:
                jm_predicted_relevance += 1

        for document_id in list(self.prm_ranking_result.keys())[:k]:

            # check if above threshold for bm25
            if self.relevance_for_folder[document_id]:
                prm_predicted_relevance += 1

        bm25_precision_at_k = bm25_predicted_relevance / k
        jlm_precision_at_k = jm_predicted_relevance / k
        prm_precision_at_k = prm_predicted_relevance / k

        self.folder_precision_at_k = (self.folder_number,
                                      bm25_precision_at_k,
                                      jlm_precision_at_k,
                                      prm_precision_at_k)

    def calculate_discounted_cumulative_gain(self):
        bm25_discounted_cumulative_gain = 0
        jm_discounted_cumulative_gain = 0
        prm_discounted_cumulative_gain = 0

        i = 1
        for document_id in list(self.bm25_ranking_result.keys())[:10]:
            if self.relevance_for_folder[document_id]:
                if i == 1:
                    bm25_discounted_cumulative_gain += 1
                else:
                    bm25_discounted_cumulative_gain += 1 / np.log(i)

            # increment
            i += 1

        i = 1
        for document_id in list(self.jm_ranking_result.keys())[:10]:
            if self.relevance_for_folder[document_id]:
                if i == 1:
                    jm_discounted_cumulative_gain += 1
                else:
                    jm_discounted_cumulative_gain += 1 / np.log(i)

            # increment
            i += 1

        i = 1
        for document_id in list(self.prm_ranking_result.keys())[:10]:
            if self.relevance_for_folder[document_id]:
                if i == 1:
                    prm_discounted_cumulative_gain += 1
                else:
                    prm_discounted_cumulative_gain += 1 / np.log(i)

            # increment
            i += 1

        self.folder_discounted_cumulative_gain = (self.folder_number,
                                                  bm25_discounted_cumulative_gain,
                                                  jm_discounted_cumulative_gain,
                                                  prm_discounted_cumulative_gain)
