import glob, os
import numpy as np
from models.document import Document


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

        self.bm25_ranking_result = {}



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

    def bm25_ranking(self, using_relevance : bool = True):
        # TODO: complete the BM25 ranking against the query

        # declare the variables
        K1 = 1.2
        K2 = 500
        B = 0.75


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

            # for query_term, query_frequency in self.corresponding_query.parsed_query_text.items():
            for query_term, query_frequency in self.corresponding_query.parsed_long_query.items():

                if query_term in document.terms:
                    fi = document.terms[query_term]
                else:
                    fi = 0

                # TODO:
                """
                Description of the BM25 ranking function
                """

                # ri is the appearance of the term in relevant documents
                if query_term in self.relevant_document_frequencies:
                    ri = self.relevant_document_frequencies[query_term]
                else:
                    ri = 0

                # ni is document frequency of the term
                if query_term  in self.document_frequencies:
                    ni = self.document_frequencies[query_term]
                else:
                    ni = 0


                # calculate each term
                term_1_num = (ri + 0.5)/(R - ri + 0.5)
                term_1_denom = (ni - ri + 0.5)/(N - ni - R + ri + 0.5)

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
                '''

                term_1 = term_1_num/term_1_denom * 2*(N-R+1)/(0.5/(R+0.5))

                term_2 = (K1 + 1) * fi / (K + fi)

                term_3 = (K2 + 1) * query_frequency / (K2 + query_frequency)

                weighting += np.log10(term_1) * term_2 * term_3

            document_weighting[document_id] = weighting

        self.bm25_ranking_result = {k: v for k, v
                                        in sorted(document_weighting.items(),
                                              key= lambda item: item[1], reverse = True)}

    def jm_lm_ranking(self):
        # TODO: complete the Jelinek-Mercer based Language Model
        print()

    def prm_ranking(self):
        # TODO: complete the personal ranking model
        print()


    def calculate_average_precision(self):
        num_relevant_documents = sum(self.relevance_for_folder.values())

        bm25_predicted_relevance = 0
        jm_predicted_relevance = 0
        prm_predicted_relevance = 0
        for document_id, bm_25_rank in self.bm25_ranking_result.items():

            # check if above threshold for bm25
            if bm_25_rank > 0 & self.relevance_for_folder[document_id]:
                bm25_predicted_relevance += 1

            # check if above threshold for jlm
            # if bm_25_rank > 0 & self.relevance_for_folder[document_id]:
            #     predicted_relevance += 1

        bm25_precision = bm25_predicted_relevance/num_relevant_documents

        # create a tuple
        self.average_precision = (bm25_precision)


    def calculate_precision_at_k(self, k = 10):


        bm25_predicted_relevance = 0
        jm_predicted_relevance = 0
        prm_predicted_relevance = 0
        for document_id, bm_25_rank in self.bm25_ranking_result.items()[:k]:

            # check if above threshold for bm25
            if bm_25_rank > 0 & self.relevance_for_folder[document_id]:
                bm25_predicted_relevance += 1

        for document_id, jlm_rank in self.jm_lm_ranking().items()[:k]:

            # check if above threshold for bm25
            if jlm_rank > 0 & self.relevance_for_folder[document_id]:
                jm_predicted_relevance += 1

        bm25_precision_at_k = bm25_predicted_relevance / k
        jlm_precision_at_k = jm_predicted_relevance / k



    # def calculate_discounted_cumulative_gain(self):
    #     bm25_discounted_cumulative_gain = 0
    #     jlm_discounted_cumulative_gain = 0
    #     prm_discounted_cumulative_gain = 0
    #
    #     i = 1
    #     for document_id, bm_25_rank in self.bm25_ranking_result.items()[:10]:
    #         if self.relevance_for_folder[document_id]:
    #
    #         i += 1
