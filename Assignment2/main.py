import os
import re

from utils.helpers import get_subdirectories, create_summary_dataframe, compare_df_ttest, \
    process_ranking_results, plot_line_chart, plot_bar_chart, print_bm25_results, print_jm_results, print_prm_results
from models.documents_folder import DocumentsFolder
from models.query import Query


def parse_queries_document(path: str) -> dict[int, Query]:
    """
    Parse queries document and return as dictionary.

    Parameters
    ----------
    path : str
        A string containing the path to the queries document.

    Returns
    -------
    dict[int, Query]
        A dictionary containing each query processed into an object of type Query.
        Access via the query number.

    """

    queries_dictionary = {}

    with open(path, "r") as queries_file:
        queries_text = queries_file.read()

        queries = re.findall(r'<Query>(.*?)</Query>', queries_text, re.DOTALL)

        for query_text in queries:
            query = Query(query_text)

            queries_dictionary[query.get_query_num()] = query

    return queries_dictionary


def parse_relevance_folder(folder_loc: str) -> dict[int, dict[int, bool]]:
    """
    Function to parse each relevance documents

    Parameters
    ----------
    folder_loc : str
        The location of the relevance documents folder.

    Returns
    -------
    dict[int, dict[int, bool]]
        A dictionary with key representing the folder number and value being another dictionary
        of RCV1 ID & boolean relevance.

    """

    # initialise the relevance dictionary
    relevance_for_each_folder = {}

    for file_name in os.listdir(folder_loc):
        if file_name.endswith(".txt"):
            # get the corresponding folder id
            folder_id = file_name.replace('Dataset', '').replace('.txt', '')

            # create path to file
            relevance_file_loc = os.path.join(folder_loc, file_name)

            # generate dictionary of relevance for docs in folder
            document_relevances = parse_relevance_file(relevance_file_loc)
            relevance_for_each_folder[int(folder_id)] = document_relevances

    return relevance_for_each_folder


def parse_relevance_file(path: str) -> dict[int, bool]:
    """
    A function to convert a relevance text file into a dictionary

    Parameters
    ----------
    path : str
        The location of the relevance text file.

    Returns
    -------
    dict[int, bool]
        A dictionary of the relevance text file with key representing the RCV1 id
        and boolean value for relevance
    """

    # initialise
    document_relevance_dict = {}

    with open(path, "r") as document_relevance_file:
        document_details = document_relevance_file.readlines()

        for document_detail in document_details:
            document_id = document_detail.split()[1]
            document_relevance = document_detail.split()[2]
            document_relevance = int(document_relevance)
            document_relevance_dict[int(document_id)] = bool(document_relevance)

    return document_relevance_dict


if __name__ == '__main__':

    # initialise the document locations
    queries_location = "../the50Queries.txt"

    # parse the queries text file
    queries_dictionary = parse_queries_document(queries_location)

    # initialise the relevance locations
    relevance_location = "../EvaluationBenchmark"

    # parse the relevances
    relevance = parse_relevance_folder(relevance_location)

    # load the data collections
    folders_location = "../Data_Collection"

    # initialise the dictionary containing all document folders
    container = {}

    # initialise the list that the dataframes will be converted to.
    average_precision_list = []
    precision_at_10_list = []
    discounted_cumulative_gain_list = []

    # iterate over folders
    for folder in get_subdirectories(folders_location):
        folder_path = os.path.join(folders_location, folder)

        # load the documents_folder_class
        documents_folder = DocumentsFolder(folder_path, folder,
                                           queries_dictionary,
                                           relevance)

        # call BM25
        documents_folder.bm25_ranking_result = documents_folder.bm25_ranking()
        process_ranking_results(documents_folder.bm25_ranking_result,
                                documents_folder.get_folder_number(),
                                "BM25")

        # call Jelinek-Mercer based Language Model
        documents_folder.jm_lm_ranking()
        process_ranking_results(documents_folder.jm_ranking_result,
                                documents_folder.get_folder_number(),
                                "JM_LM")

        # call personal ranking system
        documents_folder.prm_ranking()
        process_ranking_results(documents_folder.prm_ranking_result,
                                documents_folder.get_folder_number(),
                                "My_PRM")

        # calculate the results
        # Call average precision
        documents_folder.calculate_average_precision()

        # call precision @10
        documents_folder.calculate_precision_at_k()

        # call discounted cumulative gain
        documents_folder.calculate_discounted_cumulative_gain()

        average_precision_list.append(documents_folder.folder_average_precision)
        precision_at_10_list.append(documents_folder.folder_precision_at_k)
        discounted_cumulative_gain_list.append(documents_folder.folder_discounted_cumulative_gain)

        container[documents_folder.get_folder_number()] = documents_folder

    container = {k: v for k, v in sorted(container.items(), key=lambda x: x[0])}

    # create the dataframes and summary statistics
    average_precision_df = create_summary_dataframe(average_precision_list, 'MAP',
                                                    ["Topic", "BM25", "JM_LM", "My_PRM"])

    print(average_precision_df)
    print("Test stats of average precision: \n")
    compare_df_ttest(average_precision_df)

    precision_at_10_df = create_summary_dataframe(precision_at_10_list, 'Average',
                                                  ["Topic", "BM25", "JM_LM", "My_PRM"])

    print(precision_at_10_df)
    print("Test stats of precision @ 10: \n")
    compare_df_ttest(precision_at_10_df)

    discounted_cumulative_gain_df = create_summary_dataframe(discounted_cumulative_gain_list, 'Average',
                                                             ["Topic", "BM25", "JM_LM", "My_PRM"])

    print(discounted_cumulative_gain_df)
    print("Test stats of discounted cumulative gain: \n")
    compare_df_ttest(discounted_cumulative_gain_df)

    # Plots
    plot_bar_chart(average_precision_df, 'Average Precision')
    plot_bar_chart(precision_at_10_df, 'Precision @ 10')
    plot_bar_chart(discounted_cumulative_gain_df, 'Discounted Cumulative Gain')

    print_bm25_results(container, 'bm25_ranking_result', './BM25_Appendix.txt', 'Appendix for BM25 Model')

    print_jm_results(container, 'jm_ranking_result', './JM_Appendix.txt', 'Appendix for JM Model')

    print_prm_results(container, 'prm_ranking_result', './PRM_Appendix.txt', 'Appendix for PRM Model')


