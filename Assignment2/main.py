import os, re

import pandas as pd
from scipy.stats import ttest_rel

from utils.helpers import get_subdirectories
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
    document_relevance_dict = {}

    with open(path, "r") as document_relevance_file:
        document_details = document_relevance_file.readlines()

        for document_detail in document_details:
            document_id = document_detail.split()[1]
            document_relevance = document_detail.split()[2]
            document_relevance = int(document_relevance)
            document_relevance_dict[int(document_id)] = bool(document_relevance)

    return document_relevance_dict


def process_ranking_results(model_results: dict, query: str,
                            model_name: str, output_folder="RankingOutputs"):
    # ensure output directory exists
    if output_folder not in os.listdir():
        os.mkdir(output_folder)

    df = pd.DataFrame(list(model_results.items()), columns=['DocumentID', 'Score'])
    df['Score'] = df['Score'].apply(lambda x: f"{x:.15f}")

    # Write to CSV
    df.to_csv(os.path.join(output_folder, f"{model_name}_{query}Ranking.dat"), sep="\t", index=False, header=True)

    return None


def compare_df_ttest(df: pd.DataFrame) -> None:

    # remove the average
    bm25_vals = df['BM25'][:-1].values
    jm_vals = df['JM_LM'][:-1].values
    prm_vals = df['My_PRM'][:-1].values

    t_stats_1, p_val_1 = ttest_rel(bm25_vals, jm_vals)
    print(f"BM25 vs JM_LM: t_statistic = {t_stats_1}, p_value = {p_val_1}")
    if p_val_1 <= 0.05:
        print("This is considered statistically significant")

    t_stats_2, p_val_2 = ttest_rel(bm25_vals, prm_vals)
    print(f"BM25 vs PRM: t_statistic = {t_stats_2}, p_value = {p_val_2}")
    if p_val_2 <= 0.05:
        print("This is considered statistically significant")

    t_stats_3, p_val_3 = ttest_rel(jm_vals, prm_vals)
    print(f"JM_LM vs PRM: t_statistic = {t_stats_3}, p_value = {p_val_3}")
    if p_val_3 <= 0.05:
        print("This is considered statistically significant")




if __name__ == '__main__':

    queries_location = "../the50Queries.txt"

    # parse the queries text file
    queries_dictionary = parse_queries_document(queries_location)

    relevance_location = "../EvaluationBenchmark"

    relevance = parse_relevance_folder(relevance_location)

    # load the data collections
    folders_location = "../Data_Collection"

    # initialise the dictionary containing all document folders
    container = {}

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

        if documents_folder.get_folder_number() == 101:
            print()

        documents_folder.bm25_ranking()

        process_ranking_results(documents_folder.bm25_ranking_result,
                                documents_folder.get_folder_number(),
                                "BM25")

        # TODO: call Jelinek-Mercer based Language Model

        # TODO: call personal ranking system



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

    average_precision_df = pd.DataFrame(average_precision_list,
                                        columns=["Topic", "BM25", "JM_LM", "My_PRM"]).sort_values(by="Topic")
    average_precision_df.set_index('Topic', inplace=True)
    mean_average_precision = average_precision_df.mean().to_list()
    mean_row_df = pd.DataFrame([mean_average_precision], index=['MAP'],
                               columns=average_precision_df.columns)
    average_precision_df = pd.concat([average_precision_df, mean_row_df])

    print(average_precision_df)
    print("Test stats of average precision: \n")
    compare_df_ttest(average_precision_df)


    precision_at_10_df = pd.DataFrame(precision_at_10_list,
                                        columns=["Topic", "BM25", "JM_LM", "My_PRM"]).sort_values(by="Topic")
    precision_at_10_df.set_index('Topic', inplace=True)
    average_precision_at_10 = precision_at_10_df.mean().to_list()
    average_precision_at_10_df = pd.DataFrame([average_precision_at_10], index = ['Average'],
                                              columns = precision_at_10_df.columns)
    precision_at_10_df = pd.concat([precision_at_10_df, average_precision_at_10_df])


    print(precision_at_10_df)
    print("Test stats of precision @ 10: \n")
    compare_df_ttest(precision_at_10_df)

    discounted_cumulative_gain_df = pd.DataFrame(discounted_cumulative_gain_list,
                                        columns=["Topic", "BM25", "JM_LM", "My_PRM"]).sort_values(by="Topic")
    discounted_cumulative_gain_df.set_index('Topic', inplace=True)
    average_discounted = discounted_cumulative_gain_df.mean().to_list()
    average_discounted_df = pd.DataFrame([average_discounted], index = ['Average'],
                                        columns = discounted_cumulative_gain_df.columns)
    discounted_cumulative_gain_df = pd.concat([discounted_cumulative_gain_df, average_discounted_df])

    print(discounted_cumulative_gain_df)
    print("Test stats of discounted cumulative gain: \n")
    compare_df_ttest(discounted_cumulative_gain_df)
