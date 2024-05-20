import os, re

import pandas as pd

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


        container[documents_folder.get_folder_number()] = documents_folder


