import os
import string
import re
from stemming.porter2 import stem

from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import ttest_rel

# init a global stop word list
with open("../common-english-words.txt") as file:
    STOP_WORDS = list(set(file.read().split(",")))

# dictionary of xml and html tags with what they should be processed as
# online resources used to assist and then testing for last few
# taken in reference to: https://www.freeformatter.com/xml-escape.html
# taken in reference to: https://developer.mozilla.org/en-US/docs/Glossary/Entity
XML_ENTITY_MAP = {
    "<p>": " ",
    "</p>": " ",
    "\n": "",
    "&amp;": "&",
    "&apos;": "'",
    "&quot;": '"',
    "&lt;": "<",
    "&gt;": ">",
}


def parse_xml_content(xml_text_string: str):
    """
    Function to remove all xml tags and entities and convert to raw text.

    Parameters
    ----------
    xml_text_string : str
        A string containing raw xml text.

    Returns
    -------
    str
        The processed string with xml references removed.

    """

    processed_xml_text = xml_text_string.strip()

    for entity, replacement in XML_ENTITY_MAP.items():
        processed_xml_text = processed_xml_text.replace(entity, replacement)

    return processed_xml_text


# create a helper to ensure all tokenization is universal
def tokenize_string_text():
    # TODO: finish this method
    print()


def get_subdirectories(path: str) -> list[str]:
    # return the subdirectories
    return [sub_dir.name for sub_dir in os.scandir(path) if sub_dir.is_dir()]


def parse_query(query: str) -> dict[str, int]:
    """
    A utility function to parse any raw text query.  Will convert all to lower,
    remove whitespace and remove punctuation and digits.

    For the purpose of this function, words are considered any alphanumeric values
    that are seperated by a space.

    The function returns a dictionary of terms and frequencies where terms are
    considered any non-numeric word of length greater than 2 that is not in
    the list of stop words.

    Parameters
    ----------
    query : str
        The raw text that is going to be converted into terms and frequencies

    Returns
    -------
    Dict[str, int]
        Dictionary containing the terms as keys and frequencies as values.

    """

    # begin preprocessing
    processed_query = query.lower()

    # ensure no unnecessary whitespace
    processed_query = processed_query.strip()
    processed_query = re.sub("\s+", " ", processed_query)

    # replace '-' individually to whitespace given some words are composites of 2
    processed_query = processed_query.replace("-", " ")

    processed_query = processed_query.translate(
        str.maketrans("", "", string.punctuation)
    )
    processed_query = processed_query.translate(str.maketrans("", "", string.digits))

    processed_query = re.sub("\s+", " ", processed_query)

    # get list of stemmed terms
    terms = [
        stem(word)
        for word in processed_query.split()
        if len(stem(word)) > 2
           and word not in STOP_WORDS
           and stem(word) not in STOP_WORDS
    ]

    # sort to ensure term_frequency dictionary is sorted already
    terms.sort()

    # initialise the empty dictionary
    term_frequency = {}

    # iterate and add terms
    for term in terms:
        if term in term_frequency:
            term_frequency[term] += 1
        else:
            term_frequency[term] = 1

    # return sorted dictionary
    return {k: v for k, v in sorted(term_frequency.items(), key=lambda item: item[1], reverse=True)}


def create_summary_dataframe(data: list, average_index_name: str,
                             column_names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=column_names).sort_values(by="Topic")
    df.set_index('Topic', inplace=True)
    mean_values = df.mean().to_list()
    mean_row_df = pd.DataFrame([mean_values], index=[average_index_name], columns=df.columns)
    summary_df = pd.concat([df, mean_row_df])
    return summary_df


def compare_df_ttest(df: pd.DataFrame) -> None:
    columns = df.columns
    results = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            column1 = columns[i]
            column2 = columns[j]
            t_stat, p_val = ttest_rel(df[column1][:-1], df[column2][:-1])
            results.append({
                'Comparison': f'{column1} vs {column2}',
                'T-Statistic': t_stat,
                'P-Value': p_val
            })

            print(f"{column1} vs {column2}: t_statistic = {t_stat}, p_value = {p_val}")


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

def plot_line_chart(df, title):
    plt.figure(figsize=(10, 6))
    # Exclude the 'MAP' or 'Average' row
    df_numeric = df.drop(['MAP', 'Average'], errors='ignore')
    for column in df_numeric.columns:
        plt.plot(df_numeric.index, df_numeric[column], label=column)

    plt.title(title)
    plt.xlabel('Collection')
    plt.ylabel('Scores')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.show()

def plot_bar_chart(df, title):
    plt.figure(figsize=(10, 6))
    # Exclude the 'MAP' or 'Average' row
    df_numeric = df.drop(['MAP', 'Average'], errors='ignore')
    for column in df_numeric.columns:
        plt.bar(df_numeric.index, df_numeric[column], label=column)

    plt.title(title)
    plt.xlabel('Collection')
    plt.ylabel('Scores')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.show()


def print_bm25_results(container: dict, result_type : str, text_file_loc: str, title: str):

    documents = []

    for folder_id, document_folder in container.items():
        total_weight = 0

        for document_id, weight in document_folder.bm25_ranking_result.items():
            total_weight += weight
        
        documents.append([folder_id, weight])

    sorted_documents = sorted(documents, key=lambda item: item[1], reverse=True)

    top_15 = sorted_documents[:15]


    with open(text_file_loc, 'a') as file:
        # Write the title
        file.write(title + '\n')

        for top_15_id,_ in top_15:

            for folder_id, document_folder in container.items():

                if folder_id == top_15_id:

                    file.write(f"Query{folder_id} (DocID Weight):\n")


                    ranking_result = getattr(document_folder, result_type, None)

                    if ranking_result is None:
                        print(f"Error getting ranking results.  {result_type} is not an attribute of DocumentFolder class."
                            f"Please try again with the correct attribute.")
                        break
                    else:
                        i = 0
                        for document_id, weight in document_folder.bm25_ranking_result.items():
                            file.write(f"\t{document_id}: {weight}\n")

                            i += 1

                            if i == 15:
                                break

                        file.write('\n')

def print_jm_results(container: dict, result_type : str, text_file_loc: str, title: str):

    documents = []

    for folder_id, document_folder in container.items():
        total_weight = 0

        for document_id, weight in document_folder.jm_ranking_result.items():
            total_weight += weight
        
        documents.append([folder_id, weight])

    sorted_documents = sorted(documents, key=lambda item: item[1], reverse=True)

    top_15 = sorted_documents[:15]


    with open(text_file_loc, 'a') as file:
        # Write the title
        file.write(title + '\n')

        for top_15_id,_ in top_15:

            for folder_id, document_folder in container.items():

                if folder_id == top_15_id:

                    file.write(f"Query{folder_id} (DocID Weight):\n")


                    ranking_result = getattr(document_folder, result_type, None)

                    if ranking_result is None:
                        print(f"Error getting ranking results.  {result_type} is not an attribute of DocumentFolder class."
                            f"Please try again with the correct attribute.")
                        break
                    else:
                        i = 0
                        for document_id, weight in document_folder.jm_ranking_result.items():
                            file.write(f"\t{document_id}: {weight}\n")

                            i += 1

                            if i == 15:
                                break

                        file.write('\n')
def print_prm_results(container: dict, result_type : str, text_file_loc: str, title: str):

    documents = []

    for folder_id, document_folder in container.items():
        total_weight = 0

        for document_id, weight in document_folder.prm_ranking_result.items():
            total_weight += weight
        
        documents.append([folder_id, weight])

    sorted_documents = sorted(documents, key=lambda item: item[1], reverse=True)

    top_15 = sorted_documents[:15]


    with open(text_file_loc, 'a') as file:
        # Write the title
        file.write(title + '\n')

        for top_15_id,_ in top_15:

            for folder_id, document_folder in container.items():

                if folder_id == top_15_id:

                    file.write(f"Query{folder_id} (DocID Weight):\n")


                    ranking_result = getattr(document_folder, result_type, None)

                    if ranking_result is None:
                        print(f"Error getting ranking results.  {result_type} is not an attribute of DocumentFolder class."
                            f"Please try again with the correct attribute.")
                        break
                    else:
                        i = 0
                        for document_id, weight in document_folder.prm_ranking_result.items():
                            file.write(f"\t{document_id}: {weight}\n")

                            i += 1

                            if i == 15:
                                break

                        file.write('\n')
