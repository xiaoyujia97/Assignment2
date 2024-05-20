import os, string, re
from stemming.porter2 import stem

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
