import re

from utils.helpers import parse_xml_content, parse_query


class Document:

    def __init__(self, document_location: str):
        self.document_location = document_location

        self.document_id, self.document_title, self.document_text = self.parse_document_information()

        self.terms = parse_query(self.document_text)

        self.doc_len = len(self.document_text.split())

    def parse_document_information(self):
        """
        A function to convert an xml document into it's raw characteristics.

        Parameters
        ----------

        Returns
        -------
        document_id: str
            The document id as string.
        title_content: str
            The title of the document.
        parsed_text_content: str
            The text of the document with all xml tags removed.

        """
        # open the document
        with open(self.document_location) as file:
            document_content = file.read()

            # get the document_id using regex pattern
            document_id = re.search(r'itemid="(\d+)"', document_content).group(1)

            # use regular expression to find the title
            title_content = re.search(
                r"<title>(.*?)</title>", document_content
            ).group(1)

            # use regular expressions to find the text content
            text_content = re.search(
                r"<text>(.*?)</text>", document_content, re.DOTALL
            ).group(1)

            # use a helper function to remove xml tags
            parsed_text_content = parse_xml_content(text_content)

            return document_id, title_content, parsed_text_content

    def get_document_id(self):
        return self.document_id

    def get_document_length(self):
        return self.doc_len
