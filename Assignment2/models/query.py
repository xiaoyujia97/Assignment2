import re
from utils.helpers import parse_query


class Query:

    def __init__(self, query_text: str):
        self.query_text = query_text

        self.query_num, self.query_title, self.query_desc, self.query_narr = self.parse_query_text()

        self.parsed_query_text = parse_query(self.query_title)

        self.parsed_query_description = parse_query(self.query_desc)

        self.parsed_long_query = parse_query(self.query_title + ' ' + self.query_desc + ' ' + self.query_narr)

    def parse_query_text(self):
        """

        Function to convert the raw query text to a tuple of query number, title,
        description and narrative.

        Parameters
        ----------

        Returns
        -------
        Tuple(int, str, str, str)
            The tuple of each individual property

        """


        query_number = re.search(r'<num> Number: R(.+)\n', self.query_text).group(1)

        query_title = re.search(r'<title>(.+)\n', self.query_text).group(1)

        # unfortunately not all queries have Description (QUERY 123).  Use ternary condition
        query_desc = re.search(r'<desc>(?: Description:)?\s*((?:.|\n)*?)(?=<narr>|$)', self.query_text).group(1)

        query_narrative = re.search(r'<narr> Narrative:((.|\n)*)', self.query_text, re.DOTALL).group(1)

        # remove newlines
        query_title = query_title.replace('\n', ' ').strip()
        query_desc = query_desc.replace('\n', ' ').strip()
        query_narrative = query_narrative.replace('\n', ' ').strip()

        return int(query_number), query_title, query_desc, query_narrative

    def get_query_num(self):
        return self.query_num
