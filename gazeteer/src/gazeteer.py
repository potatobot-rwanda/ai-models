import json

class Gazeteer:

    def __init__(self, gazeteer_file : str, entity_type : str):
        self.gazeteer_file = gazeteer_file
        self.keywords = {}
        self.entities = {}
        self.entity_type = entity_type
        self.load_gazeteer()

    def load_gazeteer(self):
        """
        Load the gazeteer from the specified file.
        The file should contain lines in the format: "key=value".
        """
        self.entities = {o["name"] : o for o in json.load(open(self.gazeteer_file, 'r'))}
        for ne in self.entities.values():
            self.keywords[ne["name"].lower()] = ne["name"]
            for synonym in ne["synonyms"]:
                self.keywords[synonym.lower()] = ne["name"]

    def match(self, text, case_sensitive=False):
        """
        Match the text against the gazeteer and return the matched entities.
        """

        def find_all(haystack, needle):
            """
            Find all occurrences of needle in haystack.
            """
            start = 0
            while True:
                start = haystack.find(needle, start)
                if start == -1:
                    return
                yield start
                start += len(needle)

        matches = []
        if not case_sensitive:
            text = text.lower()

        for keyword, ne in self.keywords.items():
            for start_index in find_all(text, keyword):

                
                end_index = start_index + len(keyword)
                
                # Check if the match already exists to avoid duplicates
                if any(m["start_index"] >= start_index and m["end_index"] <= end_index and m["normalized_value"] == ne for m in matches):
                    continue

                surface_value = text[start_index:end_index]
                true_value = ne
                matches.append({
                    "start_index": start_index,
                    "end_index": end_index,
                    "normalized_value": true_value,
                    "surface_value": surface_value,
                    "annotation_type": self.entity_type
                })
        return matches

class EngPotatoGazeteer(Gazeteer):
    """
    A specific gazeteer for English PotatoBot.
    It loads the gazeteer from a predefined file.
    """
    
    def __init__(self):
        super().__init__('../data/eng_potato_varieties.json', 'POTATO')