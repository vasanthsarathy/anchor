import ast
import re

def get_span_indices(document, span):
    print(f"\nSpan: {span}")
    print(f"Document: {document}")
    res = re.search(span, document, re.IGNORECASE)
    print(f"Res: {res}")
    print(f"Find: {document.find(span)}")
    if res:
        return res.span()

def find_substring_indices(string, substring):
    substring = remove_trailing_periods(substring)
    start_index = string.lower().find(substring.lower())
    if start_index == -1:
        return None
    end_index = start_index + len(substring) - 1
    return (start_index, end_index)

def remove_trailing_periods(string):
    while string.endswith('.'):
        string = string[:-1]
    return string
