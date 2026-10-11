"""Sphinx extension for the Jupyter Book: a pybtex style that labels every
entry of the reference list by its BibTeX key.  The chapter notebooks print
citations as the bracketed key, [Choi2021], so the list on the References
page can be searched for exactly what the text shows."""
from pybtex.plugin import register_plugin
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.labels import BaseLabelStyle
from pybtex.style.sorting.author_year_title import SortingStyle


class KeyLabelStyle(BaseLabelStyle):
    def format_labels(self, sorted_entries):
        for entry in sorted_entries:
            yield entry.key


class KeyLabels(UnsrtStyle):
    default_label_style = KeyLabelStyle
    default_sorting_style = SortingStyle


def setup(app):
    register_plugin("pybtex.style.formatting", "keylabels", KeyLabels)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
