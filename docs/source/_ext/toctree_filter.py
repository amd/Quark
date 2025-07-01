"""Sphinx extension to filter out entries from the toctree directive

Based on https://stackoverflow.com/questions/15001888/conditional-toctree-in-sphinx
"""

import re
from sphinx.directives.other import TocTree


def setup(app):
    app.add_config_value('toctree_filter_exclude', [], 'html')
    app.add_directive('toctree-filter', TocTreeFilter)
    return {'version': '1.0.0'}

class TocTreeFilter(TocTree):
    """Custom TocTree that conditionally omit toctree from sphinx-build

    This directive to notifies Sphinx about the hierarchical structure of the docs,
    and includes a table-of-contents like tree in the current document.
    This version filters the entries based on a list of prefixes.

    It simply filters the content of the directive and call the super's version of run.
    The list of exclusions is stored in the **toctree_filter_exclusion** list.
    Any table of content entry prefixed by one of these strings will be excluded.
    E.g., if `toctree_filter_exclusion=['secret','draft']`, then all toc entries of the
    form `:secret:ultra-api` or `:draft:new-features` will be excuded from
    the final table of contents.
    Entries without a prefix are always included.

    """
    hasPat = re.compile(r'^\s*:(.+):(.+)$')

    # Remove any entries in the content that we dont want and strip
    # out any filter prefixes that we want but obviously don't want the
    # prefix to mess up the file name.
    def filter_entries(self, entries):
        excl = self.state.document.settings.env.config.toctree_filter_exclude
        filtered = []
        for e in entries:
            m = self.hasPat.match(e)
            if m is not None:
                if not m.groups()[0] in excl:
                    filtered.append(m.groups()[1])
            else:
                filtered.append(e)
        return filtered

    def run(self):
        # Remove all TOC entries that should not be on display
        self.content = self.filter_entries(self.content)
        return super().run()
