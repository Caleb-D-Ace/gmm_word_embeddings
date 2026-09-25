from pathlib import Path

"""Stopword handling for preprocessing."""

# Function words only. Includes contraction fragments, since the tokenizer splits "don't" into "don" and "t".
ENGLISH_STOPWORDS = frozenset("""
a about above after again against ain all am an and any are aren as at
be because been before being below between both but by
can couldn could
d did didn do does doesn doing don down during
each
few for from further
had hadn has hasn have haven having he her here hers herself him himself his how
i if in into is isn it its itself
just
ll
m me mightn more most mustn my myself
needn no nor not now
o of off on once only or other our ours ourselves out over own
re
s same shan she should shouldn so some such
t than that the their theirs them themselves then there these they this those through to too
under until up
ve very
was wasn we were weren what when where which while who whom why will with won would wouldn
y you your yours yourself yourselves
""".split())


def load_stopwords(spec: str) -> frozenset:
    """
    'english' -> the built-in list, 'none' -> no stopwords, anything else -> a path to a text file
    with one stopword per line (blank lines and lines starting with '#' are ignored).
    """
    if spec == "english":
        return ENGLISH_STOPWORDS
    if spec == "none":
        return frozenset()

    path = Path(spec)
    if not path.is_file():
        raise FileNotFoundError(f"Stopwords file '{spec}' does not exist. Use 'english', 'none', or a path to a file.")
    lines = (line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines())
    return frozenset(line for line in lines if line and not line.startswith("#"))
