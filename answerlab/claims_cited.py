"""Step 7: split cited prose, retaining the original quotation-line format."""

import re

from .records import Claim

_TAG = re.compile(r"\[([A-Za-z0-9_-]{1,60})\]")
_TAGS = re.compile(r"\[[A-Za-z0-9_-]{1,60}\](?:[ \t]*\[[A-Za-z0-9_-]{1,60}\])*")
_LEGACY = re.compile(r"(.+?)\s+\[([A-Za-z0-9_-]{1,60})\]")
_ABBREVIATION = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|No|Fig|Inc|Ltd|e\.g|i\.e|a\.m|p\.m)\."
    r"|\b(?:[A-Z]\.){2,}|\b[A-Z]\.(?=\s+[A-Z][a-z])", re.IGNORECASE
)


def _sentences(line: str) -> list[str]:
    protected = set()
    for match in _ABBREVIATION.finditer(line):
        protected.update(range(match.start(), match.end()))
    for match in re.finditer(r"(?:https?://|www\.)[^\s\[\]<>]+", line):
        url = match[0].rstrip(".!?,;:\"'”’)")
        protected.update(range(match.start(), match.start() + len(url)))
    sentences = []
    start = i = 0
    while i < len(line):
        tags = _TAGS.match(line, i)
        if tags:
            end = tags.end()
            while end < len(line) and line[end] in '.!?"”’':
                end += 1
        elif line[i] in ".!?" and i not in protected:
            if line[i] == "." and i > 0 and i + 1 < len(line) and line[i-1].isdigit() and line[i+1].isdigit():
                i += 1
                continue
            end = i + 1
            while end < len(line) and line[end] in '.!?"”’':
                end += 1
            following = end
            while following < len(line) and line[following].isspace():
                following += 1
            tags = _TAGS.match(line, following)
            if tags:
                end = tags.end()
            elif end < len(line) and not line[end].isspace():
                i += 1
                continue
        else:
            i += 1
            continue
        if line[start:end].strip():
            sentences.append(line[start:end].strip())
        start = i = end
    if line[start:].strip():
        sentences.append(line[start:].strip())
    return sentences


def extract_cited_claims(text: str) -> list[Claim]:
    if text.strip() == "INSUFFICIENT_EVIDENCE":
        return []
    claims = []
    for line in text.splitlines():
        if not line.strip():
            continue
        legacy = _LEGACY.fullmatch(line.strip())
        # A whole passage can contain several sentences. A single trailing tag
        # on its own line is ambiguous; preserve the established quotation form.
        if legacy and len(_TAG.findall(line)) == 1:
            claims.append(Claim(text=legacy[1], source_id=legacy[2], original_line=line))
            continue
        for sentence in _sentences(line):
            ids = _TAG.findall(sentence)
            plain = _TAG.sub("", sentence).strip()
            plain = re.sub(r"\s+([.!?,;:])", r"\1", plain)
            if plain:
                claims.append(Claim(text=plain, source_id=",".join(ids) or None, original_line=sentence))
    return claims
