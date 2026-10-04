#!/usr/bin/env python3
"""
EPUB2 -> MOBI6 generator (Legacy Kindle Target)
"""

from __future__ import annotations

import argparse
import html as htmlmod
import logging
import locale
import os
import posixpath
import re
import shutil
import stat
import struct
import sys
import tempfile
import urllib.parse
import unicodedata
import zipfile
import zlib
from collections import OrderedDict
from copy import deepcopy
# ElementTree is used to keep the converter dependency-free.
# Untrusted XML is size-limited and pre-screened for unsafe declarations in
# _parse_xml(). Applications requiring a hardened external parser may use
# defusedxml.ElementTree instead.
import xml.etree.ElementTree as ET
from xml.parsers import expat

from dataclasses import dataclass, replace
from contextlib import contextmanager
from datetime import datetime
from html.entities import name2codepoint
from pathlib import Path
from typing import BinaryIO, Iterator, Optional, Tuple, Union

# --- LOGGING ---
logger = logging.getLogger("epub2mobi")

# --- CONSTANTS ---
TEXT_RECORD_MAX = 4096

# PalmDB/PDB
PDB_HEADER_LEN = 78
PDB_RECORD_INFO_LEN = 8
PDB_GAP_LEN = 2

# Record 0
PALMDOC_LEN = 16
MOBI_HEADER_LEN = 232
PALMDOC_COMPRESSION = 2  # PalmDOC LZ77

MOBI_MAGIC = b"MOBI"
EXTH_MAGIC = b"EXTH"

# Encoding: Force CP1252 for Old Kindle Compatibility
MOBI_TEXT_ENCODING_ID = 1252        # Windows-1252
MOBI_TEXT_ENCODING_PY = "cp1252"    # Python codec name
MOBI_TEXT_ENCODING_NAME = "windows-1252"

# TOC filepos field width
TOC_FILEPOS_WIDTH = 10
TOC_FILEPOS_MAX = 10 ** TOC_FILEPOS_WIDTH

# Match browser URL whitespace handling before classifying emitted links.
_URL_TRIM_CHARACTERS = "".join(chr(code) for code in range(33))

# XML parsing guardrails for untrusted EPUBs
MAX_XML_BYTES = 8 * 1024 * 1024
MAX_XHTML_BYTES = 16 * 1024 * 1024
MAX_CSS_BYTES = 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_RESOURCE_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 10000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
MAX_DECOMPRESSION_WORK_BYTES = 512 * 1024 * 1024
MAX_RESOURCE_CACHE_BYTES = 8 * 1024 * 1024
MAX_RESOURCE_CACHE_ENTRIES = 128
MAX_DOCUMENT_CACHE_NODES = 65536
MAX_XML_DEPTH = 256  # Root has depth zero.
MAX_XML_ELEMENTS = 100000
MAX_XML_ATTRIBUTES = 200000
MAX_XML_ATTRIBUTE_CHARS = 65536
MAX_BOOK_ENTRIES = 10000  # Each of manifest, spine, and navigation.
MAX_SPINE_ELEMENTS = 1000000
MAX_SPINE_BYTES = 64 * 1024 * 1024  # Source bytes, counting repeated occurrences.
MAX_CSS_RULES = 10000
MAX_CSS_SELECTORS = 50000
MAX_DOCUMENT_CSS_BYTES = 4 * MAX_CSS_BYTES
MAX_CSS_PROCESSING_BYTES = 256 * 1024 * 1024
MAX_CSS_PROCESSING_SELECTORS = 5000000
MAX_OUTPUT_HTML_BYTES = 64 * 1024 * 1024  # Encoded MOBI HTML, including the footer TOC.

# EXTH Types
EXTH_AUTHOR = 100
EXTH_TITLE = 503
EXTH_SOURCE = 112
EXTH_ASIN = 113
EXTH_CDETYPE = 501  # EBOK/PDOC
EXTH_COVER_OFFSET = 201
EXTH_START_READING = 116

DC_NAMESPACE = "{http://purl.org/dc/elements/1.1/}"
OPF_NAMESPACE = "{http://www.idpf.org/2007/opf}"
EPUB_NAMESPACE = "{http://www.idpf.org/2007/ops}"

# Recognized MARC relators, including discontinued codes found in older EPUBs.
# Source: https://www.loc.gov/marc/relators/relacode.html (checked 2026-10-01).
# Keep this local so conversion never needs a network lookup. Unknown roles
# remain unspecified rather than causing creator credits to be discarded.
_MARC_RELATOR_CODES = frozenset("""
    abr acp act adi adp afd aft anc anl anm ann ant ape apl app aqt arc ard arr art
    asg asn ato att auc aud aue aui aup aus aut bdd bjd bka bkd bkp blw bnd bpd brd
    brl bsl cad cas ccp chr clb cli cll clr clt cmm cmp cmt cnd cng cns coe col com
    con cop cor cos cot cou cov cpc cpe cph cpl cpt cre crp crr crt csl csp cst ctb
    cte ctg ctr cts ctt cur cwt dbd dbp dfd dfe dft dgc dgg dgs dis djo dln dnc dnr
    dpc dpt drm drt dsr dst dtc dte dtm dto dub edc edd edm edt egr elg elt eng enj
    etr evp exp fac fds fld flm fmd fmk fmo fmp fnd fon fpy frg gdv gis grt gst his
    hnr hst ill ilu ink ins inv isb itr ive ivr jud jug lbr lbt ldr led lee lel len
    let lgd lie lil lit lsa lse lso ltg ltr lyr mcp mdc med mfp mfr mka mod mon mrb
    mrk msd mte mtk mup mus mxe nan nrt onp opn org orm osp oth own pad pan pat pbd
    pbl pdr pfr pht plt pma pmn pnc pop ppm ppt pra prc prd pre prf prg prm prn pro
    prp prs prt prv pta pte ptf pth ptt pup rap rbr rcd rce rcp rdd red ren res rev
    rpc rps rpt rpy rse rsg rsp rsr rst rth rtm rxa sad sce scl scr sde sds sec sfx
    sgd sgn sht sll sng spk spn spy srv std stg stl stm stn str swd tad tau tcd tch
    ths tld tlg tlh tlp trc trl tyd tyg uvp vac vdg vfx voc wac wal wap wam wat waw
    wdc wde wfs wft wfw win wit wpr wst wts
""".split())

# MOBI Header Offsets (Relative to MOBI Magic)
OFF_LENGTH = 0x04
OFF_TYPE = 0x08
OFF_ENCODING = 0x0C
OFF_UID = 0x10
OFF_VERSION = 0x14

OFF_ORTHO_INDEX = 0x18
OFF_INFLECT_INDEX = 0x1C
OFF_INDEX_NAMES = 0x20
OFF_INDEX_KEYS = 0x24
OFF_EXTRA_INDEX_0 = 0x28
OFF_EXTRA_INDEX_1 = 0x2C
OFF_EXTRA_INDEX_2 = 0x30
OFF_EXTRA_INDEX_3 = 0x34
OFF_EXTRA_INDEX_4 = 0x38
OFF_EXTRA_INDEX_5 = 0x3C

OFF_FIRST_NONBOOK = 0x40
OFF_FULLNAME_O = 0x44
OFF_FULLNAME_L = 0x48
OFF_LOCALE = 0x4C
OFF_MIN_VER = 0x58
OFF_FIRST_IMAGE = 0x5C
OFF_EXTH_FLAGS = 0x70

OFF_UNKNOWN_A4 = 0x94
OFF_DRM_OFFSET = 0x98
OFF_DRM_COUNT = 0x9C

# Content / Magic Pointers
OFF_FIRST_CONTENT = 0xB0  # u16
OFF_LAST_CONTENT = 0xB2   # u16
OFF_UNKNOWN_C4 = 0xB4     # u32

OFF_FCIS_REC = 0xB8
OFF_FCIS_CNT = 0xBC
OFF_FLIS_REC = 0xC0
OFF_FLIS_CNT = 0xC4

# Tail Fields
OFF_TAIL_RESERVED_8 = 0xC8  # 8 bytes zero
OFF_TAIL_E0 = 0xD0          # 0xFFFFFFFF
OFF_TAIL_E4 = 0xD4          # 0
OFF_TAIL_E8 = 0xD8          # 0xFFFFFFFF
OFF_TAIL_EC = 0xDC          # 0xFFFFFFFF
OFF_EXTRA_RECORD_DATA_FLAGS = 0xE0
OFF_INDX = 0xE4

INDX_HEADER_LEN = 192
INDX_TYPE_NORMAL = 0
INDX_TYPE_INFLECTION = 2
INDX_INVALID = 0xFFFFFFFF
INDX_LABEL_ENCODING = 65001  # UTF-8


_HTML_BLOCKS: frozenset[str] = frozenset(
    {
        "p",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "pre",
        "ul",
        "ol",
        "li",
        "br",
        "hr",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "td",
        "th",
    }
)
_HTML_INLINE: frozenset[str] = frozenset(
    {"b", "i", "strong", "em", "sup", "sub", "u", "code", "span", "a", "img", "mbp:pagebreak"}
)
_HTML_ALLOWED: frozenset[str] = _HTML_BLOCKS | _HTML_INLINE
_HTML_FLATTENED_BLOCKS: frozenset[str] = frozenset({
    "dl", "dt", "dd", "section", "article", "aside", "header", "footer",
    "main", "nav", "figure", "figcaption", "address", "caption", "details",
    "summary", "hgroup", "form", "fieldset", "legend", "menu",
})
_HEADING_HINTS: frozenset[str] = frozenset({"chapter-title", "chap-title", "heading", "chapterhead", "chapter-heading"})
_CENTER_HINTS: frozenset[str] = frozenset({"center", "centre", "centered", "centred", "epigraph", "ornament", "separator", "scene-break", "scenebreak", "asterism", "dinkus"})
_RIGHT_HINTS: frozenset[str] = frozenset({"right", "author", "attribution", "credit", "byline", "source"})


def _tokenize_hints(*values: str) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        tokens.update(token for token in re.split(r"[^a-z0-9_-]+", value.lower()) if token)
    return tokens


def _has_heading_hint(elem: ET.Element) -> bool:
    return elem.tag in {"p", "div"} and bool(
        _tokenize_hints(elem.get("class", ""), elem.get("id", "")) & _HEADING_HINTS
    )


def _strip_ns(tag: str) -> str:
    """Remove XML namespace prefix from tag names."""
    return tag.split("}", 1)[1] if "}" in tag else tag


@dataclass(frozen=True)
class MediaOmission:
    resource: str
    source: str
    reason: str


@dataclass(frozen=True)
class EpubData:
    title: str
    author: str
    uuid: str
    html_content: str
    toc_entries: tuple[tuple[str, str], ...]
    image_records: tuple[bytes, ...] = ()
    omitted_media: tuple[MediaOmission, ...] = ()
    language: Optional[str] = None
    cover_index: Optional[int] = None  # Zero-based offset among image_records.
    toc_depths: tuple[int, ...] = ()  # Preorder depths; empty means a flat TOC.
    reading_start_anchor: Optional[str] = None


@dataclass(frozen=True)
class ManifestItem:
    href: str
    media_type: str
    properties: str
    fallback: Optional[str] = None


@dataclass(frozen=True)
class _EpubPackage:
    opf_path: str
    base_dir: str
    title: str
    author: str
    uuid: str
    language: Optional[str]
    metadata: list[ET.Element]
    manifest: dict[str, ManifestItem]
    manifest_paths: dict[str, str]
    spine: ET.Element
    guide: Optional[ET.Element] = None


@dataclass(frozen=True)
class ResolvedTocEntry:
    anchor: str
    label: str
    spine_index: int
    depth: int = 0


@dataclass(frozen=True)
class SpineItem:
    index: int
    full_path: str
    aliases: tuple[str, ...]
    base_url: str
    anchor: str
    stem: str
    document: ET.Element
    body: ET.Element
    fragment_anchors: dict[str, str]
    linear: bool
    styles: Optional[_CssStyles] = None


@dataclass(frozen=True)
class _SpineIndex:
    files: dict[str, str]
    fragments: dict[tuple[str, str], str]
    items: dict[str, SpineItem]


@dataclass(frozen=True)
class TextLayout:
    text_bytes: bytes
    toc_entry_positions: tuple[int, ...]
    body_end: int  # Excludes the generated TOC and closing HTML tags.
    reading_start_offset: Optional[int] = None


@dataclass(frozen=True)
class ReadingStart:
    path: str
    fragment: Optional[str]
    source: str


@dataclass(frozen=True)
class TocTarget:
    path: Optional[str]  # None for an unlinked EPUB3 grouping heading.
    fragment: Optional[str]
    label: str
    parent: Optional[int] = None  # Index of the parent in this source's target list.


class ConversionError(ValueError):
    """Expected failure caused by input, a conversion request, or format limits."""


class _NavigationSizeError(ConversionError):
    """The logical TOC cannot fit the supported single-record layout."""


class _ResourceLimitError(ConversionError):
    """A processing budget is exhausted; optional resources cannot bypass it."""


def _toc_parent_indices(depths: tuple[int, ...]) -> list[Optional[int]]:
    """Validate preorder depths and locate each entry's nearest ancestor."""
    parents: list[Optional[int]] = []
    ancestors: list[int] = []
    for index, depth in enumerate(depths):
        if not isinstance(depth, int) or depth < 0 or depth > len(ancestors):
            raise ValueError(f"Invalid TOC depth at entry {index}: {depth}")
        del ancestors[depth:]
        parents.append(ancestors[-1] if ancestors else None)
        ancestors.append(index)
    return parents


def _palm_time_now() -> int:
    return int((datetime.now() - datetime(1904, 1, 1)).total_seconds())


def _crc32_u32(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def _encode_mobi_text(s: str) -> bytes:
    """Encode as CP1252. Use XML entities for characters that don't fit."""
    return s.encode(MOBI_TEXT_ENCODING_PY, errors="xmlcharrefreplace")


class _HtmlBuffer:
    """Bound encoded output before joining retained HTML fragments."""

    def __init__(self, limit: Optional[int] = None):
        self.limit = MAX_OUTPUT_HTML_BYTES if limit is None else limit
        self.parts: list[str] = []
        self.size = 0

    def append(self, text: str) -> None:
        size = len(_encode_mobi_text(text))
        if self.size + size > self.limit:
            raise _ResourceLimitError("Generated HTML too large")
        self.size += size
        self.parts.append(text)

    def extend(self, parts) -> None:
        for part in parts:
            self.append(part)

    def text(self) -> str:
        return "".join(self.parts)


def _encode_meta(s: str) -> bytes:
    """Encode metadata as CP1252, replacing unsupported characters."""
    return s.encode(MOBI_TEXT_ENCODING_PY, errors="replace")


def _mobi_locale(language: Optional[str]) -> int:
    """Map EPUB language tags to the Windows language IDs used by MOBI.

    Missing/unknown languages use neutral (0). A known language with an
    unmapped variant uses its primary ID without inventing a region/script.
    """
    if not language:
        return 0
    tag = language.strip().lower().replace("_", "-")
    # Common ISO 639-2 aliases used by EPUB 2 producers.
    aliases = {"eng": "en", "ita": "it", "fra": "fr", "fre": "fr",
               "deu": "de", "ger": "de", "spa": "es", "por": "pt",
               "nld": "nl", "dut": "nl", "zho": "zh", "chi": "zh",
               "jpn": "ja", "rus": "ru"}
    parts = tag.split("-")
    parts[0] = aliases.get(parts[0], parts[0])
    tag = "-".join(parts)
    exact = {code for code, name in locale.windows_locale.items()
             if name.lower().replace("_", "-") == tag}
    if len(exact) == 1:
        return exact.pop()
    primary = {code & 0x3ff for code, name in locale.windows_locale.items()
               if name.split("_")[0].lower() == parts[0]}
    if len(primary) == 1:
        code = primary.pop()
        if len(parts) > 1:
            logger.warning("Language variant '%s' has no unambiguous MOBI locale; using primary language", language)
        return code
    logger.warning("Unsupported language '%s'; using neutral MOBI locale", language)
    return 0


def _encode_index_text(s: str) -> bytes:
    return s.encode("utf-8")


def _encode_vwi(value: int) -> bytes:
    if value < 0:
        raise ValueError(f"Negative VWI value: {value}")
    chunks = [value & 0x7F]
    value >>= 7
    while value:
        chunks.append(value & 0x7F)
        value >>= 7
    chunks[0] |= 0x80
    return bytes(reversed(chunks))


def _parse_xml(data: bytes, source_name: str, size_limit: Optional[int] = None,
               *, xhtml_entities: bool = False) -> ET.Element:
    limit = MAX_XML_BYTES if size_limit is None else size_limit
    if len(data) > limit:
        raise ConversionError(f"XML file too large: {source_name}")
    # Expat sees declarations regardless of whether the XML is UTF-8, UTF-16, or UTF-32.
    guard = expat.ParserCreate(namespace_separator="}")

    def reject_entity(*_args):
        raise ConversionError(f"Unsafe XML declaration in {source_name}")

    guard.EntityDeclHandler = reject_entity
    guard.ExternalEntityRefHandler = reject_entity
    encoding = None
    doctype_start = None
    doctype_end = None
    doctype_name = None
    external_doctype = False
    subset_start = None
    subset_end = None
    in_prolog = True
    depth = elements = attributes = name_chars = 0

    def check_attributes(attrs):
        nonlocal attributes
        attributes += len(attrs)
        if attributes > MAX_XML_ATTRIBUTES:
            raise _ResourceLimitError(f"Too many XML attributes: {source_name}")
        if any(max(len(name), len(value)) > MAX_XML_ATTRIBUTE_CHARS for name, value in attrs.items()):
            raise _ResourceLimitError(f"XML attribute too long: {source_name}")

    def namespace(prefix, uri):
        # Expat removes namespace declarations from the element's attributes.
        check_attributes({"xmlns" + (":" + prefix if prefix else ""): uri or ""})

    def start_element(name, attrs):
        nonlocal in_prolog, depth, elements, name_chars
        in_prolog = False
        elements += 1
        if depth > MAX_XML_DEPTH:
            raise _ResourceLimitError(f"XML nesting too deep: {source_name}")
        if elements > MAX_XML_ELEMENTS:
            raise _ResourceLimitError(f"Too many XML elements: {source_name}")
        check_attributes(attrs)
        # A long namespace URI shared by many distinct names expands inside
        # the tree. Bound that work even when the source XML is small.
        name_chars += len(name) + sum(len(key) for key in attrs)
        if name_chars > limit:
            raise _ResourceLimitError(f"XML expanded names too large: {source_name}")
        depth += 1

    def end_element(_name):
        nonlocal depth
        depth -= 1

    def xml_declaration(_version, declared_encoding, _standalone):
        nonlocal encoding
        encoding = declared_encoding

    def declaration_token(token):
        nonlocal doctype_start, doctype_name, external_doctype, subset_start, subset_end
        if not in_prolog:
            return
        if token == "<!DOCTYPE" and doctype_start is None:
            doctype_start = guard.CurrentByteIndex
        elif doctype_start is not None and doctype_end is None:
            if doctype_name is None and token.strip():
                doctype_name = token
            elif token in {"SYSTEM", "PUBLIC"} and subset_start is None:
                external_doctype = True
            elif token == "[" and subset_start is None:
                subset_start = guard.CurrentByteIndex
            elif token == "]":
                subset_end = guard.CurrentByteIndex

    def end_doctype():
        nonlocal doctype_end
        doctype_end = guard.CurrentByteIndex

    guard.XmlDeclHandler = xml_declaration
    guard.StartElementHandler = start_element
    guard.EndElementHandler = end_element
    guard.StartNamespaceDeclHandler = namespace
    guard.DefaultHandler = declaration_token
    guard.EndDoctypeDeclHandler = end_doctype
    try:
        guard.Parse(data, True)
        if xhtml_entities and external_doctype:
            # XMLParser.entity does not expand entities in attributes. Replace
            # the external subset with trusted local declarations, after the
            # guard has rejected all source entity declarations. Removing the
            # external identifier also prevents unknown attribute entities from
            # being silently skipped. No external resource is read.
            if data.startswith((b"\xff\xfe", b"\xfe\xff")):
                encoding = "utf-16"
            elif encoding is None:
                encoding = ("utf-16-le" if data.startswith(b"<\x00") else
                            "utf-16-be" if data.startswith(b"\x00<") else "utf-8")
            text = data.decode(encoding)
            start = len(data[:doctype_start].decode(encoding))
            end = len(data[:doctype_end].decode(encoding)) + 1
            subset = ""
            if subset_start is not None and subset_end is not None:
                left = len(data[:subset_start].decode(encoding)) + 1
                right = len(data[:subset_end].decode(encoding))
                subset = text[left:right]
            declarations = "".join(f'<!ENTITY {name} "&#{code};">'
                                   for name, code in name2codepoint.items()
                                   if name not in {"amp", "lt", "gt", "quot", "apos"})
            doctype = f"<!DOCTYPE {doctype_name} [{subset}{declarations}]>"
            text = text[:start] + doctype + text[end:]
            # The initial guard skips external DTD entities in attributes.
            # Check their locally resolved values before allocating the tree.
            depth = elements = attributes = name_chars = 0
            resolved_guard = expat.ParserCreate(namespace_separator="}")
            resolved_guard.StartElementHandler = start_element
            resolved_guard.EndElementHandler = end_element
            resolved_guard.StartNamespaceDeclHandler = namespace
            resolved_guard.Parse(text, True)
            return ET.fromstring(text)
        return ET.fromstring(data)
    except ConversionError:
        # Parser callbacks raise conversion and processing-limit failures.
        # Preserve them: ConversionError itself is a ValueError subclass.
        raise
    except (ET.ParseError, expat.ExpatError, LookupError, ValueError) as e:
        raise ConversionError(f"Malformed XML: {source_name}") from e


def _find_opf(resources: _ResourceReader) -> tuple[str, str]:
    try:
        root = resources.document("META-INF/container.xml")
    except KeyError as e:
        raise ConversionError("Invalid EPUB container: missing META-INF/container.xml") from e

    opf_path = None
    for elem in root.iter():
        if elem.tag.endswith("rootfile"):
            candidate = elem.attrib.get("full-path")
            if candidate:
                opf_path = candidate
                break
    if not opf_path:
        raise ConversionError("Invalid EPUB container: no rootfile in META-INF/container.xml")

    normalized_opf = _normalize_epub_path(urllib.parse.unquote(opf_path))
    if normalized_opf in ("", "."):
        raise ConversionError("Invalid EPUB container: empty OPF path")
    return normalized_opf, posixpath.dirname(normalized_opf)


def _parse_xhtml(data: bytes, source_name: str) -> ET.Element:
    root = _parse_xml(data, source_name, MAX_XHTML_BYTES, xhtml_entities=True)
    # Normalize XHTML only; foreign vocabularies must not become HTML elements.
    stack = [root]
    while stack:
        elem = stack.pop()
        if elem.tag.startswith("{http://www.w3.org/1999/xhtml}"):
            elem.tag = _strip_ns(elem.tag)
        stack.extend(elem)
    return root


def _body_element(root: ET.Element) -> ET.Element:
    if root.tag == "html":
        body = root.find("body")
        if body is None:
            raise ConversionError("XHTML document has no body")
        return body
    return root


def _suppressed_element(elem: ET.Element) -> bool:
    return _strip_ns(elem.tag) in {"script", "style", "head", "svg"}


def _visible_elements(root: ET.Element):
    if _suppressed_element(root):
        return
    yield root
    for child in root:
        yield from _visible_elements(child)


def _visible_text(root: ET.Element, block_separators: bool = False) -> str:
    if _suppressed_element(root):
        return ""
    text = (root.text or "") + "".join(
        _visible_text(child, block_separators) + (child.tail or "") for child in root
    )
    blocks = _HTML_BLOCKS | _HTML_FLATTENED_BLOCKS
    return f" {text} " if block_separators and root.tag in blocks else text


def _fragment_ids(root: ET.Element) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        value for elem in _visible_elements(root)
        for key, value in elem.attrib.items() if key in {"id", "name"} and value
    ))


def _extract_title(root: ET.Element) -> Optional[str]:
    for tags, elements in (({"h1", "h2"}, _visible_elements(_body_element(root))),
                           ({"title"}, root.iter())):
        for elem in elements:
            if elem.tag in tags:
                title = " ".join(_visible_text(elem).split())
                if title:
                    return title
    return None


def _extract_body_snippet(root: ET.Element, book_title: str, max_words: int = 10) -> Optional[str]:
    # Preserve boundaries between blocks while joining inline text (including drop caps).
    body = _body_element(root)
    text = " ".join(_visible_text(body, block_separators=True).split())
    if not text:
        return None

    normalized_book_title = " ".join(book_title.split()).strip()
    if normalized_book_title and text.lower().startswith(normalized_book_title.lower()):
        text = text[len(normalized_book_title) :].lstrip(" :;,-.")
        text = " ".join(text.split())
        if not text:
            return None

    words = text.split()
    snippet = " ".join(words[:max_words])
    if len(words) > max_words:
        snippet += "..."
    return snippet or None


def _normalize_epub_path(path: str) -> str:
    parts: list[str] = []
    for part in path.replace("\\", "/").split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise ConversionError(f"EPUB path escapes the archive root: {path}")
            parts.pop()
        else:
            parts.append(part)
    return "/".join(parts) or "."


def _split_epub_url(url: str) -> urllib.parse.SplitResult:
    try:
        return urllib.parse.urlsplit(url)
    except ValueError as e:
        raise ConversionError(f"Invalid EPUB URL: {url}") from e


def _resolve_epub_url(base_url: str, href: str) -> urllib.parse.SplitResult:
    """Resolve local URLs without letting urljoin erase traversal above root."""
    base = _split_epub_url(base_url)
    reference = _split_epub_url(href)
    if base.scheme or base.netloc or reference.scheme or reference.netloc:
        return urllib.parse.urlsplit(urllib.parse.urljoin(base_url, href))
    base_path = urllib.parse.unquote(base.path).replace("\\", "/")
    path = urllib.parse.unquote(reference.path).replace("\\", "/")
    if not path:
        target = base_path
    elif path.startswith("/"):
        target = path
    else:
        target = posixpath.dirname(base_path) + "/" + path
    normalized = _normalize_epub_path(target)
    # A local <base href> can name a directory. Keep that distinction when
    # the canonical URL becomes the base of subsequent resource references.
    if target.endswith("/") or target.rsplit("/", 1)[-1] in {".", ".."}:
        normalized = ("" if normalized == "." else normalized) + "/"
    joined = urllib.parse.urlsplit(urllib.parse.urljoin(base_url, href))
    return joined._replace(path=urllib.parse.quote(normalized, safe="/"))


def _resolve_manifest_path(base_dir: str, href: str) -> str:
    base_url = urllib.parse.quote(base_dir, safe="/") + "/" if base_dir else ""
    parsed = _resolve_epub_url(base_url, href)
    if parsed.scheme or parsed.netloc:
        raise ConversionError(f"Invalid EPUB manifest href: {href}")
    return _normalize_epub_path(urllib.parse.unquote(parsed.path))


def _is_supported_image_media_type(media_type: str) -> bool:
    return media_type.lower() in {"image/jpeg", "image/png", "image/gif"}


def _document_base_url(root: ET.Element, current_path: str) -> str:
    document_url = urllib.parse.quote(current_path, safe="/")
    head = root.find("head") if root.tag == "html" else None
    if head is not None:
        for elem in head.iter("base"):
            if "href" in elem.attrib:
                return _resolve_epub_url(document_url, elem.attrib["href"]).geturl()
    return document_url


def _resolve_book_href(current_path: str, href: str,
                       base_url: Optional[str] = None) -> Optional[Tuple[str, Optional[str]]]:
    document_url = urllib.parse.quote(current_path, safe="/") if base_url is None else base_url
    parsed = _resolve_epub_url(document_url, href)
    if parsed.scheme or parsed.netloc:
        return None

    target_path = _normalize_epub_path(urllib.parse.unquote(parsed.path))

    fragment = urllib.parse.unquote(parsed.fragment) if parsed.fragment else None
    return target_path, fragment


def _reported_media_path(current_path: str, href: Optional[str], fallback: str,
                         base_url: Optional[str] = None) -> str:
    if not href:
        return fallback
    if href.startswith("data:"):
        return "<data URI>"
    try:
        resolved = _resolve_book_href(current_path, href, base_url)
    except ConversionError:
        return href
    document_url = urllib.parse.quote(current_path, safe="/") if base_url is None else base_url
    return resolved[0] if resolved else urllib.parse.urljoin(document_url, href)


def _media_references(body: ET.Element):
    images = []
    unsupported = []
    inline_svg = False
    stack = [body]
    while stack:
        elem = stack.pop()
        tag = elem.tag
        if _strip_ns(tag) == "svg":
            inline_svg = True
            continue
        if _suppressed_element(elem):
            continue
        stack.extend(reversed(list(elem)))
        if tag == "img":
            images.append(elem.get("src"))
        elif tag in {"audio", "video"}:
            sources = [elem.get("src")] if elem.get("src") else []
            sources.extend(child.get("src") for child in elem if child.tag == "source" and child.get("src"))
            unsupported.extend((tag, src) for src in (sources or [None]))
        elif tag in {"object", "embed"}:
            unsupported.append((tag, elem.get("data" if tag == "object" else "src")))
    return images, unsupported, inline_svg


def _reading_start_target(current_path: str, href: Optional[str], source: str,
                          base_url: Optional[str] = None) -> Optional[ReadingStart]:
    try:
        if not href:
            raise ConversionError("missing href")
        resolved = _resolve_book_href(current_path, href, base_url)
        if resolved is None:
            raise ConversionError("external destination")
    except ConversionError as e:
        logger.warning("%s reading start: ignored %r: %s", source, href, e)
        return None
    return ReadingStart(resolved[0], resolved[1], source)


def _navigation_label(elem: ET.Element) -> str:
    """Lower a navigation label to text without loading its embedded media."""
    parts: list[str] = []

    def visit(node: ET.Element) -> None:
        tag = _strip_ns(node.tag)
        if tag in {"script", "style", "head"}:
            return
        if tag in {"img", "svg", "audio", "video", "object", "embed", "iframe", "canvas"}:
            alternative = node.get("alt", "")
            if not alternative.strip():
                alternative = node.get("title", "")
            if not alternative.strip() and tag == "svg":
                title = next((child for child in node if _strip_ns(child.tag) == "title"), None)
                alternative = _visible_text(title) if title is not None else ""
            parts.append(alternative)
            return
        parts.append(node.text or "")
        for child in node:
            visit(child)
            parts.append(child.tail or "")

    visit(elem)
    return " ".join("".join(parts).split()) or " ".join(elem.get("title", "").split())


def _extract_nav_targets(
    resources: _ResourceReader,
    manifest: dict[str, ManifestItem],
    base_dir: str,
) -> tuple[list[TocTarget], bool, list[ReadingStart]]:
    nav_href = None
    for item in manifest.values():
        if "nav" in item.properties.split() and item.media_type == "application/xhtml+xml":
            nav_href = item.href
            break
    if not nav_href:
        return [], False, []

    nav_path = nav_href
    try:
        nav_path = _resolve_manifest_path(base_dir, nav_href)
        nav_root = resources.document(nav_path, xhtml=True)
    except _ResourceLimitError:
        raise
    except (KeyError, ConversionError):
        logger.warning("Unable to read EPUB3 nav document: %s", nav_path)
        return [], True, []

    try:
        nav_base_url = _document_base_url(nav_root, nav_path)
    except ConversionError as e:
        logger.warning("Unable to resolve EPUB3 nav base: %s: %s", nav_path, e)
        return [], True, []

    reading_starts: list[ReadingStart] = []
    stack = [(nav_root, False)]
    while stack:
        elem, landmarks = stack.pop()
        if _suppressed_element(elem):
            continue
        if elem.tag == "nav":
            landmarks = "landmarks" in elem.get(EPUB_NAMESPACE + "type", "").split()
        # Hidden landmarks are still navigation metadata. No label is needed.
        if landmarks and elem.tag == "a" and "bodymatter" in elem.get(EPUB_NAMESPACE + "type", "").split():
            target = _reading_start_target(nav_path, elem.get("href"), "EPUB3 bodymatter", nav_base_url)
            if target is not None:
                reading_starts.append(target)
        stack.extend((child, landmarks) for child in reversed(list(elem)))

    toc_nav = None
    for elem in _visible_elements(nav_root):
        if elem.tag != "nav":
            continue
        nav_type = ""
        for key, value in elem.attrib.items():
            local_key = _strip_ns(key).lower()
            if local_key == "type" and value:
                nav_type = value
                break
        if "toc" in nav_type.split():
            toc_nav = elem
            break
    if toc_nav is None:
        return [], False, reading_starts

    toc_targets: list[TocTarget] = []
    invalid_destinations = False
    navigation_entries = 0

    def add_target(elem: ET.Element, parent: Optional[int]) -> Optional[int]:
        nonlocal invalid_destinations, navigation_entries
        navigation_entries += 1
        if navigation_entries > MAX_BOOK_ENTRIES:
            raise _ResourceLimitError(f"Too many navigation entries: {nav_path}")
        label = _navigation_label(elem)
        if not label:
            return parent
        if elem.tag == "span":
            index = len(toc_targets)
            toc_targets.append(TocTarget(path=None, fragment=None, label=label, parent=parent))
            return index
        raw_href = elem.attrib.get("href")
        if not raw_href:
            return parent
        try:
            resolved = _resolve_book_href(nav_path, raw_href, nav_base_url)
        except ConversionError as e:
            logger.warning("EPUB3 nav TOC: skipped invalid destination %s: %s", raw_href, e)
            invalid_destinations = True
            return parent
        if resolved is None:
            return parent
        target_path, fragment = resolved
        index = len(toc_targets)
        toc_targets.append(TocTarget(path=target_path, fragment=fragment, label=label, parent=parent))
        return index

    stack = [(toc_nav, None)]
    while stack:
        elem, parent = stack.pop()
        if _suppressed_element(elem):
            continue
        if elem.tag == "a":
            add_target(elem, parent)
            continue
        # The leading link or grouping span labels this item's nested lists.
        heading = next((child for child in elem if child.tag in {"a", "span"}), None) if elem.tag == "li" else None
        child_parent = add_target(heading, parent) if heading is not None else parent
        stack.extend((child, child_parent) for child in reversed(list(elem)) if child is not heading)
    return toc_targets, invalid_destinations, reading_starts


def _extract_ncx_toc_targets(
    resources: _ResourceReader,
    spine_node: ET.Element,
    manifest: dict[str, ManifestItem],
    base_dir: str,
) -> list[TocTarget]:
    ncx_href = None
    toc_id = spine_node.attrib.get("toc")
    if toc_id in manifest:
        ncx_href = manifest[toc_id].href
    if not ncx_href:
        for item in manifest.values():
            if item.media_type == "application/x-dtbncx+xml" or item.href.lower().endswith(".ncx"):
                ncx_href = item.href
                break
    if not ncx_href:
        return []

    ncx_path = ncx_href
    try:
        ncx_path = _resolve_manifest_path(base_dir, ncx_href)
        ncx_root = resources.document(ncx_path)
    except _ResourceLimitError:
        raise
    except (KeyError, ConversionError):
        logger.warning("Unable to read NCX for TOC labels: %s", ncx_path)
        return []

    targets: list[TocTarget] = []
    points = 0
    stack = [(ncx_root, None)]
    while stack:
        elem, parent = stack.pop()
        child_parent = parent
        if _strip_ns(elem.tag) == "navPoint":
            points += 1
            if points > MAX_BOOK_ENTRIES:
                raise _ResourceLimitError(f"Too many navigation entries: {ncx_path}")
            # Only this navPoint's own label/content may describe it. Searching
            # all descendants can accidentally borrow a nested child's values.
            nav_label = next((child for child in elem if _strip_ns(child.tag) == "navLabel"), None)
            label = None
            if nav_label is not None:
                for child in nav_label:
                    candidate = " ".join((child.text or "").split())
                    if _strip_ns(child.tag) == "text" and candidate:
                        label = candidate
                        break
            content = next((child for child in elem if _strip_ns(child.tag) == "content"), None)
            src = content.get("src") if content is not None else None
            try:
                resolved = _resolve_book_href(ncx_path, src) if src else None
            except ConversionError as e:
                logger.warning("NCX TOC: skipped invalid destination %s: %s", src, e)
                resolved = None
            if label and resolved is not None:
                target_path, fragment = resolved
                child_parent = len(targets)
                targets.append(TocTarget(path=target_path, fragment=fragment, label=label, parent=parent))
        stack.extend((child, child_parent) for child in reversed(list(elem)))

    return targets


def _check_zip_compression(info: zipfile.ZipInfo) -> None:
    if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
        raise ConversionError(
            f"Unsupported EPUB ZIP compression: {info.filename} (method {info.compress_type}); "
            "only stored and DEFLATE entries are supported"
        )


def _validate_archive_member_name(info: zipfile.ZipInfo) -> list[str]:
    # Unlike URI resolution, physical member names must not be repaired.
    raw = info.orig_filename
    if any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in raw):
        raise ConversionError(f"EPUB archive member contains control characters: {raw!r}")
    parts = (raw[:-1] if raw.endswith("/") else raw).split("/")
    if ("\\" in raw or re.match(r"^[A-Za-z]:", raw)
            or any(part in {"", ".", ".."} for part in parts)):
        raise ConversionError(f"Invalid EPUB archive member name: {raw!r}")
    return parts


def _validate_archive(archive: zipfile.ZipFile) -> None:
    entries = archive.infolist()
    if len(entries) > MAX_ARCHIVE_ENTRIES:
        raise ConversionError(f"EPUB archive has too many entries: {len(entries)} > {MAX_ARCHIVE_ENTRIES}")
    declared_total = 0
    seen_names: set[str] = set()
    # Component nodes include implicit directories. Integer parent IDs avoid
    # retaining every full prefix of a deeply nested path.
    nodes: dict[tuple[int, str], tuple[int, bool]] = {}
    canonical_nodes: dict[tuple[int, str], tuple[int, str]] = {}
    collision = None
    for info in entries:
        _check_zip_compression(info)
        declared_total += info.file_size
        if declared_total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise ConversionError(
                f"EPUB archive declared content too large: exceeds {MAX_ARCHIVE_UNCOMPRESSED_BYTES} bytes"
            )
        parts = _validate_archive_member_name(info)
        raw = info.orig_filename
        if raw in seen_names:
            raise ConversionError(f"Duplicate EPUB archive member: {raw!r}")
        seen_names.add(raw)
        parent = canonical_parent = 0
        for index, part in enumerate(parts):
            directory = index < len(parts) - 1 or info.is_dir()
            key = (parent, part)
            if key not in nodes:
                nodes[key] = (len(nodes) + 1, directory)
            parent, previous_directory = nodes[key]
            if previous_directory != directory:
                raise ConversionError(f"EPUB archive file/directory conflict: {raw!r}")
            folded = unicodedata.normalize("NFC", unicodedata.normalize("NFC", part).casefold())
            canonical_key = (canonical_parent, folded)
            if canonical_key not in canonical_nodes:
                canonical_nodes[canonical_key] = (len(canonical_nodes) + 1, part)
            canonical_parent, previous_part = canonical_nodes[canonical_key]
            if previous_part != part and collision is None:
                collision = (previous_part, part, "/".join(parts[:index]) or "/")
    if collision is not None:
        # Exact ZIP lookup remains case sensitive; report interoperability
        # issues without rejecting otherwise usable books or remapping names.
        logger.warning(
            "EPUB archive names differ only by case or Unicode normalization: %r and %r in %s; "
            "exact names are preserved (further collisions are not listed)", *collision,
        )


def _read_zip_member(
    z: zipfile.ZipFile,
    member_path: str,
    *,
    size_limit: int,
    aggregate_budget: int,
    aggregate_used: int,
    kind: str,
    info: Optional[zipfile.ZipInfo] = None,
    accounted_size: Optional[int] = None,
) -> tuple[bytes, int]:
    if info is None:
        try:
            info = z.getinfo(member_path)
        except KeyError as e:
            raise KeyError(member_path) from e

    _check_zip_compression(info)

    if info.file_size > size_limit:
        raise ConversionError(
            f"{kind} too large: {member_path} ({info.file_size} bytes > {size_limit} bytes)"
        )

    # A reread after cache eviction retains the previously validated actual
    # charge, even if the ZIP declaration overstated the extracted size.
    new_total = aggregate_used + (info.file_size if accounted_size is None else accounted_size)
    if new_total > aggregate_budget:
        raise ConversionError(
            f"EPUB content too large: extracting {member_path} would exceed {aggregate_budget} bytes"
        )

    read_errors = (NotImplementedError, RuntimeError, EOFError, UnicodeDecodeError, zlib.error)
    try:
        with z.open(info) as member:
            data = member.read(size_limit + 1)
    except read_errors as e:
        raise ConversionError(f"Cannot read EPUB member {member_path}: {e}") from e
    if len(data) > size_limit:
        raise ConversionError(f"{kind} too large: {member_path}")
    actual_total = aggregate_used + len(data)
    if actual_total > aggregate_budget:
        raise ConversionError(
            f"EPUB content too large: extracting {member_path} would exceed {aggregate_budget} bytes"
        )
    return data, actual_total


def _css_parts(text: str, separators: str):
    """Split outside comments, strings and parentheses/brackets; retain delimiters."""
    part: list[str] = []
    stack: list[str] = []
    quote = None
    i = 0
    while i < len(text):
        char = text[i]
        if quote:
            part.append(char)
            if char == "\\" and i + 1 < len(text):
                i += 1
                part.append(text[i])
            elif char == quote:
                quote = None
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                # An unfinished comment consumes the rest of the source.
                break
            # Keep tokens on either side from merging into a supported identifier.
            part.append(" ")
            i = end + 1
        elif char == "\\":
            # Escaped syntax remains unsupported, but cannot become a delimiter.
            part.append(char)
            if i + 1 < len(text):
                i += 1
                part.append(text[i])
        elif char in "\"'":
            quote = char
            part.append(char)
        elif char in "([":
            stack.append(")" if char == "(" else "]")
            part.append(char)
        elif stack and char == stack[-1]:
            stack.pop()
            part.append(char)
        elif not stack and char in separators:
            yield "".join(part), char
            part = []
        else:
            part.append(char)
        i += 1
    # Unclosed strings/functions must not turn into supported declarations.
    if not quote and not stack:
        yield "".join(part), ""


def _css_value(property_name: str, value: str) -> Optional[str]:
    value = value.strip().lower()
    if property_name == "display":
        # Inspect visibility only; these values never introduce layout rules.
        if value in {"none", "inherit", "initial", "unset", "block", "inline", "inline-block",
                     "list-item", "contents", "flow-root", "flex", "inline-flex", "grid", "inline-grid",
                     "table", "inline-table", "table-row", "table-cell", "table-row-group",
                     "table-header-group", "table-footer-group", "table-column", "table-column-group", "table-caption"}:
            return value
    # Float is inspected only to recognize detached text initials, not rendered.
    if property_name == "float":
        if value in {"left", "right", "none", "inherit"}:
            return value
        if value in {"initial", "unset"}:
            return "none"
    if value == "inherit" and property_name in {"font-style", "font-weight", "text-align", "text-indent"}:
        return value
    if property_name == "font-style" and value in {"normal", "italic", "oblique"}:
        return "italic" if value == "oblique" else value
    if property_name == "font-weight":
        if value in {"normal", "bold"}:
            return value
        if re.fullmatch(r"[1-9]00", value):
            return "bold" if int(value) >= 600 else "normal"
    if property_name == "text-align" and value in {"left", "right", "center", "justify"}:
        return value
    if property_name == "text-indent":
        if value == "0":
            return "0"
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?|\.[0-9]+)(em|pt)", value)
        if match:
            # Normalize without floating point rounding or scientific notation.
            number = match[1].lstrip("0")
            if "." in number:
                number = number.rstrip("0").rstrip(".")
            if not number:
                return "0"
            if number.startswith("."):
                number = "0" + number
            return number + match[2]
    return None


def _css_declaration_values(text: str):
    """Yield declaration names and raw values outside comments and strings."""
    for declaration, _ in _css_parts(text, ";"):
        parts = list(_css_parts(declaration, ":"))
        if len(parts) != 2 or parts[0][1] != ":":
            continue
        yield parts[0][0].strip().lower(), parts[1][0].strip()


def _css_declarations(text: str) -> dict[str, str]:
    declarations: dict[str, str] = {}
    for name, raw_value in _css_declaration_values(text):
        value = _css_value(name, raw_value)
        if value is not None:
            declarations[name] = value
    return declarations


def _css_rules(text: str) -> list[tuple[str, dict[str, str]]]:
    """Read flat rules, skipping entire at-rule and malformed nested blocks."""
    rules = []
    depth = 0
    prelude = ""
    body: list[str] = []
    nested = False
    rule_count = selector_count = 0
    for part, delimiter in _css_parts(text, "{};"):
        if depth == 0:
            if delimiter == "{":
                rule_count += 1
                if rule_count > MAX_CSS_RULES:
                    raise _ResourceLimitError("Too many CSS rules")
                prelude, body, nested = part.strip(), [], False
                depth = 1
        else:
            if delimiter == "{":
                nested = True
                depth += 1
            elif delimiter == "}":
                depth -= 1
                if depth == 0 and not nested and not prelude.startswith("@"):
                    selector_count += prelude.count(",") + 1
                    if selector_count > MAX_CSS_SELECTORS:
                        raise _ResourceLimitError("Too many CSS selectors")
                    declarations = _css_declarations("".join(body) + part)
                    selectors = [selector.strip() for selector in prelude.split(",")]
                    # A selector list containing unsupported syntax is skipped whole.
                    if declarations and all(re.fullmatch(r"\.?(?:[A-Za-z_]|-[A-Za-z_-])[A-Za-z0-9_-]*", s) for s in selectors):
                        rules.extend((s if s.startswith(".") else s.lower(), declarations) for s in selectors)
            if depth == 1 and not nested:
                body.append(part + delimiter)
    return rules


class _CssStyles:
    """Resolve text styles, rendered hiddenness, and the drop-cap float hint."""

    def __init__(self, document: ET.Element, rules: list[tuple[str, dict[str, str]]]):
        by_selector: dict[str, dict[str, tuple[int, str]]] = {}
        for order, (selector, declarations) in enumerate(rules):
            target = by_selector.setdefault(selector, {})
            target.update((name, (order, value)) for name, value in declarations.items())
        self.styles: dict[ET.Element, dict[str, str]] = {}
        self.floats: dict[ET.Element, str] = {}
        self.hidden: set[ET.Element] = set()
        self.font_runs = False

        def resolve(elem: ET.Element, parent: dict[str, str], parent_float: str = "none",
                    parent_hidden: bool = False, parent_display: str = "initial") -> None:
            chosen = {name: (0, order, value)
                      for name, (order, value) in by_selector.get(elem.tag, {}).items()}
            for cls in set(elem.get("class", "").split()):
                for name, (order, value) in by_selector.get("." + cls, {}).items():
                    candidate = (1, order, value)
                    if name not in chosen or candidate[:2] > chosen[name][:2]:
                        chosen[name] = candidate
            local = {name: value for name, (_, _, value) in chosen.items()}
            local.update(_css_declarations(elem.get("style", "")))
            display = local.pop("display", "initial")
            if display == "inherit":
                display = parent_display
            hidden = parent_hidden or "hidden" in elem.attrib or display == "none"
            if hidden:
                self.hidden.add(elem)
            if _suppressed_element(elem):
                return
            floating = local.pop("float", "none")
            if floating == "inherit":
                floating = parent_float
            self.floats[elem] = floating
            self.font_runs |= not hidden and bool(local.keys() & {"font-style", "font-weight"})
            style = dict(parent)
            # Semantic markup provides local defaults, which authored CSS can reset.
            heading_hint = _has_heading_hint(elem)
            if elem.tag in {"b", "strong", "th", "h1", "h2", "h3", "h4", "h5", "h6"} or heading_hint:
                style["font-weight"] = "bold"
            if elem.tag in {"i", "em"}:
                style["font-style"] = "italic"
            for name, value in local.items():
                if value == "inherit":
                    style[name] = parent.get(name, {"font-weight": "normal", "font-style": "normal",
                                                   "text-align": "left", "text-indent": "0"}[name])
                else:
                    style[name] = value
            self.styles[elem] = style
            for child in elem:
                resolve(child, style, floating, hidden, display)

        resolve(document, {})


def _prune_hidden_body(body: ET.Element, styles: _CssStyles) -> bool:
    """Retain reading content, preserving tails outside removed subtrees."""
    if body in styles.hidden:
        return False
    if not styles.hidden:
        return True
    stack = [body]
    while stack:
        parent = stack.pop()
        retained = []
        tails: dict[Optional[ET.Element], list[str]] = {}
        for child in parent:
            if child in styles.hidden:
                if child.tail:
                    previous = retained[-1] if retained else None
                    tails.setdefault(previous, []).append(child.tail)
            else:
                retained.append(child)
                stack.append(child)
        for previous, parts in tails.items():
            if previous is None:
                parent.text = (parent.text or "") + "".join(parts)
            else:
                previous.tail = (previous.tail or "") + "".join(parts)
        parent[:] = retained
    # The retained tree now carries visibility for all downstream consumers.
    for elem in styles.hidden:
        styles.styles.pop(elem, None)
        styles.floats.pop(elem, None)
    styles.hidden.clear()
    return True


def _normalize_drop_caps(body: ET.Element, styles: _CssStyles) -> None:
    """Join unambiguous floated text initials to the following prose paragraph.

    Keep element identities so the precomputed formatting and fragment maps
    survive the move. Only whitespace and block boundaries around the initial
    are removed; its inline markup remains intact.
    """
    if "left" not in styles.floats.values():
        return
    wrappers = {"div", "p", "span", "b", "i", "strong", "em", "u", "a"}
    inline = wrappers - {"div", "p"} | {"sup", "sub", "code"}
    opening_quotes = frozenset("\"'\u2018\u201c\u00ab\u2039\u201e\u201a")
    xml_whitespace = " \t\r\n"

    def initial_layout_text(text: str) -> Optional[str]:
        stripped = text.strip(xml_whitespace)
        if any(char.isspace() for char in stripped):
            return None
        prefix = text[:len(text) - len(text.lstrip(xml_whitespace))]
        suffix = text[len(text.rstrip(xml_whitespace)):]
        # Only newline indentation can be safely removed. Explicit spaces and
        # all non-XML whitespace may separate words, even inside the initial.
        if any(part and "\n" not in part and "\r" not in part for part in (prefix, suffix)):
            return None
        return stripped

    def is_initial(elem: ET.Element) -> bool:
        elements = list(elem.iter())
        if any(node.tag not in wrappers for node in elements):
            return False
        floats = {styles.floats.get(node, "none") for node in elements}
        if "left" not in floats or "right" in floats:
            return False
        for node in elements:
            texts = [node.text or ""]
            if node is not elem:
                texts.append(node.tail or "")
            if any(initial_layout_text(text) is None for text in texts):
                return False
        text = "".join("".join(elem.itertext()).split())
        if text and text[0] in opening_quotes:
            text = text[1:]
        return bool(text and text[0].isalpha()
                    and all(unicodedata.category(char).startswith("M") for char in text[1:]))

    def continuation_start(elem: ET.Element) -> Optional[str]:
        def first(text: str) -> Optional[str]:
            stripped = text.lstrip(xml_whitespace)
            prefix = text[:len(text) - len(stripped)]
            # Newline indentation is XML formatting; a plain leading space may
            # separate a one-letter word ("A dog") and must not be discarded.
            if prefix and "\n" not in prefix and "\r" not in prefix:
                return None
            return stripped[:1]

        start = first(elem.text or "")
        if start is None or start:
            return start
        for child in elem:
            if not _suppressed_element(child):
                if child.tag not in inline:
                    return None
                start = continuation_start(child)
                if start is None or start:
                    return start
            start = first(child.tail or "")
            if start is None or start:
                return start
        return ""

    def trim_start(elem: ET.Element) -> bool:
        if elem.text:
            elem.text = elem.text.lstrip(xml_whitespace)
            if elem.text:
                return True
        for child in elem:
            if not _suppressed_element(child) and trim_start(child):
                return True
            if child.tail:
                child.tail = child.tail.lstrip(xml_whitespace)
                if child.tail:
                    return True
        return False

    def visit(parent: ET.Element) -> None:
        if _suppressed_element(parent):
            return
        children = list(parent)
        for initial, paragraph in zip(children, children[1:]):
            if (paragraph.tag != "p" or (initial.tail or "").strip(xml_whitespace)
                    or styles.floats.get(paragraph, "none") != "none"
                    or _has_heading_hint(paragraph)
                    or not is_initial(initial)):
                continue
            continuation = continuation_start(paragraph)
            # Capitalized words, punctuation, and empty paragraphs are ambiguous.
            if not continuation or not continuation.isalpha() or continuation.isupper():
                continue
            for node in initial.iter():
                if node.tag in {"div", "p"}:
                    node.tag = "span"
                if node.text:
                    node.text = initial_layout_text(node.text)
                if node is not initial and node.tail:
                    node.tail = initial_layout_text(node.tail)
            trim_start(paragraph)
            parent.remove(initial)
            initial.tail = paragraph.text
            paragraph.text = None
            paragraph.insert(0, initial)
        for child in parent:
            visit(child)

    visit(body)


def _marc_creator_role(value: str) -> Optional[str]:
    value = value.strip().lower()
    for prefix in ("http://id.loc.gov/vocabulary/relators/", "https://id.loc.gov/vocabulary/relators/"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    return value if value in _MARC_RELATOR_CODES else None


def _metadata_author(metadata: list[ET.Element]) -> str:
    refined_roles: dict[str, set[str]] = {}
    for elem in metadata:
        if (elem.tag not in {OPF_NAMESPACE + "meta", "meta"}
                or elem.get("property") != "role"
                or elem.get("scheme", "marc:relators") != "marc:relators"):
            continue
        refines = elem.get("refines", "")
        role = _marc_creator_role(elem.text or "")
        if refines.startswith("#") and role:
            refined_roles.setdefault(urllib.parse.unquote(refines[1:]), set()).add(role)

    creators: list[str] = []
    authors: list[str] = []
    for elem in metadata:
        if elem.tag != DC_NAMESPACE + "creator":
            continue
        name = (elem.text or "").strip()
        if not name:
            continue
        creators.append(name)
        roles = set(refined_roles.get(elem.get("id", ""), ()))
        legacy_role = _marc_creator_role(elem.get(OPF_NAMESPACE + "role", ""))
        if legacy_role:
            roles.add(legacy_role)
        if not roles or "aut" in roles:
            authors.append(name)
    if authors:
        return "; ".join(authors)
    if creators:
        logger.warning("No author or untyped creator found; using all creator credits as author metadata")
        return "; ".join(creators)
    return "Unknown"


def _declared_cover_item(metadata: list[ET.Element], manifest: dict[str, ManifestItem]) -> Optional[str]:
    for item_id, item in manifest.items():
        if "cover-image" in item.properties.split():
            return item_id
    for elem in metadata:
        if elem.tag in {OPF_NAMESPACE + "meta", "meta"} and elem.get("name") == "cover":
            item_id = elem.get("content", "").strip()
            if item_id:
                return item_id
    return None


def _raster_signature_matches(data: bytes, media_type: str) -> bool:
    signatures = {"image/jpeg": (b"\xff\xd8\xff",),
                  "image/png": (b"\x89PNG\r\n\x1a\n",),
                  "image/gif": (b"GIF87a", b"GIF89a")}
    return data.startswith(signatures.get(media_type.lower(), ()))


def _read_package(resources: _ResourceReader) -> _EpubPackage:
    book_title = "Unknown"
    book_author = "Unknown"
    book_uuid = "000000000000"
    book_language = None

    opf_path, base_dir = _find_opf(resources)
    try:
        opf_root = resources.document(opf_path)
    except KeyError as e:
        raise ConversionError(f"OPF declared in container is missing: {opf_path}") from e

    unique_id = opf_root.attrib.get("unique-identifier")
    fallback_uuid = None
    chosen_uuid = None

    metadata = list(next((child for child in opf_root if _strip_ns(child.tag) == "metadata"), ()))
    titles: list[str] = []
    for elem in metadata:
        value = (elem.text or "").strip()
        if not value:
            continue
        if elem.tag == DC_NAMESPACE + "title":
            titles.append(value)
        elif elem.tag == DC_NAMESPACE + "language" and not book_language:
            book_language = value
        elif elem.tag == DC_NAMESPACE + "identifier":
            ident = value
            if ident and not fallback_uuid:
                fallback_uuid = ident
            if unique_id and elem.attrib.get("id") == unique_id and ident:
                chosen_uuid = ident

    if titles:
        book_title = titles[0]
    book_author = _metadata_author(metadata)
    if chosen_uuid:
        book_uuid = chosen_uuid
    elif fallback_uuid:
        book_uuid = fallback_uuid

    manifest_node = None
    spine_node = None
    guide_node = None
    for child in list(opf_root):
        t = _strip_ns(child.tag)
        if t == "manifest":
            manifest_node = child
        elif t == "spine":
            spine_node = child
        elif t == "guide" and guide_node is None:
            guide_node = child
    if manifest_node is None or spine_node is None:
        raise ConversionError("Malformed OPF: missing manifest or spine")

    manifest: dict[str, ManifestItem] = {}
    manifest_ids: set[str] = set()
    manifest_count = 0
    for item in list(manifest_node):
        if _strip_ns(item.tag) == "item":
            manifest_count += 1
            if manifest_count > MAX_BOOK_ENTRIES:
                raise _ResourceLimitError(f"Too many manifest entries: {opf_path}")
            iid = item.attrib.get("id")
            href = item.attrib.get("href")
            if iid:
                if iid in manifest_ids:
                    raise ConversionError(f"Duplicate manifest id: {iid!r}")
                manifest_ids.add(iid)
            if iid and href:
                manifest[iid] = ManifestItem(
                    href=href,
                    media_type=item.get("media-type", ""),
                    properties=item.get("properties", ""),
                    fallback=item.get("fallback"),
                )
    manifest_paths: dict[str, str] = {}
    for iid, item in manifest.items():
        try:
            resolved = _resolve_book_href(opf_path, item.href)
        except ConversionError:
            # Required spine paths are rejected when selecting spine items;
            # optional invalid paths are retained only for omission reporting.
            continue
        if resolved is not None:
            path = resolved[0]
            if path in manifest_paths:
                raise ConversionError(
                    f"Duplicate manifest local resource path: {manifest_paths[path]!r} and {iid!r} "
                    f"both resolve to {path!r}"
                )
            manifest_paths[path] = iid

    return _EpubPackage(
        opf_path=opf_path, base_dir=base_dir, title=book_title, author=book_author,
        uuid=book_uuid, language=book_language, metadata=metadata, manifest=manifest,
        manifest_paths=manifest_paths, spine=spine_node, guide=guide_node,
    )


class _ResourceReader:
    """Own unique-member accounting, bounded caches, and omission reporting."""

    def __init__(self, archive: zipfile.ZipFile):
        self.archive = archive
        self.bytes_read = 0
        self.work_bytes = 0
        self.spine_bytes = 0
        self.spine_nodes = 0
        # Header offsets identify selected physical entries, including ZIPs
        # containing duplicate filenames. Accounting survives cache eviction.
        self._member_sizes: dict[int, int] = {}
        self._raw_cache: OrderedDict[tuple[str, int], bytes] = OrderedDict()
        self._raw_bytes = 0
        self._documents: OrderedDict[tuple[str, int, bool], tuple[ET.Element, int, int]] = OrderedDict()
        self._document_bytes = 0
        self._document_nodes = 0
        self.omissions: list[MediaOmission] = []
        self._seen_omissions: set[MediaOmission] = set()

    def read(self, path: str, *, size_limit: int, kind: str) -> bytes:
        path = _normalize_epub_path(path)
        return self._read(path, self.archive.getinfo(path), size_limit=size_limit, kind=kind)

    @staticmethod
    def _check_size(path: str, info: zipfile.ZipInfo, actual: int, size_limit: int, kind: str) -> None:
        _check_zip_compression(info)
        if max(info.file_size, actual) > size_limit:
            raise ConversionError(f"{kind} too large: {path}")

    def _read(self, path: str, info: zipfile.ZipInfo, *, size_limit: int, kind: str) -> bytes:
        member = info.header_offset
        # Including the path forces ZIP header validation if malformed central
        # entries claim different filenames for the same physical offset.
        key = (path, member)
        data = self._raw_cache.get(key)
        if data is not None:
            self._check_size(path, info, len(data), size_limit, kind)
            self._raw_cache.move_to_end(key)
            return data
        self._check_size(path, info, 0, size_limit, kind)
        accounted = self._member_sizes.get(member)
        aggregate_used = self.bytes_read - (accounted or 0)
        if aggregate_used + (info.file_size if accounted is None else accounted) > MAX_TOTAL_RESOURCE_BYTES:
            raise ConversionError(
                f"EPUB content too large: extracting {path} would exceed {MAX_TOTAL_RESOURCE_BYTES} bytes"
            )
        # Reserve both input and output work before opening. Failed reads and
        # rereads consume work; serving cached bytes or trees does not.
        work = info.compress_size + info.file_size
        if self.work_bytes + work > MAX_DECOMPRESSION_WORK_BYTES:
            raise _ResourceLimitError(
                f"EPUB decompression work too large: reading {path} would exceed "
                f"{MAX_DECOMPRESSION_WORK_BYTES} bytes"
            )
        self.work_bytes += work
        data, total = _read_zip_member(
            self.archive, path, size_limit=size_limit,
            aggregate_budget=MAX_TOTAL_RESOURCE_BYTES,
            aggregate_used=aggregate_used, kind=kind, info=info, accounted_size=accounted,
        )
        self.bytes_read = total
        self._member_sizes[member] = len(data)
        if len(data) <= MAX_RESOURCE_CACHE_BYTES and MAX_RESOURCE_CACHE_ENTRIES > 0:
            while self._raw_cache and (self._raw_bytes + len(data) > MAX_RESOURCE_CACHE_BYTES
                                      or len(self._raw_cache) >= MAX_RESOURCE_CACHE_ENTRIES):
                _, evicted = self._raw_cache.popitem(last=False)
                self._raw_bytes -= len(evicted)
            self._raw_cache[key] = data
            self._raw_bytes += len(data)
        return data

    def document(self, path: str, *, xhtml: bool = False, mutable: bool = False,
                 size_limit: Optional[int] = None, kind: str = "XML file") -> ET.Element:
        path = _normalize_epub_path(path)
        info = self.archive.getinfo(path)
        limit = MAX_XML_BYTES if size_limit is None else size_limit
        key = (path, info.header_offset, xhtml)
        cached = self._documents.get(key)
        if cached is not None:
            root, source_size, nodes = cached
            self._check_size(path, info, source_size, limit, kind)
            self._documents.move_to_end(key)
            return self._spine_document(root, source_size, nodes, copy=True) if mutable else root
        data = self._read(path, info, size_limit=limit, kind=kind)
        root = _parse_xhtml(data, path) if xhtml else _parse_xml(data, path)
        nodes = sum(1 for _ in root.iter())
        if (len(data) <= MAX_RESOURCE_CACHE_BYTES and nodes <= MAX_DOCUMENT_CACHE_NODES
                and MAX_RESOURCE_CACHE_ENTRIES > 0):
            while self._documents and (
                self._document_bytes + len(data) > MAX_RESOURCE_CACHE_BYTES
                or self._document_nodes + nodes > MAX_DOCUMENT_CACHE_NODES
                or len(self._documents) >= MAX_RESOURCE_CACHE_ENTRIES
            ):
                _, (_, source_size, evicted_nodes) = self._documents.popitem(last=False)
                self._document_bytes -= source_size
                self._document_nodes -= evicted_nodes
            self._documents[key] = root, len(data), nodes
            self._document_bytes += len(data)
            self._document_nodes += nodes
            return self._spine_document(root, len(data), nodes, copy=True) if mutable else root
        return self._spine_document(root, len(data), nodes, copy=False) if mutable else root

    def _spine_document(self, root: ET.Element, source_size: int, nodes: int, *, copy: bool) -> ET.Element:
        # A cached member may appear many times in the spine. Reserve its
        # processing cost on every occurrence, before retaining another tree.
        if self.spine_bytes + source_size > MAX_SPINE_BYTES or self.spine_nodes + nodes > MAX_SPINE_ELEMENTS:
            raise _ResourceLimitError("EPUB spine processing limit exceeded")
        self.spine_bytes += source_size
        self.spine_nodes += nodes
        return deepcopy(root) if copy else root

    def omit(self, resource: str, source: str, reason: str) -> None:
        omission = MediaOmission(resource=resource, source=source, reason=reason)
        if omission not in self._seen_omissions:
            self.omissions.append(omission)
            self._seen_omissions.add(omission)


def _report_manifest_omissions(package: _EpubPackage, resources: _ResourceReader) -> None:
    font_media_types = {
        "application/font-sfnt", "application/vnd.ms-opentype",
        "application/x-font-ttf", "application/x-font-otf",
    }
    for item in package.manifest.values():
        href = item.href
        media_type = item.media_type.lower()
        try:
            resource_path = _split_epub_url(href).path.lower()
            resolved = _resolve_book_href(package.opf_path, href)
        except ConversionError as e:
            resources.omit(href, package.opf_path, str(e))
            continue
        is_font = (
            media_type.startswith("font/")
            or media_type in font_media_types
            or resource_path.endswith((".ttf", ".otf", ".woff", ".woff2"))
        )
        if resolved is None:
            resources.omit(
                _reported_media_path(package.opf_path, href, href), package.opf_path,
                "remote or data URI manifest resource is not embedded",
            )
        elif is_font:
            resources.omit(
                _resolve_manifest_path(package.base_dir, href), package.opf_path,
                "declared font is not embedded",
            )


def _spine_references(spine: ET.Element) -> list[tuple[str, bool]]:
    linear_refs: list[tuple[str, bool]] = []
    auxiliary_refs: list[tuple[str, bool]] = []
    count = 0
    for itemref in list(spine):
        if _strip_ns(itemref.tag) != "itemref":
            continue
        count += 1
        if count > MAX_BOOK_ENTRIES:
            raise _ResourceLimitError("Too many spine entries")
        rid = itemref.attrib.get("idref")
        if rid:
            linear = itemref.attrib.get("linear", "yes").strip().lower() != "no"
            (linear_refs if linear else auxiliary_refs).append((rid, linear))
    return linear_refs + auxiliary_refs


def _select_spine_item(item_id: str, package: _EpubPackage, resources: _ResourceReader) -> tuple[str, tuple[str, ...]]:
    chain: list[str] = []
    visited: set[str] = set()
    current = item_id
    while True:
        if current in visited:
            raise ConversionError(f"Cyclic manifest fallback chain: {item_id}")
        if current not in package.manifest:
            raise ConversionError(f"Malformed OPF: spine/fallback item '{current}' is missing from manifest")
        chain.append(current)
        visited.add(current)
        media_type = package.manifest[current].media_type.lower()
        local = _resolve_book_href(package.opf_path, package.manifest[current].href) is not None
        if local and media_type == "application/xhtml+xml":
            break
        fallback = package.manifest[current].fallback
        if not fallback:
            if not local:
                raise ConversionError(f"Remote spine content is not supported: {package.manifest[current].href}")
            if media_type == "image/svg+xml":
                break  # Retain the existing SVG omission behavior.
            raise ConversionError(f"Unsupported spine media type without readable fallback: {media_type}")
        current = fallback
    aliases = tuple(_resolve_manifest_path(package.base_dir, package.manifest[iid].href) for iid in chain
                    if _resolve_book_href(package.opf_path, package.manifest[iid].href) is not None)
    for iid in chain[:-1]:
        resources.omit(_reported_media_path(package.opf_path, package.manifest[iid].href, package.manifest[iid].href),
                        package.opf_path, "spine resource replaced by manifest fallback")
    return current, aliases


def _load_spine(package: _EpubPackage, resources: _ResourceReader) -> tuple[list[SpineItem], set[str]]:
    spine_items: list[SpineItem] = []
    svg_spine_paths: set[str] = set()
    for spine_idx, (item_id, linear) in enumerate(_spine_references(package.spine), start=1):
        item_id, aliases = _select_spine_item(item_id, package, resources)
        rel = package.manifest[item_id].href
        full = _resolve_manifest_path(package.base_dir, rel)
        if package.manifest[item_id].media_type.lower() == "image/svg+xml":
            svg_spine_paths.add(full)
            resources.omit(full, package.opf_path, "SVG spine content is not rendered")
        try:
            document = resources.document(full, xhtml=True, mutable=True,
                                          size_limit=MAX_XHTML_BYTES, kind="Spine XHTML")
        except KeyError as e:
            raise ConversionError(f"Missing spine item in EPUB: {full}") from e
        try:
            body = _body_element(document)
        except ConversionError as e:
            raise ConversionError(f"Invalid XHTML: {full}: {e}") from e
        anchor = f"spine_{spine_idx}"
        spine_items.append(
            SpineItem(
                index=spine_idx,
                full_path=full,
                aliases=aliases,
                base_url=_document_base_url(document, full),
                anchor=anchor,
                stem=Path(rel).stem,
                document=document,
                body=body,
                fragment_anchors={},
                linear=linear,
            )
        )

    return spine_items, svg_spine_paths


def _index_spine(spine_items: list[SpineItem]) -> _SpineIndex:
    file_anchor_map: dict[str, str] = {}
    fragment_anchor_map: dict[tuple[str, str], str] = {}
    spine_by_path: dict[str, SpineItem] = {}
    # References from other documents to a shared fallback choose its first
    # occurrence. Each occurrence retains its own markers and self-links.
    for item in spine_items:
        for path in item.aliases:
            spine_by_path.setdefault(path, item)
            file_anchor_map.setdefault(path, item.anchor)
            for fragment, anchor in item.fragment_anchors.items():
                fragment_anchor_map.setdefault((path, fragment), anchor)

    return _SpineIndex(file_anchor_map, fragment_anchor_map, spine_by_path)


class _ImageLoader:
    """Load referenced raster images and declared covers, reusing record indexes."""

    def __init__(self, package: _EpubPackage, resources: _ResourceReader):
        self.package = package
        self.resources = resources
        self.by_path: dict[str, int] = {}
        self.records: list[bytes] = []
        self.cover_index: Optional[int] = None

    def embed(self, target_path: str, source: str, item_id: Optional[str] = None,
              *, cover: bool = False) -> Optional[int]:
        item_id = item_id if item_id is not None else self.package.manifest_paths.get(target_path)
        if item_id is None:
            self.resources.omit(target_path, source, "image is not declared in the EPUB manifest")
            return None
        media_type = self.package.manifest[item_id].media_type
        if not _is_supported_image_media_type(media_type):
            self.resources.omit(target_path, source, f"unsupported image format: {media_type or 'unspecified'}")
            return None
        recindex = self.by_path.get(target_path)
        if recindex is not None:
            image_data = self.records[recindex - 1]
        else:
            try:
                image_data = self.resources.read(target_path, size_limit=MAX_IMAGE_BYTES, kind="Image resource")
            except KeyError:
                self.resources.omit(target_path, source, "image file is missing from the EPUB")
                return None
        if not image_data:
            self.resources.omit(target_path, source, "image file is empty")
            return None
        if not _raster_signature_matches(image_data, media_type):
            kind = "cover image" if cover else "image"
            self.resources.omit(target_path, source, f"{kind} signature does not match its declared raster format")
            return None
        if recindex is None:
            self.records.append(image_data)
            recindex = len(self.records)
            self.by_path[target_path] = recindex
        return recindex

    def load(self, spine_items: list[SpineItem], svg_spine_paths: set[str]) -> None:
        for item in spine_items:
            image_sources, unsupported, inline_svg = _media_references(item.body)
            if inline_svg and item.full_path not in svg_spine_paths:
                self.resources.omit("<inline SVG>", item.full_path, "inline SVG is not rendered")
            for kind, raw_src in unsupported:
                resource = _reported_media_path(item.full_path, raw_src, f"<{kind}>", item.base_url)
                self.resources.omit(resource, item.full_path, f"{kind} is not supported")

            for raw_src in image_sources:
                if not raw_src:
                    self.resources.omit("<img without src>", item.full_path, "image has no source")
                    continue
                try:
                    resolved = _resolve_book_href(item.full_path, raw_src, item.base_url)
                except ConversionError as e:
                    self.resources.omit(raw_src, item.full_path, str(e))
                    continue
                if resolved is None:
                    resource = _reported_media_path(item.full_path, raw_src, raw_src, item.base_url)
                    self.resources.omit(resource, item.full_path, "external or data URI image is not embedded")
                    continue
                target_path, _fragment = resolved
                self.embed(target_path, item.full_path)

        self.cover_index = None
        cover_id = _declared_cover_item(self.package.metadata, self.package.manifest)
        if cover_id is not None:
            cover_item = self.package.manifest.get(cover_id)
            if cover_item is None:
                self.resources.omit(cover_id, self.package.opf_path, "cover item is not declared in the EPUB manifest")
            else:
                cover_href = cover_item.href
                try:
                    resolved = _resolve_book_href(self.package.opf_path, cover_href)
                except ConversionError as e:
                    self.resources.omit(cover_href, self.package.opf_path, str(e))
                    return
                if resolved is None:
                    self.resources.omit(_reported_media_path(self.package.opf_path, cover_href, cover_href),
                                        self.package.opf_path, "external or data URI cover is not embedded")
                else:
                    recindex = self.embed(resolved[0], self.package.opf_path, cover_id, cover=True)
                    if recindex is not None:
                        self.cover_index = recindex - 1


class _StylesheetLoader:
    """Cache stylesheet reads while reporting failures for each source document."""

    def __init__(self, resources: _ResourceReader):
        self.resources = resources
        self.cache: dict[str, tuple[int, list[tuple[str, dict[str, str]]]]] = {}
        self.cached_selectors = 0
        self.processed_selectors = 0
        self.processed_bytes = 0
        self.errors: dict[str, str] = {}

    def for_document(self, item: SpineItem) -> _CssStyles:
        rules: list[tuple[str, dict[str, str]]] = []
        css_bytes = 0

        def reserve(size: int) -> None:
            nonlocal css_bytes
            css_bytes += size
            if css_bytes > MAX_DOCUMENT_CSS_BYTES:
                raise _ResourceLimitError(f"Document CSS too large: {item.full_path}")
            if self.processed_bytes + size > MAX_CSS_PROCESSING_BYTES:
                raise _ResourceLimitError("EPUB CSS processing byte limit exceeded")
            self.processed_bytes += size

        def add_rules(additional: list[tuple[str, dict[str, str]]]) -> None:
            if len(rules) + len(additional) > MAX_CSS_SELECTORS:
                raise _ResourceLimitError(f"Too many document CSS selectors: {item.full_path}")
            if self.processed_selectors + len(additional) > MAX_CSS_PROCESSING_SELECTORS:
                raise _ResourceLimitError("EPUB CSS processing selector limit exceeded")
            self.processed_selectors += len(additional)
            rules.extend(additional)

        stack = [item.document]
        while stack:
            elem = stack.pop()
            # Head styles are intentional; foreign vocabularies and scripts are not.
            if elem.tag.startswith("{") or elem.tag in {"svg", "script"}:
                continue
            if elem.tag != "style":
                stack.extend(reversed(list(elem)))
            if elem.tag not in {"style", "link"}:
                continue
            if (elem.get("type", "text/css").strip().lower() not in {"", "text/css"}
                    or elem.get("media", "all").strip().lower() not in {"", "all"}
                    or "disabled" in elem.attrib):
                continue
            if elem.tag == "style":
                text = "".join(elem.itertext())
                size = len(text.encode("utf-8"))
                if size > MAX_CSS_BYTES:
                    raise ConversionError(f"CSS file too large: embedded style in {item.full_path}")
                reserve(size)
                add_rules(_css_rules(text))
                continue
            rel = elem.get("rel", "").lower().split()
            href = elem.get("href")
            if "stylesheet" not in rel or "alternate" in rel or not href:
                continue
            try:
                resolved = _resolve_book_href(item.full_path, href, item.base_url)
            except ConversionError as e:
                self.resources.omit(href, item.full_path, str(e))
                continue
            if resolved is None:
                self.resources.omit(_reported_media_path(item.full_path, href, href, item.base_url),
                                    item.full_path, "external or data URI stylesheet is not loaded")
                continue
            path = resolved[0]
            if path not in self.cache:
                self.cache[path] = (0, [])
                try:
                    data = self.resources.read(path, size_limit=MAX_CSS_BYTES, kind="CSS file")
                except KeyError:
                    self.errors[path] = "stylesheet is missing"
                else:
                    reserve(len(data))
                    self.cache[path] = (len(data), [])
                    try:
                        text = data.decode("utf-8-sig")
                    except UnicodeDecodeError:
                        self.errors[path] = "stylesheet is not UTF-8"
                    else:
                        parsed_rules = _css_rules(text)
                        if self.cached_selectors + len(parsed_rules) > MAX_CSS_SELECTORS:
                            raise _ResourceLimitError("Too many cached CSS selectors")
                        self.cached_selectors += len(parsed_rules)
                        self.cache[path] = (len(data), parsed_rules)
            else:
                reserve(self.cache[path][0])
            if path in self.errors:
                self.resources.omit(path, item.full_path, self.errors[path])
            add_rules(self.cache[path][1])
        return _CssStyles(item.document, rules)


def _prepare_spine(spine_items: list[SpineItem], loader: _StylesheetLoader) -> list[SpineItem]:
    retained: list[SpineItem] = []
    for item in spine_items:
        styles = loader.for_document(item)
        if not _prune_hidden_body(item.body, styles):
            continue
        body_fragments = tuple(item.body.get(key) for key in ("id", "name")
                               if item.body.tag == "body" and item.body.get(key))
        anchors = {fragment: item.anchor for fragment in body_fragments}
        for fragment_idx, fragment in enumerate(_fragment_ids(item.body), start=1):
            anchors.setdefault(fragment, f"{item.anchor}_frag_{fragment_idx}")
        retained.append(replace(item, fragment_anchors=anchors, styles=styles))
    return retained


def _render_spine(package: _EpubPackage, spine_items: list[SpineItem], index: _SpineIndex,
                  images: _ImageLoader,
                  ncx_targets: list[TocTarget]) -> tuple[str, list[ResolvedTocEntry]]:
    parts = _HtmlBuffer()
    fallback_toc: list[ResolvedTocEntry] = []
    for item in spine_items:
        parts.append(f'<a name="{item.anchor}" id="{item.anchor}"></a>')
        sanitizer = MinimalHtmlSanitizer(
            current_path=item.full_path,
            file_anchor_map=index.files,
            fragment_anchor_map=index.fragments,
            image_path_to_recindex=images.by_path,
            base_url=item.base_url,
            current_aliases=item.aliases,
            fragment_anchors=item.fragment_anchors,
            file_anchor=item.anchor,
            styles=item.styles,
            output_limit=parts.limit - parts.size,
        )
        clean = sanitizer.sanitize(item.body)
        # Derive fallback labels from the normalized reading text too.
        ncx_title = next((target.label for target in ncx_targets if target.path in item.aliases and target.fragment is None), None)
        guessed_title = _extract_title(item.document)
        chapter_title = ncx_title or guessed_title
        if (
            not chapter_title
            or chapter_title.strip().lower() in ("unknown", "untitled")
            or chapter_title.strip().lower() == package.title.strip().lower()
        ):
            body_snippet = _extract_body_snippet(item.document, package.title)
            chapter_title = body_snippet or chapter_title or item.stem or item.anchor

        if clean:
            if item.linear:
                fallback_toc.append(ResolvedTocEntry(item.anchor, chapter_title, item.index))
            parts.append(clean)
            parts.append("<mbp:pagebreak/>")

    return parts.text(), fallback_toc


def _resolve_toc_targets(targets: list[TocTarget], source: str, spine_index: _SpineIndex) -> tuple[list[ResolvedTocEntry], bool]:
    destinations: list[Optional[tuple[str, SpineItem]]] = []
    unresolved = 0
    for target in targets:
        if target.path is None:
            destinations.append(None)
            continue
        anchor = (spine_index.fragments.get((target.path, target.fragment))
                  if target.fragment else spine_index.files.get(target.path))
        item = spine_index.items.get(target.path)
        destination = (anchor, item) if anchor is not None and item is not None else None
        destinations.append(destination)
        unresolved += destination is None

    # Give unlinked groups their first retained descendant's destination.
    # A reverse pass handles nested groups and skipped links in linear work.
    first_destinations = destinations.copy()
    for index in range(len(targets) - 1, -1, -1):
        parent = targets[index].parent
        if parent is not None and destinations[parent] is None and first_destinations[index] is not None:
            first_destinations[parent] = first_destinations[index]

    entries: list[ResolvedTocEntry] = []
    retained: dict[int, Optional[int]] = {}
    for index, target in enumerate(targets):
        parent = retained.get(target.parent)
        # Children of skipped nodes attach to their nearest retained ancestor.
        retained[index] = parent
        destination = first_destinations[index] if target.path is None else destinations[index]
        if destination is None:
            continue
        anchor, spine_item = destination
        # Authored aliases can have different labels or child branches. Keep
        # each occurrence in preorder, even when its destination is shared.
        depth = entries[parent].depth + 1 if parent is not None else 0
        retained[index] = len(entries)
        entries.append(ResolvedTocEntry(anchor, target.label, spine_item.index, depth))
    if unresolved:
        logger.warning("%s TOC: skipped %d unresolved destination%s", source, unresolved,
                       "" if unresolved == 1 else "s")
    return entries, bool(unresolved)


def _resolve_reading_start(targets: list[ReadingStart], index: _SpineIndex) -> Optional[str]:
    anchors: set[str] = set()
    unresolved = 0
    for target in targets:
        item = index.items.get(target.path)
        anchor = (index.fragments.get((target.path, target.fragment)) if target.fragment
                  else index.files.get(target.path))
        if anchor is None or item is None or _suppressed_element(item.body):
            unresolved += 1
        else:
            anchors.add(anchor)
    if unresolved:
        logger.warning("%s reading start: ignored %d unresolved destination(s)", targets[0].source, unresolved)
    if len(anchors) > 1:
        logger.warning("%s reading start: ignored ambiguous destinations", targets[0].source)
        return None
    return next(iter(anchors), None)


def _select_reading_start(nav_starts: list[ReadingStart], package: _EpubPackage,
                         index: _SpineIndex) -> Optional[str]:
    anchor = _resolve_reading_start(nav_starts, index)
    if anchor is not None or package.guide is None:
        return anchor
    guide_starts: list[ReadingStart] = []
    for reference in package.guide:
        if _strip_ns(reference.tag) != "reference" or reference.get("type") != "text":
            continue
        target = _reading_start_target(package.opf_path, reference.get("href"), "EPUB2 guide")
        if target is not None:
            guide_starts.append(target)
    return _resolve_reading_start(guide_starts, index)


def _select_toc(nav_targets: list[TocTarget], ncx_targets: list[TocTarget],
                fallback_toc: list[ResolvedTocEntry], index: _SpineIndex,
                nav_invalid: bool = False) -> list[ResolvedTocEntry]:
    nav_toc, nav_incomplete = _resolve_toc_targets(nav_targets, "EPUB3 nav", index)
    ncx_toc, _ = _resolve_toc_targets(ncx_targets, "NCX", index)
    # Preserve authored TOCs even when some links are stale. A more complete
    # NCX can replace an incomplete nav; spine guesses are only a last resort.
    if nav_toc:
        raw_toc = ncx_toc if (nav_incomplete or nav_invalid) and len(ncx_toc) > len(nav_toc) else nav_toc
    else:
        raw_toc = ncx_toc or fallback_toc

    return raw_toc


def _finalize_toc(raw_toc: list[ResolvedTocEntry]) -> tuple[tuple[tuple[str, str], ...], tuple[int, ...]]:
    toc_depths = tuple(entry.depth for entry in raw_toc)
    parents = _toc_parent_indices(toc_depths)
    label_counts: dict[tuple[Optional[int], str], int] = {}
    for entry, parent in zip(raw_toc, parents):
        key = (parent, entry.label)
        label_counts[key] = label_counts.get(key, 0) + 1

    toc_entries: list[tuple[str, str]] = []
    used_labels: set[tuple[Optional[int], str]] = set()
    next_numbers: dict[tuple[Optional[int], str], int] = {}
    for entry, parent in zip(raw_toc, parents):
        # Repeated labels in different books/sections are already distinct
        # through their parents; only siblings need disambiguation.
        key = (parent, entry.label)
        resolved = entry.label
        if label_counts[key] > 1:
            number = next_numbers.get(key, 1)
            resolved = f"{entry.label} ({number})"
            # Reserve authored labels, including those occurring later. Each
            # sibling group advances its counter rather than restarting it.
            while (parent, resolved) in label_counts or (parent, resolved) in used_labels:
                number += 1
                resolved = f"{entry.label} ({number})"
            next_numbers[key] = number + 1
        used_labels.add((parent, resolved))
        toc_entries.append((entry.anchor, resolved))

    return tuple(toc_entries), toc_depths


def parse_epub(filepath: Union[str, Path]) -> EpubData:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(str(filepath))

    # Check the same open file that ZipFile will use, before it loads ZIP
    # metadata. Keep both handles scoped through failures during validation.
    with filepath.open("rb") as source:
        archive_size = os.fstat(source.fileno()).st_size
        if archive_size > MAX_ARCHIVE_BYTES:
            raise ConversionError(f"EPUB archive too large: {archive_size} bytes > {MAX_ARCHIVE_BYTES} bytes")
        try:
            archive = zipfile.ZipFile(source, "r")
        except (NotImplementedError, UnicodeDecodeError) as e:
            raise ConversionError(f"Cannot open EPUB archive {filepath}: {e}") from e
        with archive:
            _validate_archive(archive)
            return _parse_archive(archive)


def _parse_archive(archive: zipfile.ZipFile) -> EpubData:
    resources = _ResourceReader(archive)
    package = _read_package(resources)
    _report_manifest_omissions(package, resources)
    nav_targets, nav_invalid, nav_starts = _extract_nav_targets(resources, package.manifest, package.base_dir)
    ncx_targets = _extract_ncx_toc_targets(resources, package.spine, package.manifest, package.base_dir)
    spine_items, svg_spine_paths = _load_spine(package, resources)
    spine_items = _prepare_spine(spine_items, _StylesheetLoader(resources))
    index = _index_spine(spine_items)
    images = _ImageLoader(package, resources)
    images.load(spine_items, svg_spine_paths)
    html_content, fallback_toc = _render_spine(package, spine_items, index, images, ncx_targets)
    raw_toc = _select_toc(nav_targets, ncx_targets, fallback_toc, index, nav_invalid)
    toc_entries, toc_depths = _finalize_toc(raw_toc)

    logger.info("Parsed %d spine items. Title: %s", len(spine_items), package.title)
    return EpubData(
        title=package.title,
        author=package.author,
        uuid=package.uuid,
        html_content=html_content,
        toc_entries=toc_entries,
        image_records=tuple(images.records),
        omitted_media=tuple(resources.omissions),
        language=package.language,
        cover_index=images.cover_index,
        toc_depths=toc_depths,
        reading_start_anchor=_select_reading_start(nav_starts, package, index),
    )


class MinimalHtmlSanitizer:
    def __init__(
        self,
        current_path: str,
        file_anchor_map: dict[str, str],
        fragment_anchor_map: dict[tuple[str, str], str],
        image_path_to_recindex: dict[str, int],
        base_url: Optional[str] = None,
        current_aliases: Optional[tuple[str, ...]] = None,
        fragment_anchors: Optional[dict[str, str]] = None,
        file_anchor: Optional[str] = None,
        styles: Optional[_CssStyles] = None,
        output_limit: Optional[int] = None,
    ):
        self.current_path = current_path
        self.file_anchor_map = file_anchor_map
        self.fragment_anchor_map = fragment_anchor_map
        self.image_path_to_recindex = image_path_to_recindex
        self.base_url = urllib.parse.quote(current_path, safe="/") if base_url is None else base_url
        self.current_aliases = current_aliases or (current_path,)
        self.fragment_anchors = (fragment_anchors if fragment_anchors is not None else
                                 {fragment: target for (path, fragment), target in fragment_anchor_map.items()
                                  if path == current_path})
        self.file_anchor = file_anchor if file_anchor is not None else file_anchor_map.get(current_path)
        self.output_limit = output_limit
        self.fed = _HtmlBuffer(output_limit)
        self.styles = styles
        self.emitted_anchors: set[str] = set()
        self.list_layout: dict[ET.Element, tuple[Optional[str], str]] = {}

    @staticmethod
    def _list_label(number: int, kind: str) -> str:
        if kind in {"a", "A"} and number > 0:
            letters = []
            while number:
                number, digit = divmod(number - 1, 26)
                letters.append(chr(ord("a") + digit))
            label = "".join(reversed(letters))
            return label.upper() if kind == "A" else label
        if kind in {"i", "I"} and 0 < number < 4000:
            letters = []
            for value, letter in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
                                  (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                                  (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
                count, number = divmod(number, value)
                letters.append(letter * count)
            label = "".join(letters)
            return label.lower() if kind == "i" else label
        return str(number)

    def _ordered_list_layout(self, body: ET.Element) -> dict[ET.Element, tuple[Optional[str], str]]:
        """Lower custom numbering to visible labels without native list markers."""
        layout: dict[ET.Element, tuple[Optional[str], str]] = {}
        label_bytes = 0
        for ordered in _visible_elements(body):
            if ordered.tag != "ol":
                continue
            items = [child for child in ordered if child.tag == "li"]
            if (not {"start", "reversed", "type"} & ordered.attrib.keys()
                    and not any({"value", "type"} & item.attrib.keys() for item in items)):
                continue
            invalid = False

            def integer(value: str, default: int) -> int:
                nonlocal invalid
                # Bound big-integer work; ordinary book numbering is tiny.
                value = value.strip()
                if re.fullmatch(r"[+-]?[0-9]{1,64}", value):
                    return int(value)
                invalid = True
                return default

            def marker_type(value: str, default: str) -> str:
                nonlocal invalid
                value = value.strip()
                if value in {"1", "a", "A", "i", "I"}:
                    return value
                invalid = True
                return default

            step = -1 if "reversed" in ordered.attrib else 1
            number = len(items) if step == -1 else 1
            if "start" in ordered.attrib:
                number = integer(ordered.attrib["start"], number)
            kind = marker_type(ordered.get("type", "1"), "1")
            layout[ordered] = ("blockquote", "")
            for item in items:
                if "value" in item.attrib:
                    number = integer(item.attrib["value"], number)
                item_kind = marker_type(item.get("type", kind), kind)
                label = self._list_label(number, item_kind) + ". "
                label_bytes += len(label)  # All generated labels are ASCII.
                if label_bytes > self.fed.limit:
                    raise _ResourceLimitError("Generated HTML too large")
                target = item
                # Keep a label on the first paragraph's line, avoiding nested
                # paragraphs and preserving the original nodes and anchors.
                while not (target.text or "").strip() and len(target) and target[0].tag in {"p", "div"}:
                    target = target[0]
                layout[item] = ("div", label if target is item else "")
                if target is not item:
                    layout[target] = (None, label)
                number += step
            if invalid:
                logger.warning("Invalid or oversized ordered-list numbering in %s; ignored invalid attributes", self.current_path)
        return layout

    def _ensure_block_sep(self) -> None:
        if self.fed.parts:
            last = self.fed.parts[-1]
            if last and last[-1] != "\n":
                self.fed.append("\n")

    @staticmethod
    def _derive_alignment(style: str, hint_tokens: set[str]) -> Optional[str]:
        # This is a fallback hint, not margin layout: require the final complete
        # value on each side to be exactly auto, without interpreting lengths.
        margins = {name: value.lower() for name, value in _css_declaration_values(style)
                   if name in {"margin-left", "margin-right"}}
        if margins.get("margin-left") == margins.get("margin-right") == "auto":
            return "center"
        if hint_tokens & _CENTER_HINTS:
            return "center"
        if hint_tokens & _RIGHT_HINTS:
            return "right"
        return None

    def _inject_named_anchors(self, attrs, in_link: bool = False) -> None:
        seen: set[str] = set()
        for key, value in attrs:
            if key not in {"id", "name"} or not value or value in seen:
                continue
            seen.add(value)
            target = self.fragment_anchors.get(value)
            if target and target not in self.emitted_anchors:
                marker = (f'<span id="{target}"></span>' if in_link else
                          f'<a name="{target}" id="{target}"></a>')
                self.fed.append(marker)
                self.emitted_anchors.add(target)

    def _rewrite_href(self, href: str) -> Optional[str]:
        href = href.replace("\t", "").replace("\n", "").replace("\r", "").strip(_URL_TRIM_CHARACTERS)
        try:
            resolved = _resolve_book_href(self.current_path, href, self.base_url)
            if resolved is None:
                external = _split_epub_url(urllib.parse.urljoin(self.base_url, href))
                # A network-path link needs an explicit scheme in the MOBI;
                # otherwise a reader could inherit its local file scheme.
                if not external.scheme and external.netloc:
                    external = external._replace(scheme="https")
                if external.scheme in {"http", "https", "mailto"}:
                    return external.geturl()
                return None
        except ConversionError as e:
            logger.warning("Skipped invalid link in %s: %s: %s", self.current_path, href, e)
            return None

        target_path, fragment = resolved
        if fragment:
            exact = (self.fragment_anchors.get(fragment) if target_path in self.current_aliases else
                     self.fragment_anchor_map.get((target_path, fragment)))
            if exact:
                return f"#{exact}"

        fallback = (self.file_anchor if target_path in self.current_aliases else
                    self.file_anchor_map.get(target_path))
        if fallback:
            return f"#{fallback}"
        return None

    def sanitize(self, body: ET.Element) -> str:
        self.fed = _HtmlBuffer(self.output_limit)
        # The spine's file marker is emitted before this sanitizer runs.
        self.emitted_anchors = {self.file_anchor} if self.file_anchor else set()
        if self.styles is None or body not in self.styles.styles:
            self.styles = _CssStyles(body, [])
        if not _prune_hidden_body(body, self.styles):
            return ""
        _normalize_drop_caps(body, self.styles)
        self.list_layout = self._ordered_list_layout(body)
        self._emit(body)
        return self.fed.text()

    def _emit_text(self, text: str, style: dict[str, str]) -> None:
        text = htmlmod.escape(text, quote=False)
        if self.styles.font_runs:
            if style.get("font-style") == "italic":
                text = f"<i>{text}</i>"
            if style.get("font-weight") == "bold":
                text = f"<b>{text}</b>"
        self.fed.append(text)

    def _emit(self, elem: ET.Element, flatten_table: bool = False, in_link: bool = False) -> None:
        tag = elem.tag
        if _suppressed_element(elem):
            return
        attrs = elem.attrib
        style = self.styles.styles.get(elem, {})
        if tag != "body":
            self._inject_named_anchors(attrs.items(), in_link)
        if tag == "table":
            flatten_table = flatten_table or any(
                (child is not elem and child.tag == "table")
                or any(child.get(key, "1").strip() not in {"", "1"}
                       for key in ("rowspan", "colspan"))
                for child in elem.iter()
            )
        table_tag = tag in {"table", "thead", "tbody", "tfoot", "tr", "td", "th"}
        output_tag = tag if tag in _HTML_ALLOWED and not (flatten_table and table_tag) else None
        hints = _tokenize_hints(attrs.get("class", ""), attrs.get("id", ""))
        if output_tag in {"p", "div"} and _has_heading_hint(elem):
            output_tag = "h2"
        output_tag = {"strong": "b", "em": "i"}.get(output_tag, output_tag)
        if self.styles.font_runs and output_tag in {"b", "i"}:
            output_tag = None
        if (self.styles.font_runs and output_tag in {"th", "h1", "h2", "h3", "h4", "h5", "h6"}
                and any(self.styles.styles.get(child, {}).get("font-weight") == "normal"
                        for child in _visible_elements(elem))):
            # Native MOBI headings/header cells impose bold on descendants.
            # Ordinary blocks/cells allow normal-weight runs to reset it.
            output_tag = "td" if output_tag == "th" else "p"
        replacement, list_label = self.list_layout.get(elem, (None, ""))
        if replacement:
            output_tag = replacement
        if tag == "img":
            src = attrs.get("src", "")
            try:
                resolved = _resolve_book_href(self.current_path, src, self.base_url) if src else None
            except ConversionError:
                resolved = None  # The image loader already reports this omission.
            recindex = self.image_path_to_recindex.get(resolved[0]) if resolved else None
            if recindex is not None:
                self.fed.append(f'<img recindex="{recindex}"/>')
            elif attrs.get("alt", "").strip():
                self._emit_text(attrs["alt"], style)
            return
        if output_tag in {"br", "hr", "mbp:pagebreak"}:
            self.fed.append(f"<{output_tag}/>")
            return
        if output_tag:
            attr_str = ""
            if output_tag == "a" and attrs.get("href"):
                href = self._rewrite_href(attrs["href"])
                if href:
                    attr_str = f' href="{htmlmod.escape(href, quote=True)}"'
            elif output_tag in _HTML_BLOCKS:
                align = style.get("text-align") or self._derive_alignment(attrs.get("style", ""), hints)
                if align:
                    attr_str = f' align="{align}"'
                if output_tag in {"p", "div", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"} and "text-indent" in style:
                    attr_str += f' width="{style["text-indent"]}"'
            if output_tag in _HTML_BLOCKS:
                self._ensure_block_sep()
            self.fed.append(f"<{output_tag}{attr_str}>")
        elif table_tag:
            self.fed.append(" ")
        elif tag in _HTML_FLATTENED_BLOCKS:
            self._ensure_block_sep()
        if list_label:
            self._emit_text(list_label, style)
        if elem.text:
            self._emit_text(elem.text, style)
        for child in elem:
            self._emit(child, flatten_table, in_link or output_tag == "a")
            if child.tail:
                self._emit_text(child.tail, style)
        if output_tag:
            self.fed.append(f"</{output_tag}>")
            if output_tag in _HTML_BLOCKS:
                self.fed.append("\n")
        elif table_tag:
            self.fed.append("\n" if tag in {"tr", "table"} else " ")
        elif tag in _HTML_FLATTENED_BLOCKS:
            self._ensure_block_sep()


@contextmanager
def _atomic_output(output_file: Union[str, Path]) -> Iterator[BinaryIO]:
    """Close a complete temporary file before replacing the destination."""
    # Preserve existing symlink targets and keep the temporary file on the
    # destination's filesystem, for both conversion and USB deployment.
    output_path = Path(output_file).resolve()
    try:
        output_mode = stat.S_IMODE(output_path.stat().st_mode)
    except FileNotFoundError:
        output_mode = None
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=output_path.parent,
                                         prefix=".epub2mobi-", suffix=".tmp", delete=False) as f:
            temporary_path = Path(f.name)
            yield f
            f.flush()
        if output_mode is not None:
            os.chmod(temporary_path, output_mode)
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


class MobiWriter:
    def __init__(self, epub: EpubData):
        self.epub = epub

    @staticmethod
    def _anchor_positions(body_bytes: bytes) -> dict[bytes, int]:
        # IDs inside hyperlinks use a span marker to avoid nested anchors.
        positions: dict[bytes, int] = {}
        for match in re.finditer(rb'<a name="([^"]+)" id="\1"></a>|<span id="([^"]+)"></span>', body_bytes):
            positions.setdefault(match.group(1) or match.group(2), match.start())
        return positions

    def _find_anchor_positions(self, positions: dict[bytes, int]) -> list[int]:
        anchors = [a for a, _ in self.epub.toc_entries]
        anchor_positions: list[int] = []
        for anchor in anchors:
            pos = positions.get(_encode_mobi_text(anchor))
            if pos is None:
                raise ValueError(f"TOC anchor not found in content: {anchor}")
            anchor_positions.append(pos)
        return anchor_positions

    @staticmethod
    def _build_toc_html(entries: tuple[tuple[str, str], ...], file_positions: list[int],
                        depths: tuple[int, ...] = (), *, output_limit: Optional[int] = None) -> str:
        if len(entries) < 2:
            return ""
        if len(entries) != len(file_positions):
            raise ValueError("TOC entry count does not match filepos count")
        depths = depths or (0,) * len(entries)
        if len(depths) != len(entries):
            raise ValueError("TOC depth count does not match entry count")
        _toc_parent_indices(depths)

        parts = _HtmlBuffer(output_limit)
        parts.append("<h1>Table of Contents</h1>")
        current_depth = 0
        for (_, title), filepos, depth in zip(entries, file_positions, depths):
            if filepos < 0 or filepos >= TOC_FILEPOS_MAX:
                raise ValueError(f"TOC filepos out of range: {filepos}")
            safe_title = htmlmod.escape(title, quote=False)
            safe_filepos = f"{filepos:0{TOC_FILEPOS_WIDTH}d}"
            parts.extend(["</blockquote>"] * max(0, current_depth - depth))
            parts.extend(["<blockquote>"] * max(0, depth - current_depth))
            current_depth = depth
            parts.append(f'<p><a filepos="{safe_filepos}">{safe_title}</a></p>')
        parts.extend(["</blockquote>"] * current_depth)
        return parts.text()

    @staticmethod
    def _build_guide_html(toc_filepos: int) -> str:
        if toc_filepos < 0 or toc_filepos >= TOC_FILEPOS_MAX:
            raise ValueError(f"Guide filepos out of range: {toc_filepos}")
        safe_filepos = f"{toc_filepos:0{TOC_FILEPOS_WIDTH}d}"
        return (
            "<guide>"
            f'<reference type="toc" title="Table of Contents" filepos="{safe_filepos}"/>'
            "</guide>"
        )

    @staticmethod
    def _prepare_internal_links(body_bytes: bytes, positions: dict[bytes, int]) -> tuple[bytes, list[bytes]]:
        targets: list[bytes] = []

        def replace(match: re.Match[bytes]) -> bytes:
            anchor = match.group(2)
            if anchor not in positions:
                return match.group(1)
            targets.append(anchor)
            return match.group(1) + b' filepos="??????????"'

        return re.sub(rb'(<a\b[^>]*?) href="#([^"]+)"', replace, body_bytes), targets

    @staticmethod
    def _finish_internal_links(body_bytes: bytes, body_start: int, targets: list[bytes],
                               positions: dict[bytes, int]) -> bytes:
        target_iter = iter(targets)

        def replace(match: re.Match[bytes]) -> bytes:
            filepos = body_start + positions[next(target_iter)]
            if filepos >= TOC_FILEPOS_MAX:
                raise ValueError(f"Internal link filepos out of range: {filepos}")
            return match.group(1) + f' filepos="{filepos:0{TOC_FILEPOS_WIDTH}d}"'.encode("ascii")

        return re.sub(rb'(<a\b[^>]*?) filepos="\?{10}"', replace, body_bytes)

    def _build_text_layout(self) -> TextLayout:
        # The MOBI header declares the encoding. An HTML charset declaration
        # becomes stale when browser readers reserialize the decoded DOM as UTF-8.
        prefix_bytes = b"<html><head></head><body>"
        if len(self.epub.html_content) > MAX_OUTPUT_HTML_BYTES:
            raise _ResourceLimitError("Generated HTML too large")
        body_bytes = _encode_mobi_text(self.epub.html_content)
        if len(body_bytes) > MAX_OUTPUT_HTML_BYTES:
            raise _ResourceLimitError("Generated HTML too large")
        body_bytes, internal_targets = self._prepare_internal_links(body_bytes, self._anchor_positions(body_bytes))
        # Replacements change byte offsets. Scan once more, then share these
        # positions between TOC construction and final fixed-width link values.
        positions = self._anchor_positions(body_bytes)
        guide_bytes = b""
        toc_bytes = b""
        toc_separator = b""
        toc_entry_positions: tuple[int, ...] = ()
        body_start = len(prefix_bytes)
        if len(self.epub.toc_entries) >= 2:
            # Fixed-width guide digits keep the body offset independent of
            # the TOC's destination at the end of the book.
            body_start += len(_encode_mobi_text(self._build_guide_html(0)))
        final_positions = [body_start + pos for pos in self._find_anchor_positions(positions)]
        toc_entry_positions = tuple(final_positions)
        if len(self.epub.toc_entries) >= 2:
            if not body_bytes.rstrip().endswith(b"<mbp:pagebreak/>"):
                toc_separator = b"<mbp:pagebreak/>"
            remaining = MAX_OUTPUT_HTML_BYTES - body_start - len(body_bytes) - len(toc_separator) - len(b"</body></html>")
            toc_bytes = _encode_mobi_text(self._build_toc_html(self.epub.toc_entries, final_positions,
                                                              self.epub.toc_depths, output_limit=remaining))
            toc_filepos = body_start + len(body_bytes) + len(toc_separator)
            guide_bytes = _encode_mobi_text(self._build_guide_html(toc_filepos))

        if body_start + len(body_bytes) + len(toc_separator) + len(toc_bytes) + len(b"</body></html>") > MAX_OUTPUT_HTML_BYTES:
            raise _ResourceLimitError("Generated HTML too large")
        body_bytes = self._finish_internal_links(body_bytes, body_start, internal_targets, positions)
        reading_start_offset = None
        if self.epub.reading_start_anchor is not None:
            position = positions.get(_encode_mobi_text(self.epub.reading_start_anchor))
            if position is None:
                logger.warning("Authored reading start was not retained in MOBI HTML; opening hint omitted")
            else:
                reading_start_offset = body_start + position
        return TextLayout(
            text_bytes=prefix_bytes + guide_bytes + body_bytes + toc_separator + toc_bytes + b"</body></html>",
            toc_entry_positions=toc_entry_positions,
            body_end=body_start + len(body_bytes),
            reading_start_offset=reading_start_offset,
        )

    @staticmethod
    def _build_tagx(nested: bool = False) -> bytes:
        tags = (
            (1, 1, 0x01, 0),
            (2, 1, 0x02, 0),
            (3, 1, 0x04, 0),
            (4, 1, 0x08, 0),
        )
        if nested:
            # Standard NCX parent/first-child/last-child relationships. Bit
            # 0x10 is reserved for the class tag used by periodical indexes.
            tags += ((21, 1, 0x20, 0), (22, 1, 0x40, 0), (23, 1, 0x80, 0))
        tags += ((0, 0, 0x00, 1),)
        tagx = bytearray()
        tagx.extend(b"TAGX")
        tagx.extend(struct.pack(">I", 12 + (len(tags) * 4)))
        tagx.extend(struct.pack(">I", 1))
        for tag, values, mask, end_flag in tags:
            tagx.extend(bytes((tag, values, mask, end_flag)))
        return bytes(tagx)

    @staticmethod
    def _build_indx_header(
        *,
        indx_type: int,
        idxt_offset: int,
        num_records: int,
        encoding: int,
        total_entries: int,
        num_cncx: int,
        tagx_offset: int = 0,
        unk1: int = 0,
    ) -> bytes:
        header = bytearray(INDX_HEADER_LEN)
        header[0:4] = b"INDX"
        struct.pack_into(">I", header, 4, INDX_HEADER_LEN)
        struct.pack_into(">I", header, 12, unk1)
        struct.pack_into(">I", header, 16, indx_type)
        struct.pack_into(">I", header, 20, idxt_offset)
        struct.pack_into(">I", header, 24, num_records)
        struct.pack_into(">I", header, 28, encoding)
        struct.pack_into(">I", header, 32, INDX_INVALID)
        struct.pack_into(">I", header, 36, total_entries)
        struct.pack_into(">I", header, 52, num_cncx)
        struct.pack_into(">I", header, 180, tagx_offset)
        return bytes(header)

    @staticmethod
    def _build_idxt(offsets: list[int]) -> bytes:
        table = bytearray()
        table.extend(b"IDXT")
        for offset in offsets:
            if offset > 0xFFFF:
                raise _NavigationSizeError(f"IDXT offset out of range: {offset}")
            table.extend(struct.pack(">H", offset))
        pad = len(table) % 4
        if pad:
            table.extend(b"\x00" * (4 - pad))
        return bytes(table)

    def _build_navigation_records(self, layout: TextLayout) -> list[bytes]:
        if not self.epub.toc_entries:
            return []
        if len(self.epub.toc_entries) > 0xFFFF:
            raise _NavigationSizeError("Logical TOC entry count exceeds single-record limit")
        depths = self.epub.toc_depths or (0,) * len(self.epub.toc_entries)
        if len(depths) != len(self.epub.toc_entries):
            raise ValueError("TOC depth count does not match entry count")
        parents = _toc_parent_indices(depths)
        entry_positions = list(layout.toc_entry_positions)
        if any(parent is not None and entry_positions[index] < entry_positions[parent]
               for index, parent in enumerate(parents)):
            logger.warning("Logical TOC hierarchy flattened because a child precedes its parent in the text; "
                           "authored destinations and the nested in-book TOC are retained")
            depths = (0,) * len(depths)
            parents = [None] * len(parents)
        child_ranges: dict[int, tuple[int, int]] = {}
        for index, parent in enumerate(parents):
            if parent is not None:
                first_child = child_ranges.get(parent, (index, index))[0]
                child_ranges[parent] = (first_child, index)

        label_record = bytearray()
        label_offsets: list[int] = []
        for _, title in self.epub.toc_entries:
            label_offsets.append(len(label_record))
            label_bytes = _encode_index_text(title)
            label_record.extend(_encode_vwi(len(label_bytes)))
            label_record.extend(label_bytes)

        if len(label_record) > 0xFFFF:
            raise _NavigationSizeError("CNCX label record exceeds single-record limit")

        entry_offsets: list[int] = []
        entries_blob = bytearray()
        # Auxiliary spine items can put TOC order out of physical text order.
        positions_in_text_order = sorted(set(entry_positions))
        end_by_position = {
            filepos: positions_in_text_order[index + 1]
            if index + 1 < len(positions_in_text_order) else layout.body_end
            for index, filepos in enumerate(positions_in_text_order)
        }
        entry_ends = [end_by_position[filepos] for filepos in entry_positions]
        # A parent spans its descendants too. Flat entries still use the next
        # physical destination as their boundary, including out-of-order TOCs.
        for index in range(len(parents) - 1, -1, -1):
            parent = parents[index]
            if parent is not None:
                entry_ends[parent] = max(entry_ends[parent], entry_ends[index])
        for index, ((_, _title), filepos, label_offset) in enumerate(
            zip(self.epub.toc_entries, entry_positions, label_offsets)
        ):
            entry_offsets.append(INDX_HEADER_LEN + len(entries_blob))
            name = f"{index:03d}".encode("ascii")
            length = max(1, entry_ends[index] - filepos)

            entries_blob.append(len(name))
            entries_blob.extend(name)
            control = 0x0F
            if parents[index] is not None:
                control |= 0x20
            if index in child_ranges:
                control |= 0xC0
            entries_blob.append(control)
            entries_blob.extend(_encode_vwi(filepos))
            entries_blob.extend(_encode_vwi(length))
            entries_blob.extend(_encode_vwi(label_offset))
            entries_blob.extend(_encode_vwi(depths[index]))
            if parents[index] is not None:
                entries_blob.extend(_encode_vwi(parents[index]))
            if index in child_ranges:
                first_child, last_child = child_ranges[index]
                entries_blob.extend(_encode_vwi(first_child))
                entries_blob.extend(_encode_vwi(last_child))

        secondary_idxt = self._build_idxt(entry_offsets)
        secondary_header = self._build_indx_header(
            indx_type=INDX_TYPE_NORMAL,
            idxt_offset=INDX_HEADER_LEN + len(entries_blob),
            num_records=len(self.epub.toc_entries),
            encoding=INDX_INVALID,
            total_entries=0,
            num_cncx=0,
            unk1=1,
        )
        secondary_record = secondary_header + bytes(entries_blob) + secondary_idxt
        if len(secondary_record) > 0x10000:
            raise _NavigationSizeError("INDX navigation record exceeds single-record limit")

        tagx = self._build_tagx(nested=any(depths))
        last_name = f"{len(self.epub.toc_entries) - 1:03d}".encode("ascii")
        main_dummy = bytes((len(last_name),)) + last_name
        main_dummy += struct.pack(">H", len(self.epub.toc_entries))
        main_dummy += b"\x00" * (-(INDX_HEADER_LEN + len(tagx) + len(main_dummy)) % 4)
        main_idxt = self._build_idxt([INDX_HEADER_LEN + len(tagx)])
        main_header = self._build_indx_header(
            indx_type=INDX_TYPE_INFLECTION,
            idxt_offset=INDX_HEADER_LEN + len(tagx) + len(main_dummy),
            num_records=1,
            encoding=INDX_LABEL_ENCODING,
            total_entries=len(self.epub.toc_entries),
            num_cncx=1,
            tagx_offset=INDX_HEADER_LEN,
        )
        main_record = main_header + tagx + main_dummy + main_idxt

        return [main_record, secondary_record, bytes(label_record)]

    @staticmethod
    def _best_backref(data: bytes, pos: int) -> tuple[int, int]:
        window_start = max(0, pos - 2047)
        max_len = min(10, len(data) - pos)
        if max_len < 3:
            return 0, 0

        for length in range(max_len, 2, -1):
            match = data[pos : pos + length]
            hit = data.rfind(match, window_start, pos)
            if hit != -1:
                distance = pos - hit
                if 1 <= distance <= 2047:
                    return distance, length
        return 0, 0

    @staticmethod
    def _compress_palmdoc(data: bytes) -> bytes:
        if not data:
            return b""

        out = bytearray()
        i = 0
        n = len(data)
        while i < n:
            distance, length = MobiWriter._best_backref(data, i)
            if length >= 3:
                code = (distance << 3) | (length - 3)
                out.append(0x80 | ((code >> 8) & 0x3F))
                out.append(code & 0xFF)
                i += length
                continue

            if i + 1 < n and data[i] == 0x20 and 0x40 <= data[i + 1] <= 0x7F:
                out.append(data[i + 1] | 0x80)
                i += 2
                continue

            b = data[i]
            if b == 0x00 or 0x09 <= b <= 0x7F:
                out.append(b)
                i += 1
                continue

            run = bytearray()
            while i < n and len(run) < 8:
                b = data[i]
                if b == 0x00 or 0x09 <= b <= 0x7F:
                    break
                # First byte already proved non-backref by the outer loop check.
                if run and MobiWriter._best_backref(data, i)[1] >= 3:
                    break
                run.append(b)
                i += 1

            out.append(len(run))
            out.extend(run)

        return bytes(out)

    @staticmethod
    def _safe_chunk_bytes(b: bytes, limit: int) -> list[bytes]:
        # Safe for CP1252 (single byte encoding)
        if not b:
            raise ValueError("EPUB produced empty HTML payload; refusing to emit empty MOBI text record")
        return [b[i:i + limit] for i in range(0, len(b), limit)]

    @staticmethod
    def _validate_record_layout(text_rec_count: int, total_records: int) -> None:
        if text_rec_count > 0xFFFF:
            raise ConversionError(f"Too many PalmDOC text records: {text_rec_count}")
        if total_records > 0xFFFF:
            raise ConversionError(f"Too many records for PDB: {total_records}")

    @staticmethod
    def _build_flis() -> bytes:
        flis = bytearray(36)
        flis[0:4] = b"FLIS"
        struct.pack_into(">I", flis, 4, 8)
        struct.pack_into(">H", flis, 8, 65)
        struct.pack_into(">H", flis, 10, 0)
        struct.pack_into(">I", flis, 12, 0)
        struct.pack_into(">I", flis, 16, 0xFFFFFFFF)
        struct.pack_into(">H", flis, 20, 1)
        struct.pack_into(">H", flis, 22, 3)
        struct.pack_into(">I", flis, 24, 3)
        struct.pack_into(">I", flis, 28, 1)
        struct.pack_into(">I", flis, 32, 0xFFFFFFFF)
        return bytes(flis)

    @staticmethod
    def _build_fcis(text_length: int) -> bytes:
        fcis = bytearray(44)
        fcis[0:4] = b"FCIS"
        struct.pack_into(">I", fcis, 4, 20)
        struct.pack_into(">I", fcis, 8, 16)
        struct.pack_into(">I", fcis, 12, 1)
        struct.pack_into(">I", fcis, 16, 0)
        struct.pack_into(">I", fcis, 20, text_length)
        struct.pack_into(">I", fcis, 24, 0)
        struct.pack_into(">I", fcis, 28, 32)
        struct.pack_into(">I", fcis, 32, 8)
        struct.pack_into(">H", fcis, 36, 1)
        struct.pack_into(">H", fcis, 38, 1)
        struct.pack_into(">I", fcis, 40, 0)
        return bytes(fcis)

    @staticmethod
    def _build_eof() -> bytes:
        return b"\xE9\x8E\x0D\x0A"

    def _build_exth(self, *, reading_start_offset: Optional[int] = None) -> bytes:
        payload = bytearray()
        count = 0

        def add(rt: int, data: bytes) -> None:
            nonlocal count
            if not data: return
            payload.extend(struct.pack(">II", rt, len(data) + 8))
            payload.extend(data)
            count += 1

        add(EXTH_AUTHOR, _encode_meta(self.epub.author))
        add(EXTH_TITLE, _encode_meta(self.epub.title))
        add(EXTH_SOURCE, _encode_meta(self.epub.uuid))
        add(EXTH_CDETYPE, b"EBOK")
        if reading_start_offset is not None:
            if not 0 <= reading_start_offset < 2**32:
                raise ValueError("Reading start offset is outside the MOBI text range")
            add(EXTH_START_READING, struct.pack(">I", reading_start_offset))

        asin = f"B{_crc32_u32(self.epub.uuid):08X}".encode("ascii")
        add(EXTH_ASIN, asin)
        if self.epub.cover_index is not None:
            if not 0 <= self.epub.cover_index < len(self.epub.image_records):
                raise ValueError("Cover index is outside the MOBI image records")
            add(EXTH_COVER_OFFSET, struct.pack(">I", self.epub.cover_index))

        exth_len = 12 + len(payload)
        exth = bytearray()
        exth.extend(EXTH_MAGIC)
        exth.extend(struct.pack(">I", exth_len))
        exth.extend(struct.pack(">I", count))
        exth.extend(payload)

        pad = len(exth) % 4
        if pad:
            exth.extend(b"\x00" * (4 - pad))
        return bytes(exth)

    @staticmethod
    def _compute_record_indices(text_rec_count: int, following_rec_count: int) -> tuple[int, int]:
        flis_idx = 1 + text_rec_count + following_rec_count
        return flis_idx, flis_idx + 1

    def _build_record0(
        self,
        uncompressed_text_len: int,
        text_rec_count: int,
        flis_idx: int,
        fcis_idx: int,
        first_nonbook: int,
        nav_index_idx: Optional[int],
        first_image_idx: Optional[int],
        reading_start_offset: Optional[int] = None,
    ) -> bytes:
        # PalmDOC
        palmdoc = bytearray(PALMDOC_LEN)
        struct.pack_into(">H", palmdoc, 0, PALMDOC_COMPRESSION)
        struct.pack_into(">H", palmdoc, 2, 0)
        struct.pack_into(">I", palmdoc, 4, uncompressed_text_len)
        struct.pack_into(">H", palmdoc, 8, text_rec_count)
        struct.pack_into(">H", palmdoc, 10, TEXT_RECORD_MAX)
        struct.pack_into(">H", palmdoc, 12, 0)
        struct.pack_into(">H", palmdoc, 14, 0)

        # MOBI Header
        mobi = bytearray(MOBI_HEADER_LEN)
        mobi[0:4] = MOBI_MAGIC
        struct.pack_into(">I", mobi, OFF_LENGTH, MOBI_HEADER_LEN)
        struct.pack_into(">I", mobi, OFF_TYPE, 2)       # Book
        struct.pack_into(">I", mobi, OFF_ENCODING, MOBI_TEXT_ENCODING_ID) # 1252
        struct.pack_into(">I", mobi, OFF_UID, _crc32_u32(self.epub.uuid))
        struct.pack_into(">I", mobi, OFF_VERSION, 6)
        struct.pack_into(">I", mobi, OFF_MIN_VER, 6)
        struct.pack_into(">I", mobi, OFF_LOCALE, _mobi_locale(self.epub.language))

        # Initialize absent pointers
        for off in (
            OFF_ORTHO_INDEX, OFF_INFLECT_INDEX, OFF_INDEX_NAMES, OFF_INDEX_KEYS,
            OFF_EXTRA_INDEX_0, OFF_EXTRA_INDEX_1, OFF_EXTRA_INDEX_2,
            OFF_EXTRA_INDEX_3, OFF_EXTRA_INDEX_4, OFF_EXTRA_INDEX_5,
            OFF_UNKNOWN_A4, OFF_DRM_OFFSET, OFF_DRM_COUNT,
        ):
            struct.pack_into(">I", mobi, off, 0xFFFFFFFF)

        # Content Range
        struct.pack_into(">H", mobi, OFF_FIRST_CONTENT, 1)
        # Navigation and image records are content; FLIS starts the trailing metadata.
        struct.pack_into(">H", mobi, OFF_LAST_CONTENT, flis_idx - 1)
        struct.pack_into(">I", mobi, OFF_UNKNOWN_C4, 1)
        struct.pack_into(">I", mobi, OFF_EXTRA_RECORD_DATA_FLAGS, 0)
        struct.pack_into(">I", mobi, OFF_INDX, nav_index_idx if nav_index_idx is not None else INDX_INVALID)

        # EXTH Flag
        flags = struct.unpack_from(">I", mobi, OFF_EXTH_FLAGS)[0]
        struct.pack_into(">I", mobi, OFF_EXTH_FLAGS, flags | 0x40)

        # Assemble Record 0
        exth = self._build_exth(reading_start_offset=reading_start_offset)
        record0 = bytearray()
        record0.extend(palmdoc)
        record0.extend(mobi)
        record0.extend(exth)

        # Full Name (ABSOLUTE OFFSET in Record 0)
        full_name_off = len(record0)
        full_name = _encode_meta(self.epub.title)
        record0.extend(full_name)
        record0.extend(b"\x00\x00")

        # Write absolute offset from start of record0
        struct.pack_into(">I", record0, PALMDOC_LEN + OFF_FULLNAME_O, full_name_off)
        struct.pack_into(">I", record0, PALMDOC_LEN + OFF_FULLNAME_L, len(full_name))

        pad = len(record0) % 4
        if pad:
            record0.extend(b"\x00" * (4 - pad))

        struct.pack_into(">I", record0, PALMDOC_LEN + OFF_FIRST_NONBOOK, first_nonbook)
        struct.pack_into(
            ">I",
            record0,
            PALMDOC_LEN + OFF_FIRST_IMAGE,
            first_image_idx if first_image_idx is not None else INDX_INVALID,
        )

        struct.pack_into(">I", record0, PALMDOC_LEN + OFF_FLIS_REC, flis_idx)
        struct.pack_into(">I", record0, PALMDOC_LEN + OFF_FLIS_CNT, 1)
        struct.pack_into(">I", record0, PALMDOC_LEN + OFF_FCIS_REC, fcis_idx)
        struct.pack_into(">I", record0, PALMDOC_LEN + OFF_FCIS_CNT, 1)

        # Tail Fields
        struct.pack_into(">Q", record0, PALMDOC_LEN + OFF_TAIL_RESERVED_8, 0)
        struct.pack_into(">I", record0, PALMDOC_LEN + OFF_TAIL_E0, 0xFFFFFFFF)
        struct.pack_into(">I", record0, PALMDOC_LEN + OFF_TAIL_E4, 0)
        struct.pack_into(">I", record0, PALMDOC_LEN + OFF_TAIL_E8, 0xFFFFFFFF)
        struct.pack_into(">I", record0, PALMDOC_LEN + OFF_TAIL_EC, 0xFFFFFFFF)

        return bytes(record0)

    def _build_pdb_header_and_index(self, records: list[bytes]) -> tuple[bytes, bytes]:
        t = _palm_time_now()
        pdb = bytearray(PDB_HEADER_LEN)

        name_ascii = self.epub.title[:31].encode("ascii", "replace")
        pdb[0:len(name_ascii)] = name_ascii

        struct.pack_into(">I", pdb, 36, t)
        struct.pack_into(">I", pdb, 40, t)

        pdb[60:64] = b"BOOK"
        pdb[64:68] = b"MOBI"

        n = len(records)
        struct.pack_into(">I", pdb, 68, n + 1)
        struct.pack_into(">H", pdb, 76, n)

        offset_base = PDB_HEADER_LEN + (n * PDB_RECORD_INFO_LEN) + PDB_GAP_LEN
        rec_info = bytearray()
        curr = offset_base
        uid = 1
        for rec in records:
            if not rec:
                raise ConversionError("Cannot write an empty MOBI record")
            rec_info.extend(struct.pack(">I", curr))
            rec_info.append(0x00)
            rec_info.extend(struct.pack(">I", uid)[1:])
            curr += len(rec)
            uid += 1

        return bytes(pdb), bytes(rec_info)

    def build(self, output_file: str) -> None:
        for field_name, value in (("Title", self.epub.title), ("Author", self.epub.author)):
            try:
                value.encode(MOBI_TEXT_ENCODING_PY)
            except UnicodeEncodeError:
                logger.warning(
                    "%s contains characters outside %s; MOBI metadata will replace them with '?'",
                    field_name,
                    MOBI_TEXT_ENCODING_NAME,
                )

        layout = self._build_text_layout()
        try:
            nav_records = self._build_navigation_records(layout)
        except _NavigationSizeError as e:
            retained = "the inline TOC is retained" if len(self.epub.toc_entries) >= 2 else "reading content is retained"
            logger.warning("Logical TOC omitted: %s; %s", e, retained)
            nav_records = []
        text_bytes = layout.text_bytes
        image_records = list(self.epub.image_records)
        uncompressed_records = self._safe_chunk_bytes(text_bytes, TEXT_RECORD_MAX)
        text_records = [self._compress_palmdoc(rec) for rec in uncompressed_records]
        self._validate_record_layout(
            text_rec_count=len(text_records),
            total_records=1 + len(text_records) + len(nav_records) + len(image_records) + 3,
        )

        flis_idx, fcis_idx = self._compute_record_indices(
            len(text_records),
            len(nav_records) + len(image_records),
        )
        nav_index_idx = 1 + len(text_records) if nav_records else None
        first_image_idx = 1 + len(text_records) + len(nav_records) if image_records else None
        first_nonbook = 1 + len(text_records)
        record0 = self._build_record0(
            uncompressed_text_len=len(text_bytes),
            text_rec_count=len(text_records),
            flis_idx=flis_idx,
            fcis_idx=fcis_idx,
            first_nonbook=first_nonbook,
            nav_index_idx=nav_index_idx,
            first_image_idx=first_image_idx,
            reading_start_offset=layout.reading_start_offset,
        )

        records: list[bytes] = [record0]
        records.extend(text_records)
        records.extend(nav_records)
        records.extend(image_records)
        records.extend([self._build_flis(), self._build_fcis(len(text_bytes)), self._build_eof()])

        pdb_header, rec_info = self._build_pdb_header_and_index(records)

        with _atomic_output(output_file) as f:
            f.write(pdb_header)
            f.write(rec_info)
            f.write(b"\x00\x00")
            for rec in records:
                f.write(rec)

        logger.info("SUCCESS: Created %s", output_file)


def deploy_to_kindle(source_file: str) -> None:
    candidates: list[str] = []
    if sys.platform == "darwin":
        vol = "/Volumes"
        if os.path.isdir(vol):
            candidates = [os.path.join(vol, d) for d in os.listdir(vol) if os.path.isdir(os.path.join(vol, d))]
    elif sys.platform.startswith("linux"):
        user = os.environ.get("USER", "root")
        for base in (f"/media/{user}", f"/run/media/{user}", "/media"):
            if os.path.isdir(base):
                candidates.extend([os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
    elif sys.platform == "win32":
        import string
        from ctypes import windll
        bitmask = windll.kernel32.GetLogicalDrives()
        for letter in string.ascii_uppercase:
            if bitmask & 1:
                candidates.append(letter + ":\\")
            bitmask >>= 1

    matches: dict[str, str] = {}
    for path in candidates:
        docs = os.path.join(path, "documents")
        if not os.path.isdir(docs):
            continue
        if "Kindle" not in os.path.basename(path) and not os.path.exists(os.path.join(path, "system")):
            continue
        # Discovery may visit the same mount more than once or through an alias.
        matches.setdefault(os.path.normcase(os.path.realpath(path)), path)

    if not matches:
        logger.warning("No Kindle detected.")
        return
    if len(matches) > 1:
        raise ConversionError(
            f"Multiple Kindles detected ({', '.join(sorted(matches.values()))}); "
            "disconnect extra devices or copy the local MOBI manually"
        )

    path = next(iter(matches.values()))
    dest = os.path.join(path, "documents", os.path.basename(source_file))
    # Actual copy failures propagate to main(), which returns a failure
    # status. The completed local conversion remains available for retry.
    with _atomic_output(dest) as destination:
        with open(source_file, "rb") as source:
            shutil.copyfileobj(source, destination)
    logger.info("Copied to Kindle: %s", dest)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert EPUB files to legacy MOBI6.")
    parser.add_argument("input_epub", help="Path to the input EPUB file")
    parser.add_argument(
        "-o",
        "--output",
        dest="output_mobi",
        help="Path to the output MOBI file (defaults to input path with .mobi suffix)",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Copy the generated MOBI to a connected Kindle if one is detected",
    )
    parser.add_argument(
        "--report-omissions",
        action="store_true",
        help="List resources that were omitted from the MOBI output",
    )
    return parser


def _log_media_omissions(omissions: tuple[MediaOmission, ...], detailed: bool) -> None:
    if not omissions:
        if detailed:
            logger.info("No omissions detected by the resource scan; CSS background assets were not inspected.")
        return

    logger.warning(
        "%d resource omission%s detected%s",
        len(omissions),
        "" if len(omissions) == 1 else "s",
        ":" if detailed else " (use --report-omissions for details).",
    )
    if detailed:
        for omission in omissions:
            logger.warning(
                "  %s — %s (referenced in %s)",
                omission.resource, omission.reason, omission.source,
            )


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    infile = Path(args.input_epub)
    outfile = Path(args.output_mobi) if args.output_mobi else infile.with_suffix(".mobi")

    try:
        if outfile.exists() and infile.samefile(outfile):
            raise ConversionError("Output path resolves to the input EPUB; choose a different output path")
        epub_data = parse_epub(infile)
        MobiWriter(epub_data).build(str(outfile))
        _log_media_omissions(epub_data.omitted_media, args.report_omissions)

        if args.deploy:
            deploy_to_kindle(str(outfile))

        return 0
    except (ConversionError, OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as e:
        logger.error("Error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
