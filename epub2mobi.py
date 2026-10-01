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
# ElementTree is used to keep the converter dependency-free.
# Untrusted XML is size-limited and pre-screened for unsafe declarations in
# _parse_xml(). Applications requiring a hardened external parser may use
# defusedxml.ElementTree instead.
import xml.etree.ElementTree as ET
from xml.parsers import expat

from dataclasses import dataclass
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
HTML_META_CHARSET = "windows-1252"

# TOC filepos field width
TOC_FILEPOS_WIDTH = 10
TOC_FILEPOS_MAX = 10 ** TOC_FILEPOS_WIDTH

# XML parsing guardrails for untrusted EPUBs
MAX_XML_BYTES = 8 * 1024 * 1024
MAX_XHTML_BYTES = 16 * 1024 * 1024
MAX_CSS_BYTES = 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_RESOURCE_BYTES = 256 * 1024 * 1024

# EXTH Types
EXTH_AUTHOR = 100
EXTH_TITLE = 503
EXTH_SOURCE = 112
EXTH_ASIN = 113
EXTH_CDETYPE = 501  # EBOK/PDOC
EXTH_COVER_OFFSET = 201

DC_NAMESPACE = "{http://purl.org/dc/elements/1.1/}"
OPF_NAMESPACE = "{http://www.idpf.org/2007/opf}"

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


@dataclass(frozen=True)
class SpineItem:
    index: int
    href: str
    full_path: str
    aliases: tuple[str, ...]
    base_url: str
    anchor: str
    stem: str
    document: ET.Element
    body: ET.Element
    body_fragments: tuple[str, ...]
    fragment_anchors: dict[str, str]
    linear: bool


@dataclass(frozen=True)
class TextLayout:
    text_bytes: bytes
    toc_filepos: Optional[int]
    toc_entry_positions: tuple[int, ...]


@dataclass(frozen=True)
class TocTarget:
    path: str
    fragment: Optional[str]
    label: str


class _NavigationSizeError(ValueError):
    """The logical TOC cannot fit the supported single-record layout."""


def _palm_time_now() -> int:
    return int((datetime.now() - datetime(1904, 1, 1)).total_seconds())


def _crc32_u32(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def _encode_mobi_text(s: str) -> bytes:
    """Encode as CP1252. Use XML entities for characters that don't fit."""
    return s.encode(MOBI_TEXT_ENCODING_PY, errors="xmlcharrefreplace")


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
    if len(data) > (MAX_XML_BYTES if size_limit is None else size_limit):
        raise ValueError(f"XML file too large: {source_name}")
    # Expat sees declarations regardless of whether the XML is UTF-8, UTF-16, or UTF-32.
    guard = expat.ParserCreate()

    def reject_entity(*_args):
        raise ValueError(f"Unsafe XML declaration in {source_name}")

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

    def start_element(_name, _attributes):
        nonlocal in_prolog
        in_prolog = False

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
            return ET.fromstring(text[:start] + doctype + text[end:])
        return ET.fromstring(data)
    except (ET.ParseError, expat.ExpatError, LookupError, UnicodeError) as e:
        raise ValueError(f"Malformed XML: {source_name}") from e


def _find_opf(z: zipfile.ZipFile) -> tuple[str, str]:
    try:
        txt = _read_xml_member(z, "META-INF/container.xml")
    except KeyError as e:
        raise ValueError("Invalid EPUB container: missing META-INF/container.xml") from e

    root = _parse_xml(txt, "META-INF/container.xml")

    opf_path = None
    for elem in root.iter():
        if elem.tag.endswith("rootfile"):
            candidate = elem.attrib.get("full-path")
            if candidate:
                opf_path = candidate
                break
    if not opf_path:
        raise ValueError("Invalid EPUB container: no rootfile in META-INF/container.xml")

    normalized_opf = _normalize_epub_path(urllib.parse.unquote(opf_path))
    if normalized_opf in ("", "."):
        raise ValueError("Invalid EPUB container: empty OPF path")
    return normalized_opf, posixpath.dirname(normalized_opf)


def _parse_xhtml(data: bytes, source_name: str) -> ET.Element:
    root = _parse_xml(data, source_name, MAX_XHTML_BYTES, xhtml_entities=True)
    # Normalize XHTML only; foreign vocabularies must not become HTML elements.
    stack = [(root, 0)]
    while stack:
        elem, depth = stack.pop()
        if depth > 256:
            raise ValueError(f"XHTML nesting too deep: {source_name}")
        if elem.tag.startswith("{http://www.w3.org/1999/xhtml}"):
            elem.tag = _strip_ns(elem.tag)
        stack.extend((child, depth + 1) for child in elem)
    return root


def _body_element(root: ET.Element) -> ET.Element:
    if root.tag == "html":
        body = root.find("body")
        if body is None:
            raise ValueError("XHTML document has no body")
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
    blocks = MinimalHtmlSanitizer._BLOCKS | MinimalHtmlSanitizer._FLATTENED_BLOCKS
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
    return posixpath.normpath(path.replace("\\", "/")).lstrip("/")


def _resolve_manifest_path(base_dir: str, href: str) -> str:
    parsed = urllib.parse.urlsplit(href)
    if parsed.scheme or parsed.netloc:
        raise ValueError(f"Invalid EPUB manifest href: {href}")
    return _normalize_epub_path(posixpath.join(base_dir, urllib.parse.unquote(parsed.path)))


def _is_supported_image_media_type(media_type: str) -> bool:
    return media_type.lower() in {"image/jpeg", "image/png", "image/gif"}


def _document_base_url(root: ET.Element, current_path: str) -> str:
    document_url = urllib.parse.quote(current_path, safe="/")
    head = root.find("head") if root.tag == "html" else None
    if head is not None:
        for elem in head.iter("base"):
            if "href" in elem.attrib:
                return urllib.parse.urljoin(document_url, elem.attrib["href"])
    return document_url


def _resolve_book_href(current_path: str, href: str,
                       base_url: Optional[str] = None) -> Optional[Tuple[str, Optional[str]]]:
    document_url = urllib.parse.quote(current_path, safe="/") if base_url is None else base_url
    parsed = urllib.parse.urlsplit(urllib.parse.urljoin(document_url, href))
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
    resolved = _resolve_book_href(current_path, href, base_url)
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


def _extract_nav_toc_targets(
    z: zipfile.ZipFile,
    manifest_hrefs: dict[str, str],
    manifest_media_types: dict[str, str],
    manifest_properties: dict[str, str],
    base_dir: str,
) -> list[TocTarget]:
    nav_href = None
    for item_id, href in manifest_hrefs.items():
        properties = manifest_properties.get(item_id, "")
        media_type = manifest_media_types.get(item_id, "")
        if "nav" in properties.split() and media_type == "application/xhtml+xml":
            nav_href = href
            break
    if not nav_href:
        return []

    nav_path = nav_href
    try:
        nav_path = _resolve_manifest_path(base_dir, nav_href)
        nav_root = _parse_xhtml(_read_xml_member(z, nav_path), nav_path)
    except (KeyError, ValueError):
        logger.warning("Unable to read EPUB3 nav document: %s", nav_path)
        return []

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
        return []

    toc_targets: list[TocTarget] = []
    nav_base_url = _document_base_url(nav_root, nav_path)
    for elem in _visible_elements(toc_nav):
        if elem.tag != "a":
            continue
        raw_href = elem.attrib.get("href")
        if not raw_href:
            continue
        label = " ".join(_visible_text(elem).split())
        if not label:
            continue
        resolved = _resolve_book_href(nav_path, raw_href, nav_base_url)
        if resolved is None:
            continue
        target_path, fragment = resolved
        toc_targets.append(TocTarget(path=target_path, fragment=fragment, label=label))
    return toc_targets


def _build_ncx_label_map(
    z: zipfile.ZipFile,
    spine_node: ET.Element,
    manifest_hrefs: dict[str, str],
    manifest_media_types: dict[str, str],
    base_dir: str,
) -> list[TocTarget]:
    ncx_href = None
    toc_id = spine_node.attrib.get("toc")
    if toc_id:
        ncx_href = manifest_hrefs.get(toc_id)
    if not ncx_href:
        for item_id, href in manifest_hrefs.items():
            media_type = manifest_media_types.get(item_id, "")
            if media_type == "application/x-dtbncx+xml" or href.lower().endswith(".ncx"):
                ncx_href = href
                break
    if not ncx_href:
        return []

    ncx_path = ncx_href
    try:
        ncx_path = _resolve_manifest_path(base_dir, ncx_href)
        ncx_root = _parse_xml(_read_xml_member(z, ncx_path), ncx_path)
    except (KeyError, ValueError):
        logger.warning("Unable to read NCX for TOC labels: %s", ncx_path)
        return []

    targets: list[TocTarget] = []
    for nav_point in ncx_root.iter():
        if not nav_point.tag.endswith("navPoint"):
            continue

        src = None
        label = None
        for elem in nav_point.iter():
            if label is None and elem.tag.endswith("text") and elem.text:
                candidate = " ".join(elem.text.split())
                if candidate:
                    label = candidate
            if src is None and elem.tag.endswith("content"):
                raw_src = elem.attrib.get("src")
                if raw_src:
                    src = raw_src

        if not src or not label:
            continue

        resolved = _resolve_book_href(ncx_path, src)
        if resolved is None:
            continue
        target_path, fragment = resolved
        targets.append(TocTarget(path=target_path, fragment=fragment, label=label))

    return targets


def _read_zip_member(
    z: zipfile.ZipFile,
    member_path: str,
    *,
    size_limit: int,
    aggregate_budget: int,
    aggregate_used: int,
    kind: str,
) -> tuple[bytes, int]:
    try:
        info = z.getinfo(member_path)
    except KeyError as e:
        raise KeyError(member_path) from e

    if info.file_size > size_limit:
        raise ValueError(
            f"{kind} too large: {member_path} ({info.file_size} bytes > {size_limit} bytes)"
        )

    new_total = aggregate_used + info.file_size
    if new_total > aggregate_budget:
        raise ValueError(
            f"EPUB content too large: extracting {member_path} would exceed {aggregate_budget} bytes"
        )

    with z.open(member_path) as member:
        data = member.read(size_limit + 1)
    if len(data) > size_limit:
        raise ValueError(f"{kind} too large: {member_path}")
    actual_total = aggregate_used + len(data)
    if actual_total > aggregate_budget:
        raise ValueError(
            f"EPUB content too large: extracting {member_path} would exceed {aggregate_budget} bytes"
        )
    return data, actual_total


def _read_xml_member(z: zipfile.ZipFile, member_path: str) -> bytes:
    data, _ = _read_zip_member(
        z,
        member_path,
        size_limit=MAX_XML_BYTES,
        aggregate_budget=MAX_TOTAL_RESOURCE_BYTES,
        aggregate_used=0,
        kind="XML file",
    )
    return data


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
    for part, delimiter in _css_parts(text, "{};"):
        if depth == 0:
            if delimiter == "{":
                prelude, body, nested = part.strip(), [], False
                depth = 1
        else:
            if delimiter == "{":
                nested = True
                depth += 1
            elif delimiter == "}":
                depth -= 1
                if depth == 0 and not nested and not prelude.startswith("@"):
                    declarations = _css_declarations("".join(body) + part)
                    selectors = [selector.strip() for selector in prelude.split(",")]
                    # A selector list containing unsupported syntax is skipped whole.
                    if declarations and all(re.fullmatch(r"\.?(?:[A-Za-z_]|-[A-Za-z_-])[A-Za-z0-9_-]*", s) for s in selectors):
                        rules.extend((s if s.startswith(".") else s.lower(), declarations) for s in selectors)
            if depth == 1 and not nested:
                body.append(part + delimiter)
    return rules


class _CssStyles:
    """Resolve supported text properties and a non-inherited drop-cap float hint."""

    def __init__(self, document: ET.Element, rules: list[tuple[str, dict[str, str]]]):
        by_selector: dict[str, dict[str, tuple[int, str]]] = {}
        for order, (selector, declarations) in enumerate(rules):
            target = by_selector.setdefault(selector, {})
            target.update((name, (order, value)) for name, value in declarations.items())
        self.styles: dict[ET.Element, dict[str, str]] = {}
        self.floats: dict[ET.Element, str] = {}
        self.font_runs = False

        def resolve(elem: ET.Element, parent: dict[str, str], parent_float: str = "none") -> None:
            if _suppressed_element(elem):
                return
            chosen = {name: (0, order, value)
                      for name, (order, value) in by_selector.get(elem.tag, {}).items()}
            for cls in set(elem.get("class", "").split()):
                for name, (order, value) in by_selector.get("." + cls, {}).items():
                    candidate = (1, order, value)
                    if name not in chosen or candidate[:2] > chosen[name][:2]:
                        chosen[name] = candidate
            local = {name: value for name, (_, _, value) in chosen.items()}
            local.update(_css_declarations(elem.get("style", "")))
            floating = local.pop("float", "none")
            if floating == "inherit":
                floating = parent_float
            self.floats[elem] = floating
            self.font_runs |= bool(local.keys() & {"font-style", "font-weight"})
            style = dict(parent)
            # Semantic markup provides local defaults, which authored CSS can reset.
            heading_hint = (elem.tag in {"p", "div"} and
                            MinimalHtmlSanitizer._tokenize_hints(elem.get("class", ""), elem.get("id", ""))
                            & MinimalHtmlSanitizer._HEADING_HINTS)
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
                resolve(child, style, floating)

        resolve(document, {})


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
                    or MinimalHtmlSanitizer._tokenize_hints(paragraph.get("class", ""), paragraph.get("id", ""))
                    & MinimalHtmlSanitizer._HEADING_HINTS
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


def _declared_cover_item(metadata: list[ET.Element], manifest_properties: dict[str, str]) -> Optional[str]:
    for item_id, properties in manifest_properties.items():
        if "cover-image" in properties.split():
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


def parse_epub(filepath: Union[str, Path]) -> EpubData:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(str(filepath))

    book_title = "Unknown"
    book_author = "Unknown"
    book_uuid = "000000000000"
    book_language = None

    with zipfile.ZipFile(filepath, "r") as z:
        opf_path, base_dir = _find_opf(z)
        try:
            opf_xml = _read_xml_member(z, opf_path)
        except KeyError as e:
            raise ValueError(f"OPF declared in container is missing: {opf_path}") from e
        opf_root = _parse_xml(opf_xml, opf_path)

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
        for child in list(opf_root):
            t = _strip_ns(child.tag)
            if t == "manifest":
                manifest_node = child
            elif t == "spine":
                spine_node = child
        if manifest_node is None or spine_node is None:
            raise ValueError("Malformed OPF: missing manifest or spine")

        manifest: dict[str, str] = {}
        manifest_media_types: dict[str, str] = {}
        manifest_properties: dict[str, str] = {}
        manifest_fallbacks: dict[str, str] = {}
        for item in list(manifest_node):
            if _strip_ns(item.tag) == "item":
                iid = item.attrib.get("id")
                href = item.attrib.get("href")
                if iid and href:
                    manifest[iid] = href
                    manifest_media_types[iid] = item.attrib.get("media-type", "")
                    manifest_properties[iid] = item.attrib.get("properties", "")
                    if item.attrib.get("fallback"):
                        manifest_fallbacks[iid] = item.attrib["fallback"]
        manifest_paths = {
            _resolve_manifest_path(base_dir, href): iid
            for iid, href in manifest.items()
            if _resolve_book_href(opf_path, href) is not None
        }

        omissions: list[MediaOmission] = []
        seen_omissions: set[MediaOmission] = set()

        def record_omission(resource: str, source: str, reason: str) -> None:
            omission = MediaOmission(resource=resource, source=source, reason=reason)
            if omission not in seen_omissions:
                omissions.append(omission)
                seen_omissions.add(omission)

        font_media_types = {
            "application/font-sfnt", "application/vnd.ms-opentype",
            "application/x-font-ttf", "application/x-font-otf",
        }
        for item_id, href in manifest.items():
            media_type = manifest_media_types[item_id].lower()
            resource_path = urllib.parse.urlsplit(href).path.lower()
            is_font = (
                media_type.startswith("font/")
                or media_type in font_media_types
                or resource_path.endswith((".ttf", ".otf", ".woff", ".woff2"))
            )
            if _resolve_book_href(opf_path, href) is None:
                record_omission(
                    _reported_media_path(opf_path, href, href), opf_path,
                    "remote or data URI manifest resource is not embedded",
                )
            elif is_font:
                record_omission(
                    _resolve_manifest_path(base_dir, href), opf_path,
                    "declared font is not embedded",
                )

        linear_refs: list[tuple[str, bool]] = []
        auxiliary_refs: list[tuple[str, bool]] = []
        for itemref in list(spine_node):
            if _strip_ns(itemref.tag) != "itemref":
                continue
            rid = itemref.attrib.get("idref")
            if rid:
                linear = itemref.attrib.get("linear", "yes").strip().lower() != "no"
                (linear_refs if linear else auxiliary_refs).append((rid, linear))
        spine_refs = linear_refs + auxiliary_refs

        nav_targets = _extract_nav_toc_targets(
            z=z,
            manifest_hrefs=manifest,
            manifest_media_types=manifest_media_types,
            manifest_properties=manifest_properties,
            base_dir=base_dir,
        )
        ncx_targets = _build_ncx_label_map(
            z=z,
            spine_node=spine_node,
            manifest_hrefs=manifest,
            manifest_media_types=manifest_media_types,
            base_dir=base_dir,
        )

        extracted_resource_bytes = 0
        spine_items: list[SpineItem] = []
        svg_spine_paths: set[str] = set()

        def select_spine_item(item_id: str) -> tuple[str, tuple[str, ...]]:
            chain: list[str] = []
            current = item_id
            while True:
                if current in chain:
                    raise ValueError(f"Cyclic manifest fallback chain: {item_id}")
                if current not in manifest:
                    raise ValueError(f"Malformed OPF: spine/fallback item '{current}' is missing from manifest")
                chain.append(current)
                media_type = manifest_media_types[current].lower()
                local = _resolve_book_href(opf_path, manifest[current]) is not None
                if local and media_type == "application/xhtml+xml":
                    break
                fallback = manifest_fallbacks.get(current)
                if not fallback:
                    if not local:
                        raise ValueError(f"Remote spine content is not supported: {manifest[current]}")
                    if media_type == "image/svg+xml":
                        break  # Retain the existing SVG omission behavior.
                    raise ValueError(f"Unsupported spine media type without readable fallback: {media_type}")
                current = fallback
            aliases = tuple(_resolve_manifest_path(base_dir, manifest[iid]) for iid in chain
                            if _resolve_book_href(opf_path, manifest[iid]) is not None)
            for iid in chain[:-1]:
                record_omission(_reported_media_path(opf_path, manifest[iid], manifest[iid]),
                                opf_path, "spine resource replaced by manifest fallback")
            return current, aliases

        for spine_idx, (item_id, linear) in enumerate(spine_refs, start=1):
            item_id, aliases = select_spine_item(item_id)
            rel = manifest[item_id]
            full = _resolve_manifest_path(base_dir, rel)
            if manifest_media_types.get(item_id, "").lower() == "image/svg+xml":
                svg_spine_paths.add(full)
                record_omission(full, opf_path, "SVG spine content is not rendered")
            try:
                raw_bytes, extracted_resource_bytes = _read_zip_member(
                    z,
                    full,
                    size_limit=MAX_XHTML_BYTES,
                    aggregate_budget=MAX_TOTAL_RESOURCE_BYTES,
                    aggregate_used=extracted_resource_bytes,
                    kind="Spine XHTML",
                )
            except KeyError as e:
                raise ValueError(f"Missing spine item in EPUB: {full}") from e
            document = _parse_xhtml(raw_bytes, full)
            try:
                body = _body_element(document)
            except ValueError as e:
                raise ValueError(f"Invalid XHTML: {full}: {e}") from e
            anchor = f"spine_{spine_idx}"
            body_fragments = tuple(body.get(key) for key in ("id", "name")
                                   if body.tag == "body" and body.get(key))
            fragment_anchors = {fragment: anchor for fragment in body_fragments}
            for fragment_idx, fragment in enumerate(_fragment_ids(body), start=1):
                if fragment not in fragment_anchors:
                    fragment_anchors[fragment] = f"{anchor}_frag_{fragment_idx}"
            spine_items.append(
                SpineItem(
                    index=spine_idx,
                    href=rel,
                    full_path=full,
                    aliases=aliases,
                    base_url=_document_base_url(document, full),
                    anchor=anchor,
                    stem=Path(rel).stem,
                    document=document,
                    body=body,
                    body_fragments=body_fragments,
                    fragment_anchors=fragment_anchors,
                    linear=linear,
                )
            )

        file_anchor_map: dict[str, str] = {}
        fragment_anchor_map: dict[tuple[str, str], str] = {}
        # References from other documents to a shared fallback choose its first
        # occurrence. Each occurrence retains its own markers and self-links.
        for item in spine_items:
            for path in item.aliases:
                file_anchor_map.setdefault(path, item.anchor)
                for fragment, anchor in item.fragment_anchors.items():
                    fragment_anchor_map.setdefault((path, fragment), anchor)

        image_path_to_recindex: dict[str, int] = {}
        image_records: list[bytes] = []

        def embed_image(target_path: str, source: str, item_id: Optional[str] = None,
                        *, cover: bool = False) -> Optional[int]:
            nonlocal extracted_resource_bytes
            item_id = item_id if item_id is not None else manifest_paths.get(target_path)
            if item_id is None:
                record_omission(target_path, source, "image is not declared in the EPUB manifest")
                return None
            media_type = manifest_media_types.get(item_id, "")
            if not _is_supported_image_media_type(media_type):
                record_omission(target_path, source, f"unsupported image format: {media_type or 'unspecified'}")
                return None
            recindex = image_path_to_recindex.get(target_path)
            if recindex is not None:
                image_data = image_records[recindex - 1]
            else:
                try:
                    image_data, extracted_resource_bytes = _read_zip_member(
                        z, target_path, size_limit=MAX_IMAGE_BYTES,
                        aggregate_budget=MAX_TOTAL_RESOURCE_BYTES,
                        aggregate_used=extracted_resource_bytes, kind="Image resource",
                    )
                except KeyError:
                    record_omission(target_path, source, "image file is missing from the EPUB")
                    return None
            if cover and not _raster_signature_matches(image_data, media_type):
                record_omission(target_path, source, "cover image signature does not match its declared raster format")
                return None
            if recindex is None:
                image_records.append(image_data)
                recindex = len(image_records)
                image_path_to_recindex[target_path] = recindex
            return recindex

        for item in spine_items:
            image_sources, unsupported, inline_svg = _media_references(item.body)
            if inline_svg and item.full_path not in svg_spine_paths:
                record_omission("<inline SVG>", item.full_path, "inline SVG is not rendered")
            for kind, raw_src in unsupported:
                resource = _reported_media_path(item.full_path, raw_src, f"<{kind}>", item.base_url)
                record_omission(resource, item.full_path, f"{kind} is not supported")

            for raw_src in image_sources:
                if not raw_src:
                    record_omission("<img without src>", item.full_path, "image has no source")
                    continue
                resolved = _resolve_book_href(item.full_path, raw_src, item.base_url)
                if resolved is None:
                    resource = _reported_media_path(item.full_path, raw_src, raw_src, item.base_url)
                    record_omission(resource, item.full_path, "external or data URI image is not embedded")
                    continue
                target_path, _fragment = resolved
                embed_image(target_path, item.full_path)

        cover_index = None
        cover_id = _declared_cover_item(metadata, manifest_properties)
        if cover_id is not None:
            cover_href = manifest.get(cover_id)
            if cover_href is None:
                record_omission(cover_id, opf_path, "cover item is not declared in the EPUB manifest")
            else:
                resolved = _resolve_book_href(opf_path, cover_href)
                if resolved is None:
                    record_omission(_reported_media_path(opf_path, cover_href, cover_href),
                                    opf_path, "external or data URI cover is not embedded")
                else:
                    recindex = embed_image(resolved[0], opf_path, cover_id, cover=True)
                    if recindex is not None:
                        cover_index = recindex - 1

        css_cache: dict[str, list[tuple[str, dict[str, str]]]] = {}
        css_errors: dict[str, str] = {}

        def document_styles(item: SpineItem) -> _CssStyles:
            nonlocal extracted_resource_bytes
            rules: list[tuple[str, dict[str, str]]] = []
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
                    if len(text.encode("utf-8")) > MAX_CSS_BYTES:
                        raise ValueError(f"CSS file too large: embedded style in {item.full_path}")
                    rules.extend(_css_rules(text))
                    continue
                rel = elem.get("rel", "").lower().split()
                href = elem.get("href")
                if "stylesheet" not in rel or "alternate" in rel or not href:
                    continue
                resolved = _resolve_book_href(item.full_path, href, item.base_url)
                if resolved is None:
                    record_omission(_reported_media_path(item.full_path, href, href, item.base_url),
                                    item.full_path, "external or data URI stylesheet is not loaded")
                    continue
                path = resolved[0]
                if path not in css_cache:
                    css_cache[path] = []
                    try:
                        data, extracted_resource_bytes = _read_zip_member(
                            z, path, size_limit=MAX_CSS_BYTES,
                            aggregate_budget=MAX_TOTAL_RESOURCE_BYTES,
                            aggregate_used=extracted_resource_bytes, kind="CSS file")
                    except KeyError:
                        css_errors[path] = "stylesheet is missing"
                    else:
                        try:
                            text = data.decode("utf-8-sig")
                        except UnicodeDecodeError:
                            css_errors[path] = "stylesheet is not UTF-8"
                        else:
                            css_cache[path] = _css_rules(text)
                if path in css_errors:
                    record_omission(path, item.full_path, css_errors[path])
                rules.extend(css_cache[path])
            return _CssStyles(item.document, rules)

        parts: list[str] = []
        fallback_toc: list[tuple[str, str, str, int]] = []
        for item in spine_items:
            sanitizer = MinimalHtmlSanitizer(
                current_path=item.full_path,
                file_anchor_map=file_anchor_map,
                fragment_anchor_map=fragment_anchor_map,
                image_path_to_recindex=image_path_to_recindex,
                base_url=item.base_url,
                current_aliases=item.aliases,
                fragment_anchors=item.fragment_anchors,
                file_anchor=item.anchor,
                styles=document_styles(item),
            )
            clean = sanitizer.sanitize(item.body)
            # Derive fallback labels from the normalized reading text too.
            ncx_title = next((target.label for target in ncx_targets if target.path in item.aliases and target.fragment is None), None)
            guessed_title = _extract_title(item.document)
            chapter_title = ncx_title or guessed_title
            if (
                not chapter_title
                or chapter_title.strip().lower() in ("unknown", "untitled")
                or chapter_title.strip().lower() == book_title.strip().lower()
            ):
                body_snippet = _extract_body_snippet(item.document, book_title)
                chapter_title = body_snippet or chapter_title or item.stem or item.anchor

            parts.append(f'<a name="{item.anchor}" id="{item.anchor}"></a>')
            if clean:
                if item.linear:
                    fallback_toc.append((item.anchor, chapter_title, item.stem, item.index))
                parts.append(clean)
                parts.append("<mbp:pagebreak/>")

        def resolve_toc_targets(targets: list[TocTarget]) -> tuple[list[tuple[str, str, str, int]], bool]:
            entries: list[tuple[str, str, str, int]] = []
            seen_anchors: set[str] = set()
            unresolved = False
            for target in targets:
                anchor = (
                    fragment_anchor_map.get((target.path, target.fragment))
                    if target.fragment else file_anchor_map.get(target.path)
                )
                if anchor is None:
                    unresolved = True
                    continue
                if anchor in seen_anchors:
                    continue
                spine_item = next((item for item in spine_items if target.path in item.aliases), None)
                if spine_item is None:
                    unresolved = True
                    continue
                seen_anchors.add(anchor)
                entries.append((anchor, target.label, spine_item.stem, spine_item.index))
            return entries, unresolved

        nav_toc, nav_incomplete = resolve_toc_targets(nav_targets)
        ncx_toc, ncx_incomplete = resolve_toc_targets(ncx_targets)
        # Keep a complete source TOC, even if it intentionally lists fewer chapters.
        # Replace a partially broken TOC only when another source has more entries.
        if nav_incomplete:
            raw_toc = max((nav_toc, ncx_toc, fallback_toc), key=len)
        elif nav_toc:
            raw_toc = nav_toc
        elif ncx_incomplete:
            raw_toc = max((ncx_toc, fallback_toc), key=len)
        else:
            raw_toc = ncx_toc or fallback_toc

        label_counts: dict[str, int] = {}
        for _, label, _, _ in raw_toc:
            label_counts[label] = label_counts.get(label, 0) + 1

        toc_entries: list[tuple[str, str]] = []
        used_labels: set[str] = set()
        for anchor, label, stem, spine_idx in raw_toc:
            if label_counts[label] > 1:
                resolved = f"{label} ({spine_idx})"
            else:
                resolved = label

            if resolved in used_labels:
                resolved = f"Chapter {spine_idx}"
            used_labels.add(resolved)
            toc_entries.append((anchor, resolved))

        html_content = "".join(parts)
        logger.info("Parsed %d spine items. Title: %s", len(spine_items), book_title)
        return EpubData(
            title=book_title,
            author=book_author,
            uuid=book_uuid,
            html_content=html_content,
            toc_entries=tuple(toc_entries),
            image_records=tuple(image_records),
            omitted_media=tuple(omissions),
            language=book_language,
            cover_index=cover_index,
        )


class MinimalHtmlSanitizer:
    _BLOCKS: frozenset[str] = frozenset(
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
    _INLINE: frozenset[str] = frozenset(
        {"b", "i", "strong", "em", "sup", "sub", "u", "code", "span", "a", "img", "mbp:pagebreak"}
    )
    _ALLOWED: frozenset[str] = _BLOCKS | _INLINE
    _FLATTENED_BLOCKS: frozenset[str] = frozenset({
        "dl", "dt", "dd", "section", "article", "aside", "header", "footer",
        "main", "nav", "figure", "figcaption", "address", "caption", "details",
        "summary", "hgroup", "form", "fieldset", "legend", "menu",
    })
    _HEADING_HINTS: frozenset[str] = frozenset({"chapter-title", "chap-title", "heading", "chapterhead", "chapter-heading"})
    _CENTER_HINTS: frozenset[str] = frozenset({"center", "centre", "centered", "centred", "epigraph", "ornament", "separator", "scene-break", "scenebreak", "asterism", "dinkus"})
    _RIGHT_HINTS: frozenset[str] = frozenset({"right", "author", "attribution", "credit", "byline", "source"})

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
        self.fed: list[str] = []
        self.styles = styles

    def _ensure_block_sep(self) -> None:
        if self.fed:
            last = self.fed[-1]
            if last and last[-1] != "\n":
                self.fed.append("\n")

    @staticmethod
    def _tokenize_hints(*values: str) -> set[str]:
        tokens: set[str] = set()
        for value in values:
            tokens.update(token for token in re.split(r"[^a-z0-9_-]+", value.lower()) if token)
        return tokens

    @staticmethod
    def _derive_alignment(style: str, hint_tokens: set[str]) -> Optional[str]:
        # This is a fallback hint, not margin layout: require the final complete
        # value on each side to be exactly auto, without interpreting lengths.
        margins = {name: value.lower() for name, value in _css_declaration_values(style)
                   if name in {"margin-left", "margin-right"}}
        if margins.get("margin-left") == margins.get("margin-right") == "auto":
            return "center"
        if hint_tokens & MinimalHtmlSanitizer._CENTER_HINTS:
            return "center"
        if hint_tokens & MinimalHtmlSanitizer._RIGHT_HINTS:
            return "right"
        return None

    def _inject_named_anchors(self, attrs, in_link: bool = False) -> None:
        seen: set[str] = set()
        for key, value in attrs:
            if key not in {"id", "name"} or not value or value in seen:
                continue
            seen.add(value)
            target = self.fragment_anchors.get(value)
            if target:
                marker = (f'<span id="{target}"></span>' if in_link else
                          f'<a name="{target}" id="{target}"></a>')
                self.fed.append(marker)

    def _rewrite_href(self, href: str) -> Optional[str]:
        resolved = _resolve_book_href(self.current_path, href, self.base_url)
        if resolved is None:
            return urllib.parse.urljoin(self.base_url, href)

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
        self.fed = []
        if self.styles is None or body not in self.styles.styles:
            self.styles = _CssStyles(body, [])
        _normalize_drop_caps(body, self.styles)
        self._emit(body)
        return "".join(self.fed)

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
        output_tag = tag if tag in self._ALLOWED and not (flatten_table and table_tag) else None
        hints = self._tokenize_hints(attrs.get("class", ""), attrs.get("id", ""))
        if output_tag in {"p", "div"} and hints & self._HEADING_HINTS:
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
        if tag == "img":
            src = attrs.get("src", "")
            resolved = _resolve_book_href(self.current_path, src, self.base_url) if src else None
            recindex = self.image_path_to_recindex.get(resolved[0]) if resolved else None
            if recindex is not None:
                self.fed.append(f'<img recindex="{recindex}"/>')
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
            elif output_tag in self._BLOCKS:
                align = style.get("text-align") or self._derive_alignment(attrs.get("style", ""), hints)
                if align:
                    attr_str = f' align="{align}"'
                if output_tag in {"p", "div", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"} and "text-indent" in style:
                    attr_str += f' width="{style["text-indent"]}"'
            if output_tag in self._BLOCKS:
                self._ensure_block_sep()
            self.fed.append(f"<{output_tag}{attr_str}>")
        elif table_tag:
            self.fed.append(" ")
        elif tag in self._FLATTENED_BLOCKS:
            self._ensure_block_sep()
        if elem.text:
            self._emit_text(elem.text, style)
        for child in elem:
            self._emit(child, flatten_table, in_link or output_tag == "a")
            if child.tail:
                self._emit_text(child.tail, style)
        if output_tag:
            self.fed.append(f"</{output_tag}>")
            if output_tag in self._BLOCKS:
                self.fed.append("\n")
        elif table_tag:
            self.fed.append("\n" if tag in {"tr", "table"} else " ")
        elif tag in self._FLATTENED_BLOCKS:
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
    def _build_toc_html(entries: tuple[tuple[str, str], ...], file_positions: list[int]) -> str:
        if len(entries) < 2:
            return ""
        if len(entries) != len(file_positions):
            raise ValueError("TOC entry count does not match filepos count")

        parts = ["<h1>Table of Contents</h1>"]
        for (_, title), filepos in zip(entries, file_positions):
            if filepos < 0 or filepos >= TOC_FILEPOS_MAX:
                raise ValueError(f"TOC filepos out of range: {filepos}")
            safe_title = htmlmod.escape(title, quote=False)
            safe_filepos = f"{filepos:0{TOC_FILEPOS_WIDTH}d}"
            parts.append(f'<p><a filepos="{safe_filepos}">{safe_title}</a></p>')
        parts.append("<mbp:pagebreak/>")
        return "".join(parts)

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

    def _compute_toc_positions(self, html_prefix_len: int, positions: dict[bytes, int]) -> list[int]:
        entries = self.epub.toc_entries
        if len(entries) < 2:
            return []

        body_anchor_positions = self._find_anchor_positions(positions)
        provisional_positions = [0] * len(entries)
        # Fixed-width filepos digits keep TOC byte length stable between provisional/final passes.
        provisional_toc = _encode_mobi_text(MobiWriter._build_toc_html(entries, provisional_positions))
        toc_len = len(provisional_toc)
        return [html_prefix_len + toc_len + pos for pos in body_anchor_positions]

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
        html_prefix = (
            "<html><head>"
            f'<meta http-equiv="Content-Type" content="text/html; charset={HTML_META_CHARSET}"/>'
            "</head><body>"
        )
        html_suffix = "</body></html>"

        prefix_bytes = _encode_mobi_text(html_prefix)
        body_bytes = _encode_mobi_text(self.epub.html_content)
        body_bytes, internal_targets = self._prepare_internal_links(body_bytes, self._anchor_positions(body_bytes))
        # Replacements change byte offsets. Scan once more, then share these
        # positions between TOC construction and final fixed-width link values.
        positions = self._anchor_positions(body_bytes)
        guide_bytes = b""
        toc_bytes = b""
        toc_filepos = None
        toc_entry_positions: tuple[int, ...] = ()
        toc_prefix_len = len(prefix_bytes)
        if len(self.epub.toc_entries) >= 2:
            provisional_guide = _encode_mobi_text(MobiWriter._build_guide_html(0))
            toc_filepos = len(prefix_bytes) + len(provisional_guide)
            guide_bytes = _encode_mobi_text(MobiWriter._build_guide_html(toc_filepos))
            toc_prefix_len += len(guide_bytes)
            final_positions = self._compute_toc_positions(toc_prefix_len, positions)
            toc_entry_positions = tuple(final_positions)
            toc_bytes = _encode_mobi_text(MobiWriter._build_toc_html(self.epub.toc_entries, final_positions))

        body_bytes = self._finish_internal_links(
            body_bytes, len(prefix_bytes) + len(guide_bytes) + len(toc_bytes), internal_targets, positions
        )
        suffix_bytes = _encode_mobi_text(html_suffix)
        return TextLayout(
            text_bytes=prefix_bytes + guide_bytes + toc_bytes + body_bytes + suffix_bytes,
            toc_filepos=toc_filepos,
            toc_entry_positions=toc_entry_positions,
        )

    @staticmethod
    def _build_tagx() -> bytes:
        tags = (
            (1, 1, 0x01, 0),
            (2, 1, 0x02, 0),
            (3, 1, 0x04, 0),
            (4, 1, 0x08, 0),
            (0, 0, 0x00, 1),
        )
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
        if len(self.epub.toc_entries) < 2:
            return []
        if len(self.epub.toc_entries) > 0xFFFF:
            raise _NavigationSizeError("Logical TOC entry count exceeds single-record limit")

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
        entry_positions = list(layout.toc_entry_positions)
        text_length = len(layout.text_bytes)
        # Auxiliary spine items can put TOC order out of physical text order.
        positions_in_text_order = sorted(set(entry_positions))
        end_by_position = {
            filepos: positions_in_text_order[index + 1]
            if index + 1 < len(positions_in_text_order) else text_length
            for index, filepos in enumerate(positions_in_text_order)
        }
        for index, ((_, _title), filepos, label_offset) in enumerate(
            zip(self.epub.toc_entries, entry_positions, label_offsets)
        ):
            entry_offsets.append(INDX_HEADER_LEN + len(entries_blob))
            name = f"{index:03d}".encode("ascii")
            length = max(1, end_by_position[filepos] - filepos)

            entries_blob.append(len(name))
            entries_blob.extend(name)
            entries_blob.append(0x0F)
            entries_blob.extend(_encode_vwi(filepos))
            entries_blob.extend(_encode_vwi(length))
            entries_blob.extend(_encode_vwi(label_offset))
            entries_blob.extend(_encode_vwi(0))

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

        tagx = self._build_tagx()
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
                if i + 1 < n and b == 0x20 and 0x40 <= data[i + 1] <= 0x7F:
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
            raise ValueError(f"Too many PalmDOC text records: {text_rec_count}")
        if total_records > 0xFFFF:
            raise ValueError(f"Too many records for PDB: {total_records}")

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

    def _build_exth(self) -> bytes:
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
    def _compute_record_indices(text_rec_count: int, nav_rec_count: int) -> tuple[int, int]:
        flis_idx = 1 + text_rec_count + nav_rec_count
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
        exth = self._build_exth()
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
                    HTML_META_CHARSET,
                )

        layout = self._build_text_layout()
        try:
            nav_records = self._build_navigation_records(layout)
        except _NavigationSizeError as e:
            logger.warning("Logical TOC omitted: %s; the inline TOC is retained", e)
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
        first_nonbook_candidates = [idx for idx in (nav_index_idx, first_image_idx, flis_idx) if idx is not None]
        first_nonbook = min(first_nonbook_candidates)
        record0 = self._build_record0(
            uncompressed_text_len=len(text_bytes),
            text_rec_count=len(text_records),
            flis_idx=flis_idx,
            fcis_idx=fcis_idx,
            first_nonbook=first_nonbook,
            nav_index_idx=nav_index_idx,
            first_image_idx=first_image_idx,
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

    for path in candidates:
        docs = os.path.join(path, "documents")
        if not os.path.isdir(docs):
            continue
        if "Kindle" not in os.path.basename(path) and not os.path.exists(os.path.join(path, "system")):
            continue
        dest = os.path.join(docs, os.path.basename(source_file))
        # Actual copy failures propagate to main(), which returns a failure
        # status. The completed local conversion remains available for retry.
        with _atomic_output(dest) as destination:
            with open(source_file, "rb") as source:
                shutil.copyfileobj(source, destination)
        logger.info("Copied to Kindle: %s", dest)
        return

    logger.warning("No Kindle detected.")


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
        help="List media resources that were omitted from the MOBI output",
    )
    return parser


def _log_media_omissions(omissions: tuple[MediaOmission, ...], detailed: bool) -> None:
    if not omissions:
        if detailed:
            logger.info("No omissions detected by the media scan; CSS background assets were not inspected.")
        return

    logger.warning(
        "%d media omission%s detected%s",
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
            raise ValueError("Output path resolves to the input EPUB; choose a different output path")
        epub_data = parse_epub(infile)
        MobiWriter(epub_data).build(str(outfile))
        _log_media_omissions(epub_data.omitted_media, args.report_omissions)

        if args.deploy:
            deploy_to_kindle(str(outfile))

        return 0
    except Exception as e:
        logger.error("Error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
