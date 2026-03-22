#!/usr/bin/env python3
"""
EPUB2 -> MOBI6 generator (Legacy Kindle Target)
"""

from __future__ import annotations

import argparse
import html as htmlmod
import logging
import os
import posixpath
import re
import shutil
import struct
import sys
import urllib.parse
import zipfile
import zlib
# Note: Standard ET is not secure against maliciously constructed XML data
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional, Tuple, Union

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
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
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_RESOURCE_BYTES = 256 * 1024 * 1024
_XML_UNSAFE_DECL_RE = re.compile(br"<!\s*ENTITY\b", flags=re.IGNORECASE)

# EXTH Types
EXTH_AUTHOR = 100
EXTH_TITLE = 503
EXTH_SOURCE = 112
EXTH_ASIN = 113
EXTH_CDETYPE = 501  # EBOK/PDOC

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
class EpubData:
    title: str
    author: str
    uuid: str
    html_content: str
    toc_entries: tuple[tuple[str, str], ...]
    image_records: tuple[bytes, ...] = ()


@dataclass(frozen=True)
class SpineItem:
    index: int
    href: str
    full_path: str
    anchor: str
    stem: str
    raw_html: str
    body_html: str


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


def _palm_time_now() -> int:
    return int((datetime.now() - datetime(1904, 1, 1)).total_seconds())


def _crc32_u32(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF


def _encode_mobi_text(s: str) -> bytes:
    """Encode as CP1252. Use XML entities for characters that don't fit."""
    return s.encode(MOBI_TEXT_ENCODING_PY, errors="xmlcharrefreplace")


def _encode_meta(s: str) -> bytes:
    """Encode metadata same as content."""
    return s.encode(MOBI_TEXT_ENCODING_PY, errors="replace")


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


def _parse_xml(data: bytes, source_name: str) -> ET.Element:
    if len(data) > MAX_XML_BYTES:
        raise ValueError(f"XML file too large: {source_name}")
    # Block DTD/entity declarations to avoid expansion attacks in untrusted input.
    if _XML_UNSAFE_DECL_RE.search(data):
        raise ValueError(f"Unsafe XML declaration in {source_name}")
    try:
        return ET.fromstring(data)
    except ET.ParseError as e:
        raise ValueError(f"Malformed XML: {source_name}") from e


def _find_opf(z: zipfile.ZipFile) -> tuple[str, str]:
    try:
        txt = z.read("META-INF/container.xml")
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

    normalized_opf = posixpath.normpath(opf_path.replace("\\", "/")).lstrip("/")
    if normalized_opf in ("", "."):
        raise ValueError("Invalid EPUB container: empty OPF path")
    return normalized_opf, posixpath.dirname(normalized_opf)


def _extract_title(html_str: str) -> Optional[str]:
    parser = SimpleTitleExtractor()
    parser.feed(html_str)
    parser.close()
    heading = parser.heading.strip()
    if heading:
        return heading
    title = parser.title.strip()
    if title:
        return title
    return None


def _extract_body_snippet(html_str: str, book_title: str, max_words: int = 10) -> Optional[str]:
    candidate = _extract_body_html(html_str)

    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", candidate)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = htmlmod.unescape(text)
    text = " ".join(text.split())
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
    snippet = re.sub(r"^([A-Z]) (?=[a-z])", r"\1", snippet)
    return snippet or None


def _normalize_epub_path(path: str) -> str:
    return posixpath.normpath(path.replace("\\", "/")).lstrip("/")


def _is_supported_image_media_type(media_type: str) -> bool:
    return media_type in {"image/jpeg", "image/png", "image/gif"}


def _extract_body_html(html_str: str) -> str:
    lower = html_str.lower()
    body_open = lower.find("<body")
    if body_open != -1:
        body_tag_end = html_str.find(">", body_open)
        start = body_tag_end + 1 if body_tag_end != -1 else body_open
        body_close = lower.rfind("</body>")
        if body_close != -1 and body_close > start:
            return html_str[start:body_close]
        return html_str[start:]

    head_close = lower.find("</head>")
    if head_close != -1:
        return html_str[head_close + len("</head>") :]
    return html_str


def _resolve_book_href(current_path: str, href: str) -> Optional[Tuple[str, Optional[str]]]:
    parsed = urllib.parse.urlsplit(href)
    if parsed.scheme or parsed.netloc:
        return None

    target_path = current_path
    if parsed.path:
        target_path = _normalize_epub_path(
            posixpath.join(posixpath.dirname(current_path), urllib.parse.unquote(parsed.path))
        )

    fragment = urllib.parse.unquote(parsed.fragment) if parsed.fragment else None
    return target_path, fragment


class FragmentIdCollector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.fragments: list[str] = []
        self._seen: set[str] = set()

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in {"id", "name"} and value and value not in self._seen:
                self.fragments.append(value)
                self._seen.add(value)


class ImageRefCollector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sources: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "img":
            return
        for key, value in attrs:
            if key == "src" and value:
                self.sources.append(value)
                break


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

    nav_path = _normalize_epub_path(posixpath.join(base_dir, nav_href))
    try:
        nav_root = _parse_xml(z.read(nav_path), nav_path)
    except (KeyError, ValueError):
        logger.warning("Unable to read EPUB3 nav document: %s", nav_path)
        return []

    nav_base = posixpath.dirname(nav_path)
    toc_nav = None
    for elem in nav_root.iter():
        if not elem.tag.endswith("nav"):
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
    for elem in toc_nav.iter():
        if not elem.tag.endswith("a"):
            continue
        raw_href = elem.attrib.get("href")
        if not raw_href:
            continue
        label = " ".join("".join(elem.itertext()).split())
        if not label:
            continue
        resolved = _resolve_book_href(nav_path, raw_href)
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

    ncx_path = _normalize_epub_path(posixpath.join(base_dir, ncx_href))
    try:
        ncx_root = _parse_xml(z.read(ncx_path), ncx_path)
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

    return z.read(member_path), new_total


def _decode_xhtml(data: bytes) -> str:
    if data.startswith((b"\x00\x00\xfe\xff", b"\xff\xfe\x00\x00")):
        return data.decode("utf-32", errors="replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig", errors="replace")

    head = data[:1024].decode("ascii", errors="ignore")
    match = re.search(r"""encoding=["']([A-Za-z0-9._-]+)["']""", head, flags=re.IGNORECASE)
    encoding = match.group(1) if match else "utf-8"
    try:
        return data.decode(encoding, errors="replace")
    except LookupError:
        logger.warning("Unknown XHTML encoding '%s'; falling back to utf-8", encoding)
        return data.decode("utf-8", errors="replace")


def parse_epub(filepath: Union[str, Path]) -> EpubData:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(str(filepath))

    book_title = "Unknown"
    book_author = "Unknown"
    book_uuid = "000000000000"

    with zipfile.ZipFile(filepath, "r") as z:
        opf_path, base_dir = _find_opf(z)
        try:
            opf_xml = z.read(opf_path)
        except KeyError as e:
            raise ValueError(f"OPF declared in container is missing: {opf_path}") from e
        opf_root = _parse_xml(opf_xml, opf_path)

        unique_id = opf_root.attrib.get("unique-identifier")
        fallback_uuid = None
        chosen_uuid = None

        for elem in opf_root.iter():
            t = _strip_ns(elem.tag)
            if t == "title" and elem.text:
                book_title = elem.text.strip() or book_title
            elif t == "creator" and elem.text:
                book_author = elem.text.strip() or book_author
            elif t == "identifier" and elem.text:
                ident = elem.text.strip()
                if ident and not fallback_uuid:
                    fallback_uuid = ident
                if unique_id and elem.attrib.get("id") == unique_id and ident:
                    chosen_uuid = ident

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
        for item in list(manifest_node):
            if _strip_ns(item.tag) == "item":
                iid = item.attrib.get("id")
                href = item.attrib.get("href")
                if iid and href:
                    manifest[iid] = href
                    manifest_media_types[iid] = item.attrib.get("media-type", "")
                    manifest_properties[iid] = item.attrib.get("properties", "")
        manifest_paths = {
            _normalize_epub_path(posixpath.join(base_dir, href)): iid
            for iid, href in manifest.items()
        }

        spine_refs: list[str] = []
        for itemref in list(spine_node):
            if _strip_ns(itemref.tag) != "itemref":
                continue
            if itemref.attrib.get("linear", "yes").strip().lower() == "no":
                continue
            rid = itemref.attrib.get("idref")
            if rid:
                spine_refs.append(rid)

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
        for spine_idx, item_id in enumerate(spine_refs, start=1):
            rel = manifest.get(item_id)
            if not rel:
                raise ValueError(f"Malformed OPF: spine item '{item_id}' is missing from manifest")
            full = _normalize_epub_path(posixpath.join(base_dir, rel))
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
            raw = _decode_xhtml(raw_bytes)
            body_html = _extract_body_html(raw)
            spine_items.append(
                SpineItem(
                    index=spine_idx,
                    href=rel,
                    full_path=full,
                    anchor=f"spine_{spine_idx}",
                    stem=Path(rel).stem,
                    raw_html=raw,
                    body_html=body_html,
                )
            )

        file_anchor_map = {item.full_path: item.anchor for item in spine_items}
        fragment_anchor_map: dict[tuple[str, str], str] = {}
        for item in spine_items:
            collector = FragmentIdCollector()
            collector.feed(item.body_html)
            collector.close()
            for fragment_idx, fragment in enumerate(collector.fragments, start=1):
                fragment_anchor_map[(item.full_path, fragment)] = f"{item.anchor}_frag_{fragment_idx}"

        image_path_to_recindex: dict[str, int] = {}
        image_records: list[bytes] = []
        for item in spine_items:
            collector = ImageRefCollector()
            collector.feed(item.body_html)
            collector.close()
            for raw_src in collector.sources:
                resolved = _resolve_book_href(item.full_path, raw_src)
                if resolved is None:
                    continue
                target_path, _fragment = resolved
                if target_path in image_path_to_recindex:
                    continue
                item_id = manifest_paths.get(target_path)
                if item_id is None:
                    continue
                media_type = manifest_media_types.get(item_id, "")
                if not _is_supported_image_media_type(media_type):
                    continue
                try:
                    image_data, extracted_resource_bytes = _read_zip_member(
                        z,
                        target_path,
                        size_limit=MAX_IMAGE_BYTES,
                        aggregate_budget=MAX_TOTAL_RESOURCE_BYTES,
                        aggregate_used=extracted_resource_bytes,
                        kind="Image resource",
                    )
                except KeyError:
                    logger.warning("Missing image in EPUB: %s", target_path)
                    continue
                image_records.append(image_data)
                image_path_to_recindex[target_path] = len(image_records)

        parts: list[str] = []
        fallback_toc: list[tuple[str, str, str, int]] = []
        for item in spine_items:
            ncx_title = next((target.label for target in ncx_targets if target.path == item.full_path and target.fragment is None), None)
            guessed_title = _extract_title(item.raw_html)
            chapter_title = ncx_title or guessed_title
            if (
                not chapter_title
                or chapter_title.strip().lower() in ("unknown", "untitled")
                or chapter_title.strip().lower() == book_title.strip().lower()
            ):
                body_snippet = _extract_body_snippet(item.raw_html, book_title)
                chapter_title = body_snippet or chapter_title or item.stem or item.anchor

            sanitizer = MinimalHtmlSanitizer(
                current_path=item.full_path,
                file_anchor_map=file_anchor_map,
                fragment_anchor_map=fragment_anchor_map,
                image_path_to_recindex=image_path_to_recindex,
            )
            clean = sanitizer.sanitize(item.body_html)
            if clean:
                fallback_toc.append((item.anchor, chapter_title, item.stem, item.index))
                parts.append(f'<a name="{item.anchor}" id="{item.anchor}"></a>')
                parts.append(clean)
                parts.append("<mbp:pagebreak/>")

        source_targets = nav_targets or ncx_targets
        raw_toc: list[tuple[str, str, str, int]] = []
        seen_anchors: set[str] = set()
        for target in source_targets:
            if target.fragment:
                anchor = fragment_anchor_map.get((target.path, target.fragment))
            else:
                anchor = file_anchor_map.get(target.path)
            if anchor is None or anchor in seen_anchors:
                continue
            spine_item = next((item for item in spine_items if item.full_path == target.path), None)
            if spine_item is None:
                continue
            seen_anchors.add(anchor)
            raw_toc.append((anchor, target.label, spine_item.stem, spine_item.index))

        if not raw_toc:
            raw_toc = fallback_toc

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
        )


class MinimalHtmlSanitizer(HTMLParser):
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
        {"b", "i", "strong", "em", "code", "span", "a", "img", "mbp:pagebreak"}
    )
    _ALLOWED: frozenset[str] = _BLOCKS | _INLINE

    _SUPPRESSED: frozenset[str] = frozenset({"script", "style"})
    _HEADING_HINTS: frozenset[str] = frozenset({"chapter-title", "chap-title", "heading", "chapterhead", "chapter-heading"})
    _CENTER_HINTS: frozenset[str] = frozenset({"center", "centre", "centered", "centred", "epigraph", "ornament", "separator", "scene-break", "scenebreak", "asterism", "dinkus"})
    _RIGHT_HINTS: frozenset[str] = frozenset({"right", "author", "attribution", "credit", "byline", "source"})

    def __init__(
        self,
        current_path: str,
        file_anchor_map: dict[str, str],
        fragment_anchor_map: dict[tuple[str, str], str],
        image_path_to_recindex: dict[str, int],
    ):
        super().__init__(convert_charrefs=True)
        self.current_path = current_path
        self.file_anchor_map = file_anchor_map
        self.fragment_anchor_map = fragment_anchor_map
        self.image_path_to_recindex = image_path_to_recindex
        self.fed: list[str] = []
        self._suppressed_depth = 0
        self._table_stack: list[tuple[bool, int]] = []
        self._output_tag_stack: list[Optional[str]] = []

    def _ensure_block_sep(self) -> None:
        if self.fed:
            last = self.fed[-1]
            if last and last[-1] != "\n":
                self.fed.append("\n")

    @staticmethod
    def _attr_map(attrs) -> dict[str, str]:
        return {k.lower(): (v or "") for k, v in attrs}

    @staticmethod
    def _tokenize_hints(*values: str) -> set[str]:
        tokens: set[str] = set()
        for value in values:
            tokens.update(token for token in re.split(r"[^a-z0-9_-]+", value.lower()) if token)
        return tokens

    @staticmethod
    def _derive_alignment(style: str, hint_tokens: set[str]) -> Optional[str]:
        normalized = style.lower().replace(" ", "")
        if "text-align:center" in normalized or ("margin-left:auto" in normalized and "margin-right:auto" in normalized):
            return "center"
        if "text-align:right" in normalized:
            return "right"
        if hint_tokens & MinimalHtmlSanitizer._CENTER_HINTS:
            return "center"
        if hint_tokens & MinimalHtmlSanitizer._RIGHT_HINTS:
            return "right"
        return None

    def _table_is_simple(self, tag: str, attr_map: dict[str, str]) -> bool:
        if tag == "table" and self._table_stack:
            return False
        for key in ("rowspan", "colspan"):
            value = attr_map.get(key, "").strip()
            if value and value not in {"1", ""}:
                return False
        return True

    def _in_simple_table(self) -> bool:
        return bool(self._table_stack and self._table_stack[-1][0])

    def _mark_current_table_complex(self) -> None:
        if not self._table_stack or not self._table_stack[-1][0]:
            return
        _is_simple, start_idx = self._table_stack[-1]
        del self.fed[start_idx:]
        self._table_stack[-1] = (False, start_idx)

    def _inject_named_anchors(self, attrs) -> None:
        seen: set[str] = set()
        for key, value in attrs:
            if key not in {"id", "name"} or not value or value in seen:
                continue
            seen.add(value)
            target = self.fragment_anchor_map.get((self.current_path, value))
            if target:
                self.fed.append(f'<a name="{target}" id="{target}"></a>')

    def _rewrite_href(self, href: str) -> Optional[str]:
        resolved = _resolve_book_href(self.current_path, href)
        if resolved is None:
            return href

        target_path, fragment = resolved
        if fragment:
            exact = self.fragment_anchor_map.get((target_path, fragment))
            if exact:
                return f"#{exact}"

        fallback = self.file_anchor_map.get(target_path)
        if fallback:
            return f"#{fallback}"
        return None

    def sanitize(self, html_str: str) -> str:
        self.fed = []
        self._suppressed_depth = 0
        self._table_stack = []
        self._output_tag_stack = []
        self.reset()
        self.feed(html_str)
        self.close()
        return "".join(self.fed)

    def handle_starttag(self, tag, attrs):
        if tag in self._SUPPRESSED:
            self._suppressed_depth += 1
            return

        self._inject_named_anchors(attrs)
        attr_map = self._attr_map(attrs)
        hint_tokens = self._tokenize_hints(attr_map.get("class", ""), attr_map.get("id", ""))
        effective_tag = tag

        if tag in {"p", "div"} and hint_tokens & self._HEADING_HINTS:
            effective_tag = "h2"

        if tag == "table":
            if self._table_stack:
                self._mark_current_table_complex()
            self._table_stack.append((self._table_is_simple(tag, attr_map), len(self.fed)))
        elif tag in {"td", "th"} and not self._table_is_simple(tag, attr_map) and self._table_stack:
            self._mark_current_table_complex()
        elif tag in {"thead", "tbody", "tfoot", "tr", "td", "th"} and self._table_stack and not self._table_stack[-1][0]:
            pass

        if tag in self._ALLOWED:
            if tag in {"thead", "tbody", "tfoot", "tr", "td", "th", "table"} and self._table_stack and not self._table_stack[-1][0]:
                if tag in {"td", "th"}:
                    self.fed.append(" ")
                if tag not in {"br", "mbp:pagebreak", "hr", "img"}:
                    self._output_tag_stack.append(None)
                return
            if tag in ("br", "mbp:pagebreak", "hr"):
                self.fed.append(f"<{tag}/>")
                return
            if tag == "img":
                src = attr_map.get("src", "")
                resolved = _resolve_book_href(self.current_path, src) if src else None
                recindex = self.image_path_to_recindex.get(resolved[0]) if resolved else None
                if recindex is not None:
                    self.fed.append(f'<img recindex="{recindex}"/>')
                return

            # Reconstruct the tag
            attr_str = ""
            # Only keep 'href' for anchors, ignore classes/styles as legacy MOBI ignores them mostly
            if effective_tag == "a":
                for k, v in attrs:
                    if k == "href":
                        rewritten = self._rewrite_href(v)
                        if rewritten:
                            attr_str = f' href="{htmlmod.escape(rewritten, quote=True)}"'
                        break
            elif effective_tag in self._BLOCKS:
                align = self._derive_alignment(attr_map.get("style", ""), hint_tokens)
                if align:
                    attr_str = f' align="{align}"'

            # Map modern semantics to legacy
            if effective_tag == "strong": effective_tag = "b"
            if effective_tag == "em": effective_tag = "i"

            if effective_tag in self._BLOCKS:
                self._ensure_block_sep()
            self.fed.append(f"<{effective_tag}{attr_str}>")
            self._output_tag_stack.append(effective_tag)
        elif tag not in {"br", "mbp:pagebreak", "hr", "img"}:
            self._output_tag_stack.append(None)

    def handle_endtag(self, tag):
        if tag in self._SUPPRESSED:
            self._suppressed_depth = max(0, self._suppressed_depth - 1)
            return

        output_tag = self._output_tag_stack.pop() if self._output_tag_stack else None

        if tag == "table" and self._table_stack:
            is_simple, _start_idx = self._table_stack.pop()
            if not is_simple:
                self.fed.append("\n")
                return
        elif tag in {"thead", "tbody", "tfoot", "tr", "td", "th"} and self._table_stack and not self._table_stack[-1][0]:
            if tag == "tr":
                self.fed.append("\n")
            elif tag in {"td", "th"}:
                self.fed.append(" ")
            return

        # Map modern semantics to legacy
        if output_tag is None:
            return
        if output_tag in {"hr", "img"}:
            return

        if output_tag in self._ALLOWED and output_tag not in ("br", "mbp:pagebreak"):
            self.fed.append(f"</{output_tag}>")

            # Legacy Kindle sometimes merges block elements if there isn't a newline
            if output_tag in self._BLOCKS:
                self.fed.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag in self._SUPPRESSED:
            return

        self._inject_named_anchors(attrs)
        if tag in self._ALLOWED:
            if tag == "br":
                self.fed.append("<br/>")
            elif tag == "hr":
                self.fed.append("<hr/>")
            elif tag == "img":
                attr_map = self._attr_map(attrs)
                src = attr_map.get("src", "")
                resolved = _resolve_book_href(self.current_path, src) if src else None
                recindex = self.image_path_to_recindex.get(resolved[0]) if resolved else None
                if recindex is not None:
                    self.fed.append(f'<img recindex="{recindex}"/>')
            elif tag == "mbp:pagebreak":
                self.fed.append("<mbp:pagebreak/>")

    def handle_data(self, data):
        if self._suppressed_depth:
            return
        # Escape content to ensure XML validity
        self.fed.append(htmlmod.escape(data, quote=False))


class SimpleTitleExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._in_title = False
        self._in_heading = False
        self.title = ""
        self.heading = ""

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
        elif tag in ("h1", "h2") and not self.heading:
            self._in_heading = True

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        elif tag in ("h1", "h2"):
            self._in_heading = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._in_heading:
            self.heading += data


class MobiWriter:
    def __init__(self, epub: EpubData):
        self.epub = epub

    def _find_anchor_positions(self, body_bytes: bytes) -> list[int]:
        anchors = [a for a, _ in self.epub.toc_entries]
        anchor_positions: list[int] = []
        for anchor in anchors:
            tag = _encode_mobi_text(f'<a name="{anchor}" id="{anchor}"></a>')
            pos = body_bytes.find(tag)
            if pos == -1:
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

    def _compute_toc_positions(self, html_prefix_len: int, body_bytes: bytes) -> list[int]:
        entries = self.epub.toc_entries
        if len(entries) < 2:
            return []

        body_anchor_positions = self._find_anchor_positions(body_bytes)
        provisional_positions = [0] * len(entries)
        # Fixed-width filepos digits keep TOC byte length stable between provisional/final passes.
        provisional_toc = _encode_mobi_text(MobiWriter._build_toc_html(entries, provisional_positions))
        toc_len = len(provisional_toc)
        return [html_prefix_len + toc_len + pos for pos in body_anchor_positions]

    def _build_text_layout(self) -> TextLayout:
        html_prefix = (
            "<html><head>"
            f'<meta http-equiv="Content-Type" content="text/html; charset={HTML_META_CHARSET}"/>'
            "</head><body>"
        )
        html_suffix = "</body></html>"

        prefix_bytes = _encode_mobi_text(html_prefix)
        body_bytes = _encode_mobi_text(self.epub.html_content)
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
            final_positions = self._compute_toc_positions(toc_prefix_len, body_bytes)
            toc_entry_positions = tuple(final_positions)
            toc_bytes = _encode_mobi_text(MobiWriter._build_toc_html(self.epub.toc_entries, final_positions))

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
                raise ValueError(f"IDXT offset out of range: {offset}")
            table.extend(struct.pack(">H", offset))
        pad = len(table) % 4
        if pad:
            table.extend(b"\x00" * (4 - pad))
        return bytes(table)

    def _build_navigation_records(self, layout: TextLayout) -> list[bytes]:
        if len(self.epub.toc_entries) < 2:
            return []

        label_record = bytearray()
        label_offsets: list[int] = []
        for _, title in self.epub.toc_entries:
            label_offsets.append(len(label_record))
            label_bytes = _encode_index_text(title)
            label_record.extend(_encode_vwi(len(label_bytes)))
            label_record.extend(label_bytes)

        if len(label_record) > 0xFFFF:
            raise ValueError("CNCX label record exceeds single-record limit")

        entry_offsets: list[int] = []
        entries_blob = bytearray()
        entry_positions = list(layout.toc_entry_positions)
        text_length = len(layout.text_bytes)
        for index, ((_, _title), filepos, label_offset) in enumerate(
            zip(self.epub.toc_entries, entry_positions, label_offsets)
        ):
            entry_offsets.append(INDX_HEADER_LEN + len(entries_blob))
            name = f"{index:03d}".encode("ascii")
            next_filepos = entry_positions[index + 1] if index + 1 < len(entry_positions) else text_length
            length = max(1, next_filepos - filepos)

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
        main_dummy = bytes((len(b"000"),)) + b"000"
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
        struct.pack_into(">I", mobi, OFF_LOCALE, 1033)

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
        struct.pack_into(">H", mobi, OFF_LAST_CONTENT, text_rec_count)
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
        layout = self._build_text_layout()
        nav_records = self._build_navigation_records(layout)
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

        with open(output_file, "wb") as f:
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
        try:
            dest = os.path.join(docs, os.path.basename(source_file))
            shutil.copy2(source_file, dest)
            logger.info("Copied to Kindle: %s", dest)
            return
        except Exception as e:
            logger.error("Copy failed: %s", e)
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
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    infile = Path(args.input_epub)
    outfile = Path(args.output_mobi) if args.output_mobi else infile.with_suffix(".mobi")

    try:
        epub_data = parse_epub(infile)
        MobiWriter(epub_data).build(str(outfile))

        if args.deploy:
            deploy_to_kindle(str(outfile))

        return 0
    except Exception as e:
        logger.error("Error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
