#!/usr/bin/env python3
"""
EPUB2 -> MOBI6 generator (Legacy Kindle Target)
"""

from __future__ import annotations

import html as htmlmod
import logging
import os
import re
import shutil
import struct
import sys
import zipfile
import zlib
# Note: Standard ET is not secure against maliciously constructed XML data
import xml.etree.ElementTree as ET

from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

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

MOBI_MAGIC = b"MOBI"
EXTH_MAGIC = b"EXTH"

# Encoding: Force CP1252 for Old Kindle Compatibility
MOBI_TEXT_ENCODING_ID = 1252        # Windows-1252
MOBI_TEXT_ENCODING_PY = "cp1252"    # Python codec name
HTML_META_CHARSET = "windows-1252"

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


class EpubParser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.title = "Unknown"
        self.author = "Unknown"
        self.uuid = "000000000000"
        self.html_content = ""

    def process(self) -> None:
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(self.filepath)

        with zipfile.ZipFile(self.filepath, "r") as z:
            opf_path, base_dir = self._find_opf(z)
            opf_root = ET.fromstring(z.read(opf_path))

            def strip_ns(tag: str) -> str:
                return tag.split("}", 1)[1] if "}" in tag else tag

            unique_id = opf_root.attrib.get("unique-identifier")
            fallback_uuid = None
            chosen_uuid = None

            for elem in opf_root.iter():
                t = strip_ns(elem.tag)
                if t == "title" and elem.text:
                    self.title = elem.text.strip() or self.title
                elif t == "creator" and elem.text:
                    self.author = elem.text.strip() or self.author
                elif t == "identifier" and elem.text:
                    ident = elem.text.strip()
                    if ident and not fallback_uuid:
                        fallback_uuid = ident
                    if unique_id and elem.attrib.get("id") == unique_id and ident:
                        chosen_uuid = ident

            if chosen_uuid:
                self.uuid = chosen_uuid
            elif fallback_uuid:
                self.uuid = fallback_uuid

            manifest_node = None
            spine_node = None
            for child in list(opf_root):
                t = strip_ns(child.tag)
                if t == "manifest":
                    manifest_node = child
                elif t == "spine":
                    spine_node = child
            if manifest_node is None or spine_node is None:
                raise ValueError("Malformed OPF: missing manifest or spine")

            manifest: dict[str, str] = {}
            for item in list(manifest_node):
                if strip_ns(item.tag) == "item":
                    iid = item.attrib.get("id")
                    href = item.attrib.get("href")
                    if iid and href:
                        manifest[iid] = href

            spine_refs: list[str] = []
            for itemref in list(spine_node):
                if strip_ns(itemref.tag) == "itemref":
                    rid = itemref.attrib.get("idref")
                    if rid:
                        spine_refs.append(rid)

            sanitizer = MinimalHtmlSanitizer()
            parts: list[str] = []
            for item_id in spine_refs:
                rel = manifest.get(item_id)
                if not rel:
                    continue
                full = os.path.join(base_dir, rel).replace("\\", "/")
                try:
                    # Read as UTF-8 from source, sanitize, then we encode to cp1252 later
                    raw = self._decode_xhtml(z.read(full))
                except KeyError:
                    logger.warning(f"Missing file in EPUB: {full}")
                    continue
                clean = sanitizer.sanitize(raw)
                if clean:
                    parts.append(clean)
                    parts.append("<mbp:pagebreak/>")

            self.html_content = "".join(parts)
            logger.info(f"Parsed {len(spine_refs)} spine items. Title: {self.title}")

    @staticmethod
    def _find_opf(z: zipfile.ZipFile) -> tuple[str, str]:
        try:
            txt = z.read("META-INF/container.xml")
            root = ET.fromstring(txt)
            opf_path = None
            for elem in root.iter():
                if elem.tag.endswith("rootfile"):
                    opf_path = elem.attrib.get("full-path")
                    break
            if not opf_path:
                raise ValueError("No rootfile found")
            return opf_path, os.path.dirname(opf_path)
        except Exception:
            for name in z.namelist():
                if name.endswith(".opf"):
                    return name, os.path.dirname(name)
            raise ValueError("No OPF file found")

    @staticmethod
    def _decode_xhtml(data: bytes) -> str:
        head = data[:1024].decode("ascii", errors="ignore")
        match = re.search(r'encoding=["\']([A-Za-z0-9._-]+)["\']', head)
        encoding = match.group(1) if match else "utf-8"
        try:
            return data.decode(encoding, errors="replace")
        except LookupError:
            return data.decode("utf-8", errors="replace")


class MinimalHtmlSanitizer(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        # Block elements that imply a new line
        self.blocks = {
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
        }
        # Inline elements to keep
        self.inline = {"b", "i", "strong", "em", "code", "span", "a", "mbp:pagebreak"}

        self.allowed = self.blocks | self.inline
        self.fed: list[str] = []

    def _ensure_block_sep(self) -> None:
        if self.fed:
            last = self.fed[-1]
            if last and last[-1] != "\n":
                self.fed.append("\n")

    def sanitize(self, html_str: str) -> str:
        self.fed = []
        self.reset()
        self.feed(html_str)
        self.close()
        return "".join(self.fed)

    def handle_starttag(self, tag, attrs):
        if tag in self.allowed:
            if tag in ("br", "mbp:pagebreak"):
                self.fed.append(f"<{tag}/>")
                return

            # Reconstruct the tag
            attr_str = ""
            # Only keep 'href' for anchors, ignore classes/styles as legacy MOBI ignores them mostly
            if tag == "a":
                for k, v in attrs:
                    if k == "href":
                        attr_str = f' href="{htmlmod.escape(v, quote=True)}"'
                        break

            # Map modern semantics to legacy
            if tag == "strong": tag = "b"
            if tag == "em": tag = "i"

            if tag in self.blocks:
                self._ensure_block_sep()
            self.fed.append(f"<{tag}{attr_str}>")

    def handle_endtag(self, tag):
        # Map modern semantics to legacy
        if tag == "strong": tag = "b"
        if tag == "em": tag = "i"

        if tag in self.allowed and tag not in ("br", "mbp:pagebreak"):
            self.fed.append(f"</{tag}>")

            # Legacy Kindle sometimes merges block elements if there isn't a newline
            if tag in self.blocks:
                self.fed.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag in self.allowed:
            if tag == "br":
                self.fed.append("<br/>")
            elif tag == "mbp:pagebreak":
                self.fed.append("<mbp:pagebreak/>")

    def handle_data(self, data):
        # Escape content to ensure XML validity
        self.fed.append(htmlmod.escape(data, quote=False))


class MobiWriter:
    def __init__(self, epub: EpubParser):
        self.epub = epub

    @staticmethod
    def _safe_chunk_bytes(b: bytes, limit: int) -> list[bytes]:
        # Safe for CP1252 (single byte encoding)
        return [b[i:i + limit] for i in range(0, len(b), limit)] or [b""]

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

    def _build_record0(self, text_len: int, text_rec_count: int, include_dummy: bool) -> tuple[bytes, int, int]:
        # PalmDOC
        palmdoc = bytearray(PALMDOC_LEN)
        struct.pack_into(">H", palmdoc, 0, 1)  # No compression
        struct.pack_into(">H", palmdoc, 2, 0)
        struct.pack_into(">I", palmdoc, 4, text_len)
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

        # Indices Logic
        dummy_idx = 1 + text_rec_count
        flis_idx = dummy_idx + (1 if include_dummy else 0)
        fcis_idx = flis_idx + 1

        first_nonbook = flis_idx

        struct.pack_into(">I", record0, PALMDOC_LEN + OFF_FIRST_NONBOOK, first_nonbook)
        struct.pack_into(">I", record0, PALMDOC_LEN + OFF_FIRST_IMAGE, first_nonbook)

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

        return bytes(record0), flis_idx, fcis_idx

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
        if n > 0xFFFF:
            raise ValueError(f"Too many records: {n}")
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
        html_str = (
            "<html><head>"
            f'<meta http-equiv="Content-Type" content="text/html; charset={HTML_META_CHARSET}"/>'
            "</head><body>"
            f"{self.epub.html_content}"
            "</body></html>"
        )

        # Encode text as CP1252 (with fallback)
        text_bytes = _encode_mobi_text(html_str)
        text_records = self._safe_chunk_bytes(text_bytes, TEXT_RECORD_MAX)

        include_dummy = False
        record0, _, _ = self._build_record0(
            text_len=len(text_bytes),
            text_rec_count=len(text_records),
            include_dummy=include_dummy,
        )

        records: list[bytes] = [record0]
        records.extend(text_records)
        records.extend([self._build_flis(), self._build_fcis(len(text_bytes)), self._build_eof()])

        pdb_header, rec_info = self._build_pdb_header_and_index(records)

        with open(output_file, "wb") as f:
            f.write(pdb_header)
            f.write(rec_info)
            f.write(b"\x00\x00")
            for rec in records:
                f.write(rec)

        logger.info(f"SUCCESS: Created {output_file}")


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
            logger.info(f"Copied to Kindle: {dest}")
            return
        except Exception as e:
            logger.error(f"Copy failed: {e}")
            return

    logger.warning("No Kindle detected.")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 epub2mobi.py <input.epub> [--deploy]")
        return

    infile = sys.argv[1]
    do_deploy = "--deploy" in sys.argv
    outfile = Path(infile).with_suffix(".mobi")

    try:
        p = EpubParser(infile)
        p.process()
        MobiWriter(p).build(outfile)

        if do_deploy:
            deploy_to_kindle(outfile)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
