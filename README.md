# epub2mobi.py

`epub2mobi.py` is a zero-dependency Python tool that converts EPUB files to legacy MOBI6 for older Kindle devices.

It writes PalmDB/MOBI structures directly with the standard library and focuses on robust, text-first output.

## Features

- Zero dependencies (`python3` only).
- EPUB parsing via OPF manifest/spine with path normalization for ZIP internals.
- Table of Contents generation with fixed-width `filepos` links.
- TOC labels prioritized from NCX nav labels, then headings/titles, then body snippets/fallbacks.
- PalmDOC compression (type `2`), applied per 4096-byte uncompressed text record.
- Legacy-compatible content sanitization (text/paragraphs/basic inline tags).
- Optional USB deploy to Kindle `documents` folder (`--deploy`).

## Requirements

- Python 3.10+

## CLI Usage

Convert:

```bash
python3 epub2mobi.py my_book.epub
```

Convert and deploy to a connected Kindle:

```bash
python3 epub2mobi.py my_book.epub --deploy
```

## Module Usage

```python
from epub2mobi import parse_epub, MobiWriter

epub_data = parse_epub("my_book.epub")
MobiWriter(epub_data).build("my_book.mobi")
```

## Scope and Limitations

- Output target is MOBI6 (not AZW3/KF8).
- Text-first conversion: advanced CSS, JavaScript, embedded fonts, and rich modern layout are not preserved.
- Images are currently not emitted into MOBI records.
- XHTML decoding supports BOMs and declared encodings, with fallback behavior for unknown encodings.
- XML guardrails reject entity declarations and oversized XML payloads.

## License

MIT License.
