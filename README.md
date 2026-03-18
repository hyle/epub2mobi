# epub2mobi.py

`epub2mobi.py` is a zero-dependency Python tool that converts EPUB files to legacy MOBI6 for older Kindle devices.

It writes PalmDB/MOBI structures directly with the standard library and focuses on robust, text-first output.

## Features

- Zero dependencies (`python3` only).
- EPUB parsing via OPF manifest/spine with path normalization for ZIP internals.
- EPUB3 `nav.xhtml` TOC support, with NCX fallback.
- Inline and logical Table of Contents generation for Kindle-compatible navigation.
- Fragment-level TOC extraction when valid in-spine targets exist.
- TOC labels prioritized from EPUB nav/NCX labels, then headings/titles, then body snippets/fallbacks.
- Internal EPUB links and fragment targets are rewritten to work in the flattened MOBI output.
- Supported EPUB raster images (`jpeg`, `png`, `gif`) are emitted as MOBI resources when referenced from spine content.
- Layout preservation for common book patterns such as centered headings, right-aligned attributions, scene breaks, and simple tables.
- PalmDOC compression (type `2`), applied per 4096-byte uncompressed text record.
- Legacy-compatible body sanitization.
- Optional USB deploy to Kindle `documents` folder (`--deploy`).

## Requirements

- Python 3.9+

## CLI Usage

Convert:

```bash
python3 epub2mobi.py my_book.epub
```

Convert and deploy to a connected Kindle:

```bash
python3 epub2mobi.py my_book.epub --deploy
```

## Scope and Limitations

- Output target is MOBI6 (not AZW3/KF8).
- Text-first conversion: advanced CSS, JavaScript, embedded fonts, SVG, fixed layout, and full modern EPUB styling are not preserved.
- Layout preservation is intentionally narrow and heuristic-based, not a general CSS engine.
- Tables are preserved only when they are simple rectangular structures; complex tables are flattened to text.
- Unsupported or missing images are skipped without failing the conversion.
- XHTML decoding supports BOMs and declared encodings, with fallback behavior for unknown encodings.
- XML guardrails reject entity declarations and oversized XML payloads.

## License

MIT License.
