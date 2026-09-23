# epub2mobi.py

`epub2mobi.py` is a zero-dependency Python tool that converts EPUB files to legacy MOBI6 for older Kindle devices.

It writes PalmDB/MOBI structures directly with the standard library and focuses on robust, text-first output.

## Features

- Zero dependencies (`python3` only).
- EPUB parsing via OPF manifest/spine, including URL-encoded resource paths and linked auxiliary content such as notes.
- EPUB3 `nav.xhtml` TOC support, with NCX fallback when nav targets cannot be resolved.
- Inline and logical Table of Contents generation for Kindle-compatible navigation.
- Fragment-level TOC extraction when valid in-spine targets exist.
- TOC labels prioritized from EPUB nav/NCX labels, then headings/titles, then body snippets/fallbacks.
- Internal EPUB links, including links to body IDs, are converted to MOBI byte-position links.
- Supported EPUB raster images (`jpeg`, `png`, `gif`) are emitted as MOBI resources when referenced from spine content.
- Omitted media are counted in a warning; `--report-omissions` lists affected resources, source documents, and reasons.
- Limited layout preservation for common book patterns such as centered headings, right-aligned attributions, scene breaks, and simple tables.
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

Choose an output path with `-o` or `--output`:

```bash
python3 epub2mobi.py my_book.epub -o converted.mobi
```

The output path cannot resolve to the input EPUB.

Convert and deploy to a connected Kindle:

```bash
python3 epub2mobi.py my_book.epub --deploy
```

To see which media were omitted:

```bash
python3 epub2mobi.py my_book.epub --report-omissions
```

The report covers referenced images, inline SVG, audio/video, embedded objects in spine content, and fonts declared in the manifest. The converter shows a one-line warning when it detects omissions even without the option.

## Scope and Limitations

- Output target is MOBI6 (not AZW3/KF8).
- Text-first conversion: advanced CSS, JavaScript, embedded fonts, SVG, fixed layout, and full modern EPUB styling are not preserved.
- Layout preservation is intentionally narrow and heuristic-based, not a general CSS engine.
- Tables are preserved only when they are simple rectangular structures; complex tables are flattened while retaining their text.
- Unsupported or missing images are skipped without failing the conversion.
- The omission report does not inspect CSS background images or other visual effects defined only in stylesheets.
- XHTML decoding supports BOMs and declared encodings, with fallback behavior for unknown encodings.
- XML guardrails reject entity declarations across supported encodings and limit ZIP members before reading them.

## License

MIT License.
