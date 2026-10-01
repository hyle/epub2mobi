# epub2mobi.py

`epub2mobi.py` is a zero-dependency Python tool that converts EPUB files to legacy MOBI6 for older Kindle devices.


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


## Features

- Zero dependencies (`python3` only).
- EPUB parsing via OPF manifest/spine, including URL-encoded resource paths and linked auxiliary content such as notes.
- Metadata uses the first nonempty Dublin Core title. Author names follow creator order, joined with `; `; EPUB2 role attributes and EPUB3 MARC role refinements select authors (`aut`) and creators without an understood role.
- EPUB3 `nav.xhtml` TOC support, with NCX fallback when nav targets cannot be resolved.
- Inline and logical Table of Contents generation for Kindle-compatible navigation.
- Fragment-level TOC extraction when valid in-spine targets exist.
- TOC labels prioritized from EPUB nav/NCX labels, then headings/titles, then body snippets/fallbacks.
- Internal EPUB links, including links to body IDs, are converted to MOBI byte-position links.
- Relative links and media honor the first XHTML `base` element with an `href`.
- Foreign spine resources use readable XHTML manifest fallbacks; fallback cycles and missing targets are rejected.
- Shared fallback documents retain distinct anchors and self-links for each spine occurrence. Links from other documents to the shared fallback path target its first occurrence.
- Supported EPUB raster images (`jpeg`, `png`, `gif`) are emitted as MOBI resources when referenced from spine content.
- Declared local raster covers are embedded even when no spine page references them, and identified through MOBI EXTH 201. EPUB3 `cover-image` declarations take precedence over EPUB2 `<meta name="cover">`; referenced cover images are reused.
- Omitted media are counted in a warning; `--report-omissions` lists affected resources, source documents, and reasons.
- Limited layout preservation for common book patterns such as centered headings, right-aligned attributions, scene breaks, and simple tables.
- Preserves explicit bold, italic, superscript (`sup`), subscript (`sub`), and underline (`u`) markup.
- PalmDOC compression (type `2`), applied per 4096-byte uncompressed text record.
- Legacy-compatible body sanitization.
- Optional USB deploy to Kindle `documents` folder (`--deploy`).


## Advanced CLI Usage

Choose an output path with `-o` or `--output`:

```bash
python3 epub2mobi.py my_book.epub -o converted.mobi
```

The output path cannot resolve to the input EPUB.

To see which media were omitted:

```bash
python3 epub2mobi.py my_book.epub --report-omissions
```

The report covers declared covers, referenced images, inline SVG, audio/video, embedded objects in spine content, fonts declared in the manifest, and remote manifest resources. Remote resources are not downloaded; local text conversion continues. The converter shows a one-line warning when it detects omissions even without the option.


## Scope and Limitations

- Output target is MOBI6 (not AZW3/KF8).
- Text-first conversion: advanced CSS, JavaScript, embedded fonts, SVG, fixed layout, and full modern EPUB styling are not preserved.
- Layout preservation is intentionally narrow and heuristic-based, not a general CSS engine.
- Logical TOCs that exceed the supported single-record size limits are omitted with a warning; the inline TOC and its links remain available.
- Tables are preserved only when they are simple rectangular structures; complex tables are flattened while retaining their text.
- Unsupported or missing images are skipped without failing the conversion.
- Covers are limited to declared local JPEG, PNG, and GIF resources. Their file signatures must match the manifest media type; image data are copied without decoding, resizing, or generating thumbnails. No cover page is inserted. Existing image and total resource size limits also apply to covers; exceeding a limit rejects conversion.
- Creator roles accept three-letter MARC codes and Library of Congress relator URLs; other schemes and values are treated as unspecified. If all creators have understood non-author roles, their credits are retained in the author field with a warning.
- The omission report does not inspect CSS background images or other visual effects defined only in stylesheets.
- XHTML is parsed as XML, preserving namespaces, CDATA, and empty elements. Standard XHTML character entities in documents declaring an external DTD are resolved locally in both text and attributes without downloading the DTD. Unknown entities, malformed XHTML, custom entity declarations, and excessive nesting are rejected.
- Flattened block elements preserve text boundaries. Omitted SVG subtrees contribute no text, styles, scripts, images, or fragment destinations.
- MOBI output uses Windows-1252; conversion warns when title or author characters cannot be represented and will be replaced in metadata.
- The first nonempty `dc:language` sets the MOBI locale. Recognized regional tags retain their region; unmapped variants fall back to the primary language with a warning. Missing or unknown languages use a neutral locale (unknown languages produce a warning).
- XML guardrails reject entity declarations across supported encodings and limit ZIP members before reading them.


## License

MIT License.
