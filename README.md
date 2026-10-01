# epub2mobi.py

`epub2mobi.py` is a zero-dependency Python tool that converts EPUB files to legacy MOBI6 for older Kindle devices.

Just grab the script, run it from the command line, and breathe new life into an old e-reader.


## Requirements

- Python 3.9+


## Install and simple CLI usage

Download the script:

```bash
curl -fL https://raw.githubusercontent.com/hyle/epub2mobi/main/epub2mobi.py -o epub2mobi.py
```

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
- A small CSS subset preserves bold, italic, alignment, and paragraph indentation from local stylesheets, XHTML style blocks, and inline styles.
- Detached text drop caps are joined to their following paragraph when their structure and authored float styles identify an unambiguous initial.
- Preserves explicit bold, italic, superscript (`sup`), subscript (`sub`), and underline (`u`) markup.
- PalmDOC compression (type `2`), applied per 4096-byte uncompressed text record.
- Legacy-compatible body sanitization.
- Optional USB deploy to Kindle `documents` folder (`--deploy`).


## Advanced CLI usage

Choose an output path with `-o` or `--output`:

```bash
python3 epub2mobi.py my_book.epub -o converted.mobi
```

The output path cannot resolve to the input EPUB.

Output is written to a temporary file in the destination directory, flushed and closed, then replaced atomically. Failed writes leave existing output intact and remove the temporary file. Existing output permissions and symlink targets are preserved; new files use the temporary file's private permissions.

To see which media were omitted:

```bash
python3 epub2mobi.py my_book.epub --report-omissions
```

The report covers declared covers, referenced images, inline SVG, audio/video, embedded objects in spine content, fonts declared in the manifest, remote manifest resources, and missing, remote, or incorrectly encoded linked stylesheets. Remote resources are not downloaded; local text conversion continues. The converter shows a one-line warning when it detects omissions even without the option.


## Supported CSS

Styles come from XHTML `<style>` blocks, local `<link rel="stylesheet">` resources, and inline `style` attributes. Linked stylesheets must use UTF-8 (an optional BOM is accepted); relative paths honor the document's `base` URL. Stylesheets are read once per resource and limited to 1 MiB each, within the shared content size budget. Oversized stylesheets reject conversion. Missing, remote, or non-UTF-8 stylesheets are skipped and reported.

Selectors are single element names (`p`) or single class names (`.italic`), using ASCII identifiers. Class names are case-sensitive. Comma-separated lists are supported when every selector in the list is supported.

| Property | Supported values | MOBI output |
| --- | --- | --- |
| `font-style` | `normal`, `italic`, `oblique` | Italic markup; `oblique` becomes italic |
| `font-weight` | `normal`, `bold`, `100` through `900` in steps of 100 | Bold markup for weights 600–900 |
| `text-align` | `left`, `right`, `center`, `justify` | Block `align` attribute |
| `text-indent` | `0`, nonnegative lengths in `em` or `pt` | `width` attribute on paragraphs, divisions, blockquotes, and headings |

Precedence is resolved separately for each property: inline declarations override class rules, which override element rules. Later declarations win ties, with style blocks and linked stylesheets processed in document order. Properties inherit from their parent; `inherit` is also supported explicitly. Semantic tags provide local defaults that CSS can override. Alignment hints from class/ID names or auto margins are used only when no authored alignment applies.

Normal values reset inherited bold/italic formatting. A heading containing a normal-weight reset becomes a paragraph so MOBI's built-in heading bold cannot override it; other text retains its resolved bold formatting. Table header cells containing such resets become ordinary cells.

`float` is inspected only to recognize detached text drop caps, with the same selector and declaration precedence. It does not inherit unless explicitly set to `inherit`; `none`, `initial`, and `unset` clear the hint. A left-floating initial in text-only `div`, `p`, or inline wrappers can be moved into the immediately following prose paragraph. Nested wrappers, an opening quotation mark, combining accents, inline formatting, and fragment/link targets are preserved. The initial becomes ordinary inline text; its oversized decorative layout is discarded.

Normalization requires a lowercase or uncased letter at the start of the continuation. Right floats, substantial text, images, intervening content, uppercase or punctuated continuations, and leading spaces that could separate words are left unchanged. XML newline indentation around the initial and continuation is removed when joining them. Existing initials already inline in a paragraph are preserved. Image-based initials and CSS-generated letters are not reconstructed; general float layout remains unsupported.

Other properties, unsupported values, compound/descendant/ID/pseudo selectors, and entire at-rule blocks are skipped. `@import`, `@media`, and `!important` are unsupported. Stylesheets with a `media` attribute other than empty or `all`, alternate stylesheets, and disabled stylesheets are skipped. This subset does not implement the full CSS cascade or modern EPUB layout.


## Scope and limitations

- Output target is MOBI6 (not AZW3/KF8).
- Text-first conversion: advanced CSS, JavaScript, embedded fonts, SVG, fixed layout, and full modern EPUB styling are not preserved.
- Layout preservation uses the CSS subset above and a few common class/ID hints; complex CSS layout is not supported.
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
