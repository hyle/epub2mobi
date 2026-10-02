# epub2mobi.py

`epub2mobi.py` is a single-script, zero-dependency Python tool that converts EPUB files to legacy MOBI6 for older Kindle devices.

Just grab the script, run it from the command line, and breathe new life into an old e-reader. It focuses on readable text, basic formatting, and working navigation.

## Install and use

Requires **Python 3.9+**. No Python packages or external converters to install.

Download the script:

```bash
curl -fL https://raw.githubusercontent.com/hyle/epub2mobi/main/epub2mobi.py -o epub2mobi.py
```

Convert an EPUB to a `.mobi` beside the input file:

```bash
python3 epub2mobi.py my_book.epub
```

Choose an output path:

```bash
python3 epub2mobi.py my_book.epub -o converted.mobi
```

Copy the result to a connected Kindle over USB:

```bash
python3 epub2mobi.py my_book.epub --deploy
```

List omitted media and the reasons:

```bash
python3 epub2mobi.py my_book.epub --report-omissions
```

Options can be combined. Use `--help` for the CLI reference.

## What it preserves

- EPUB reading order, linked notes, and readable XHTML manifest fallbacks.
- Nested EPUB3 nav or NCX tables of contents, chapter fragments, and internal links.
- Headings, scene breaks, bold, italic, underline, superscript, and subscript.
- Basic alignment and paragraph indentation through a small CSS subset.
- Simple tables; nested tables and cells spanning rows or columns are flattened while retaining text.
- Referenced local JPEG, PNG, and GIF images, plus declared raster covers.
- Title, author credits, and language metadata.

Output uses PalmDOC compression and targets **MOBI6**. Advanced CSS, fixed layout, JavaScript, embedded fonts, SVG, audio, video, and AZW3/KF8 output are unsupported. Remote resources are never downloaded.

Missing or unsupported media are skipped and reported. Conversion shows an omission count; `--report-omissions` adds resource paths, source documents, and reasons. CSS background assets are not inspected.

## Navigation and metadata

EPUB3 navigation is preferred. A more complete NCX can replace an incomplete nav; a spine-based TOC is generated only when no authored destinations resolve. Unresolved destinations produce warnings while valid entries and their hierarchy remain. Children of skipped entries attach to the nearest retained ancestor.

Books with multiple TOC entries also get a linked TOC at the end, with a guide link for readers that use it for navigation. If the logical MOBI TOC exceeds its supported record size, it is omitted with a warning and the in-book TOC remains.

The first nonempty Dublin Core title is used. Creator order is preserved, with names joined by `; `. Known EPUB2/EPUB3 MARC roles select authors and creators without a recognized role. Unknown roles retain their credits; if all creators have recognized non-author roles, all are retained with a warning. The recognized role list is bundled.

The first nonempty language sets the MOBI locale. Recognized regions are retained; unmapped variants fall back to the primary language with a warning. Missing or unknown languages use a neutral locale, with a warning for unknown values.

The MOBI header declares Windows-1252. Body characters outside that encoding use numeric character references; title and author metadata replace unrepresentable characters with `?` and produce a warning. Generated HTML omits a redundant charset declaration so readers can reserialize decoded text without retaining a conflicting encoding hint.

## Supported formatting

Styles come from XHTML `<style>` blocks, local UTF-8 stylesheets (an optional BOM is accepted), and inline `style` attributes. Selectors are single ASCII element names (`p`) or class names (`.italic`); class names are case-sensitive. Comma-separated lists work when every selector is supported.

| Property | Supported values |
| --- | --- |
| `font-style` | `normal`, `italic`, `oblique` (rendered as italic) |
| `font-weight` | `normal`, `bold`, `100`–`900` (600 and above render as bold) |
| `text-align` | `left`, `right`, `center`, `justify` |
| `text-indent` | `0`, nonnegative lengths in `em` or `pt` |

Precedence is **inline style > class rule > element rule**, separately for each property. Later declarations win ties in document order. Properties inherit; explicit `inherit` is supported. Semantic tags supply defaults that CSS can override, including resets to normal weight or style. Headings and table headers may become ordinary paragraphs or cells to allow those resets.

Common class/ID hints and valid inline auto margins provide alignment when no supported CSS alignment applies. Compound, descendant, ID, and pseudo selectors, other properties, `!important`, and at-rule blocks such as `@import` and `@media` are skipped. Alternate or disabled stylesheets, and those whose `media` is neither empty nor `all`, are skipped. Missing, remote, or non-UTF-8 stylesheets are reported.

`float` is inspected only to repair detached text drop caps. A left-floating initial can join the immediately following paragraph when its continuation starts with a lowercase or uncased letter and the structure makes the join unambiguous. Formatting, accents, and link targets are preserved; the initial becomes ordinary inline text. Explicit spaces at the join, including nonbreaking spaces, prevent joining; XML newline indentation may be removed when joining is safe. Image initials and CSS-generated letters are not reconstructed, and general float layout is unsupported.

Covers must be declared local JPEG, PNG, or GIF resources with matching file signatures. EPUB3 `cover-image` takes precedence over EPUB2 cover metadata. Image data are copied without resizing or thumbnail generation; no cover page is inserted.

## Input and output handling

Relative links and media honor the first XHTML `base` with an `href`, including URL-encoded paths. Parent references within the EPUB are allowed; traversal above the archive root is rejected, including encoded and backslash forms. Invalid required package or spine paths stop conversion; invalid optional links and resources are skipped and reported. Repeated spine documents retain independent formatting and self-links; links from other documents to a shared path target its first occurrence.

XHTML is parsed as XML, preserving namespaces, CDATA, and empty elements. Standard XHTML entities in documents declaring an external DTD are resolved locally in text and attributes. Malformed XML, unknown entities, custom entity declarations, and processing-limit violations are rejected. No external DTD is fetched.

Conversion and USB deployment write to a temporary file beside the destination, then replace it atomically after flushing and closing. Failed writes preserve existing output; temporary files are cleaned up. Existing permissions and symlink targets are preserved; new files use private temporary-file permissions. The output path cannot resolve to the input EPUB.

Expected input, conversion, archive, and file errors return a nonzero CLI status with a concise message. Unexpected programming errors retain their traceback. If deployment fails, the completed local MOBI remains available; if no Kindle is detected, conversion succeeds with a warning. Library callers can catch `ConversionError` (a `ValueError` subclass); filesystem and ZIP validation errors retain their standard types.

## Resource limits

Size and processing limits keep malformed or unusually large books from consuming excessive resources. Exceeding a processing budget stops conversion, including during optional TOC processing. These limits do not impose a wall-clock timeout.

<details>
<summary>Default limits and accounting</summary>

| Resource | Limit |
| --- | --- |
| Archive file | 512 MiB |
| ZIP entries | 10,000, including unused and duplicate entries |
| Total declared uncompressed ZIP data | 1 GiB |
| Unique physical members read | 256 MiB combined |
| Archive read work | 512 MiB of declared compressed + uncompressed bytes |
| XML / spine XHTML document | 8 MiB / 16 MiB |
| XML structure per document | 100,000 elements, 200,000 attributes, depth 256 (root depth zero) |
| XML attribute names and values | 65,536 characters each; namespace declarations count as attributes |
| Manifest / spine / each navigation document | 10,000 entries each, before filtering duplicates or invalid entries |
| Spine processing | 64 MiB of source data and 1,000,000 elements, counting every occurrence |
| Image or cover | 64 MiB each |
| Style block or stylesheet | 1 MiB, 10,000 top-level rules, 50,000 selectors |
| Combined style blocks and linked stylesheets per document | 4 MiB of source data, 50,000 supported selector entries |
| CSS processing across the book | 256 MiB of source data, 5,000,000 supported selector entries |
| Linked stylesheet cache | 50,000 supported selector entries |
| Generated HTML | 64 MiB in MOBI encoding, including character references, link positions, and the footer TOC |

Only stored and DEFLATE ZIP entries are accepted. Archive size is checked before opening the ZIP directory; entry count, compression methods, and total declared content are checked after directory loading and before member reads.

Unique-member accounting survives cache eviction. Read work charges failed reads and rereads; cache hits are free. Spine and CSS processing count repeated references even when cached. Each resource use still enforces its size limit.

XML structure is checked before tree construction, including locally resolved XHTML attributes. The combined length of namespace-expanded element and attribute names is bounded by the document's XML byte limit. HTML is checked during emission and before compression.

The byte cache holds at most 8 MiB and 128 entries. A separate parsed XML/XHTML cache holds at most 8 MiB of source data, 65,536 elements, and 128 entries. Larger documents within the conversion limits are processed without caching. Cached spine trees are copied before editing.

</details>

## License

MIT License.
