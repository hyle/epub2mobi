# epub2mobi.py

A lightweight, zero-dependency Python script to convert EPUB files into the legacy MOBI (MOBI6) format.

This tool is designed for **simplicity** and **longevity**. It does not rely on `kindlegen`, `calibre`, or heavy external libraries. It writes the binary headers manually using the Python standard library, making it ideal for archival purposes or supporting also older Kindle devices.

## Features

* **Zero Dependencies:** Runs on any machine with Python 3.7+. No `pip install` required.
* **Legacy Support:** Forces `CP1252` encoding and simplified HTML to ensure maximum compatibility with old E-Ink devices.
* **Auto-Deploy:** Automatically detects a connected Kindle via USB and copies the file to the `documents` folder.
* **Smart Sanitization:** Strips complex CSS/JS but preserves structure (chapters, paragraphs, bold/italic, lists) for a distraction-free reading experience.

## Usage

1. **Download** the script:
```bash
wget https://raw.githubusercontent.com/hyle/epub2mobi/main/epub2mobi.py

```


2. **Run** the conversion:
```bash
python3 epub2mobi.py my_book.epub

```


*Creates `my_book.mobi` in the same directory.*
3. **Convert & Copy to Kindle:**
```bash
python3 epub2mobi.py my_book.epub --deploy

```


## Limitations

* **Text & Structure Only:** Images are intentionally stripped to keep the file size low and the code simple.
* **MOBI6 Format:** This generates the older "MobiPocket" format, not the newer AZW3/KF8 format. It does not support embedded fonts or advanced CSS.

## License

MIT License. Feel free to modify and publish.
