"""Minimal Markdown -> DOCX converter that creates a basic .docx
without external dependencies.

It produces a plain Word document by generating the required XML parts
and zipping them into a .docx archive. It supports headings (#, ##, ###),
bullet lists starting with '- ', and regular paragraphs.
"""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import xml.sax.saxutils as sax
from datetime import datetime


def xmlesc(s: str) -> str:
    return sax.escape(s)


def paragraph_xml(text: str) -> str:
    return f"""
      <w:p>
        <w:r>
          <w:t xml:space="preserve">{xmlesc(text)}</w:t>
        </w:r>
      </w:p>
"""


def heading_xml(text: str, level: int) -> str:
    # represent heading as a bold paragraph (no styles required)
    return f"""
      <w:p>
        <w:r>
          <w:rPr><w:b/></w:rPr>
          <w:t xml:space="preserve">{xmlesc(text)}</w:t>
        </w:r>
      </w:p>
"""


def bullet_xml(text: str) -> str:
    # simple bullet using a normal paragraph (Word will not show numbering without numbering part)
    return paragraph_xml(f"• {text}")


def build_document_xml(lines: list[str]) -> str:
    body = []
    in_code = False
    for raw in lines:
        line = raw.rstrip("\n")
        s = line.strip()
        if not s:
            continue
        if s.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            body.append(paragraph_xml(line))
            continue
        if s.startswith("#"):
            hashes = 0
            for ch in s:
                if ch == "#":
                    hashes += 1
                else:
                    break
            title = s[hashes:].strip()
            body.append(heading_xml(title, min(hashes, 4)))
            continue
        if s.startswith("- "):
            body.append(bullet_xml(s[2:].strip()))
            continue
        # fallback: plain paragraph
        body.append(paragraph_xml(s))

    body_xml = "\n".join(body)
    doc = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    doc += '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
    doc += "  <w:body>\n"
    doc += body_xml
    doc += "    <w:sectPr/>\n"
    doc += "  </w:body>\n"
    doc += "</w:document>"
    return doc


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

CORE_PROPS = lambda title: f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{xmlesc(title)}</dc:title>
  <dc:creator>Auto-convert</dc:creator>
  <dcterms:created xsi:type="dcterms:W3CDTF">{datetime.utcnow().isoformat()}Z</dcterms:created>
</cp:coreProperties>"""

APP_PROPS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
  <TotalTime>0</TotalTime>
</Properties>"""


def convert(md_path: Path, docx_path: Path):
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    document_xml = build_document_xml(lines)

    # create archive
    with ZipFile(docx_path, "w", ZIP_DEFLATED) as z:
        # content types
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        # root rels
        z.writestr("_rels/.rels", ROOT_RELS)
        # doc props
        z.writestr("docProps/core.xml", CORE_PROPS(md_path.stem))
        z.writestr("docProps/app.xml", APP_PROPS)
        # word document
        z.writestr("word/document.xml", document_xml)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    md = root / "reports" / "model_report.md"
    out = root / "reports" / "model_report.docx"
    if not md.exists():
        print("Markdown report not found:", md)
    else:
        convert(md, out)
        print("Wrote", out)
