from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape
import re


BASE_DIR = Path(__file__).resolve().parent
README_PATH = BASE_DIR / "README.md"
OUTPUT_PATH = BASE_DIR / "EMIPredict_AI_Project_Documentation.docx"


def markdown_to_paragraphs(text: str):
    lines = text.splitlines()
    paragraphs = []
    in_code = False

    for line in lines:
        stripped = line.rstrip()

        if stripped.startswith("```"):
            in_code = not in_code
            continue

        if in_code:
            if stripped:
                paragraphs.append(("code", stripped))
            else:
                paragraphs.append(("blank", ""))
            continue

        if not stripped:
            paragraphs.append(("blank", ""))
            continue

        if stripped.startswith("# "):
            paragraphs.append(("title", stripped[2:].strip()))
            continue

        if stripped.startswith("## "):
            paragraphs.append(("heading1", stripped[3:].strip()))
            continue

        if stripped.startswith("### "):
            paragraphs.append(("heading2", stripped[4:].strip()))
            continue

        match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if match:
            paragraphs.append(("number", match.group(1).strip()))
            continue

        if stripped.startswith("- "):
            paragraphs.append(("bullet", stripped[2:].strip()))
            continue

        paragraphs.append(("body", stripped))

    return paragraphs


def clean_inline_markup(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("`", "")
    return text


def paragraph_xml(style: str, text: str) -> str:
    text = clean_inline_markup(text)
    escaped = escape(text)

    style_map = {
        "title": "Title",
        "heading1": "Heading1",
        "heading2": "Heading2",
        "code": "NoSpacing",
    }

    ppr = []
    if style in style_map:
        ppr.append(f"<w:pStyle w:val=\"{style_map[style]}\"/>")

    if style == "bullet":
        ppr.append("<w:ind w:left=\"720\" w:hanging=\"360\"/>")
    if style == "number":
        ppr.append("<w:ind w:left=\"720\" w:hanging=\"360\"/>")

    if style == "bullet":
        run_text = f"- {escaped}"
    elif style == "number":
        run_text = f"1. {escaped}"
    else:
        run_text = escaped

    run_props = ""
    if style == "code":
        run_props = (
            "<w:rPr>"
            "<w:rFonts w:ascii=\"Consolas\" w:hAnsi=\"Consolas\"/>"
            "<w:sz w:val=\"20\"/>"
            "</w:rPr>"
        )

    if style == "blank":
        return "<w:p/>"

    ppr_xml = f"<w:pPr>{''.join(ppr)}</w:pPr>" if ppr else ""
    return (
        "<w:p>"
        f"{ppr_xml}"
        "<w:r>"
        f"{run_props}"
        f"<w:t xml:space=\"preserve\">{run_text}</w:t>"
        "</w:r>"
        "</w:p>"
    )


def build_document_xml(paragraphs):
    body = "".join(paragraph_xml(style, text) for style, text in paragraphs)
    section = (
        "<w:sectPr>"
        "<w:pgSz w:w=\"12240\" w:h=\"15840\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" "
        "w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
        "</w:sectPr>"
    )
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:wpc=\"http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas\" "
        "xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" "
        "xmlns:o=\"urn:schemas-microsoft-com:office:office\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:m=\"http://schemas.openxmlformats.org/officeDocument/2006/math\" "
        "xmlns:v=\"urn:schemas-microsoft-com:vml\" "
        "xmlns:wp14=\"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing\" "
        "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
        "xmlns:w10=\"urn:schemas-microsoft-com:office:word\" "
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:w14=\"http://schemas.microsoft.com/office/word/2010/wordml\" "
        "xmlns:wpg=\"http://schemas.microsoft.com/office/word/2010/wordprocessingGroup\" "
        "xmlns:wpi=\"http://schemas.microsoft.com/office/word/2010/wordprocessingInk\" "
        "xmlns:wne=\"http://schemas.microsoft.com/office/word/2006/wordml\" "
        "xmlns:wps=\"http://schemas.microsoft.com/office/word/2010/wordprocessingShape\" "
        "mc:Ignorable=\"w14 wp14\">"
        f"<w:body>{body}{section}</w:body>"
        "</w:document>"
    )


def content_types_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""


def rels_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def document_rels_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""


def styles_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:rPr>
      <w:b/>
      <w:sz w:val="32"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:rPr>
      <w:b/>
      <w:sz w:val="28"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:rPr>
      <w:b/>
      <w:sz w:val="24"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="NoSpacing">
    <w:name w:val="No Spacing"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:rPr>
      <w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>
      <w:sz w:val="20"/>
    </w:rPr>
  </w:style>
</w:styles>
"""


def core_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>EMIPredict AI Project Documentation</dc:title>
  <dc:creator>OpenAI Codex</dc:creator>
  <cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
</cp:coreProperties>
"""


def app_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office Word</Application>
</Properties>
"""


def main():
    text = README_PATH.read_text(encoding="utf-8")
    paragraphs = markdown_to_paragraphs(text)
    document_xml = build_document_xml(paragraphs)

    with ZipFile(OUTPUT_PATH, "w", compression=ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types_xml())
        docx.writestr("_rels/.rels", rels_xml())
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/_rels/document.xml.rels", document_rels_xml())
        docx.writestr("word/styles.xml", styles_xml())
        docx.writestr("docProps/core.xml", core_xml())
        docx.writestr("docProps/app.xml", app_xml())

    print(f"Created: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
