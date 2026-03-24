"""
docx_to_latex.py
Converts the project proposal .docx to a LaTeX .tex file,
preserving headings, paragraphs, bold/italic/underline inline runs,
bullet/numbered lists, and tables.
"""

import re
from docx import Document
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

DOCX_PATH = "1235425-W01-01-Objective Head-Related Transfer Function Evaluation Metrics Improvement for More Consistent Perceptual Assessment(Project-proposal).docx"
TEX_PATH  = "1235425-W01-01-Project-proposal.tex"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def escape_latex(text: str) -> str:
    """Escape special LaTeX characters in plain text."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&",  r"\&"),
        ("%",  r"\%"),
        ("$",  r"\$"),
        ("#",  r"\#"),
        ("_",  r"\_"),
        ("{",  r"\{"),
        ("}",  r"\}"),
        ("~",  r"\textasciitilde{}"),
        ("^",  r"\textasciicircum{}"),
    ]
    for ch, repl in replacements:
        text = text.replace(ch, repl)
    # Smart quotes → LaTeX quotes
    text = re.sub(r"\u201c", "``", text)
    text = re.sub(r"\u201d", "''", text)
    text = re.sub(r"\u2018", "`",  text)
    text = re.sub(r"\u2019", "'",  text)
    # Em-dash / en-dash
    text = text.replace("\u2014", "---")
    text = text.replace("\u2013", "--")
    # Non-breaking space
    text = text.replace("\u00a0", "~")
    return text


def run_to_latex(run) -> str:
    """Convert a single Run to LaTeX, applying bold/italic/underline."""
    text = escape_latex(run.text)
    if not text:
        return ""
    if run.bold:
        text = r"\textbf{" + text + "}"
    if run.italic:
        text = r"\textit{" + text + "}"
    if run.underline:
        text = r"\underline{" + text + "}"
    return text


def paragraph_to_latex(para) -> str:
    """Convert a paragraph's runs to a LaTeX string."""
    return "".join(run_to_latex(r) for r in para.runs)


def is_list_paragraph(para) -> bool:
    style_name = para.style.name if para.style else ""
    return "List" in style_name


def get_list_level(para) -> int:
    """Return 0-based indentation level for list items."""
    try:
        ilvl = para._p.find(qn("w:pPr"))
        if ilvl is not None:
            numPr = ilvl.find(qn("w:numPr"))
            if numPr is not None:
                ilvl_el = numPr.find(qn("w:ilvl"))
                if ilvl_el is not None:
                    return int(ilvl_el.get(qn("w:val"), 0))
    except Exception:
        pass
    return 0


def is_numbered_list(para) -> bool:
    style_name = para.style.name if para.style else ""
    return "List Number" in style_name


def table_to_latex(table) -> str:
    """Convert a docx table to a LaTeX tabular environment."""
    rows = table.rows
    if not rows:
        return ""
    ncols = len(rows[0].cells)
    col_spec = "|" + "l|" * ncols

    lines = [r"\begin{table}[h!]", r"\centering",
             r"\begin{tabular}{" + col_spec + "}", r"\hline"]
    for i, row in enumerate(rows):
        cells = []
        for cell in row.cells:
            cell_text = " ".join(
                paragraph_to_latex(p) for p in cell.paragraphs
            ).strip()
            cells.append(cell_text)
        row_str = " & ".join(cells) + r" \\"
        lines.append(row_str)
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Style → heading level mapping
# ---------------------------------------------------------------------------

HEADING_MAP = {
    "Heading 1": 1,
    "Heading 2": 2,
    "Heading 3": 3,
    "Heading 4": 4,
    "Heading 5": 5,
    "Title":     0,   # will become \title
}

SECTION_CMDS = {
    1: r"\section",
    2: r"\subsection",
    3: r"\subsubsection",
    4: r"\paragraph",
    5: r"\subparagraph",
}

# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(docx_path: str, tex_path: str):
    doc = Document(docx_path)

    title_text   = ""
    author_text  = ""
    body_lines   = []

    in_itemize    = False
    in_enumerate  = False
    list_level    = 0

    def close_lists():
        nonlocal in_itemize, in_enumerate
        if in_itemize:
            body_lines.append(r"\end{itemize}")
            in_itemize = False
        if in_enumerate:
            body_lines.append(r"\end{enumerate}")
            in_enumerate = False

    # We iterate over top-level block items (paragraphs + tables)
    from docx.oxml.ns import qn as _qn
    body_xml = doc.element.body

    for child in body_xml:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        # ---- TABLE ----
        if tag == "tbl":
            from docx.table import Table
            close_lists()
            tbl = Table(child, doc)
            body_lines.append(table_to_latex(tbl))
            continue

        # ---- PARAGRAPH ----
        if tag != "p":
            continue

        from docx.text.paragraph import Paragraph
        para = Paragraph(child, doc)
        style_name = para.style.name if para.style else "Normal"
        text_raw   = para.text.strip()

        # Skip truly empty paragraphs (no runs with content)
        if not text_raw and not para.runs:
            close_lists()
            body_lines.append("")
            continue

        # --- Heading / Title ---
        if style_name in HEADING_MAP:
            close_lists()
            level = HEADING_MAP[style_name]
            content = paragraph_to_latex(para)
            if level == 0:
                title_text = content
            elif level in SECTION_CMDS:
                body_lines.append(f"{SECTION_CMDS[level]}{{{content}}}")
            body_lines.append("")
            continue

        # --- List items ---
        if is_list_paragraph(para):
            content = paragraph_to_latex(para)
            numbered = is_numbered_list(para)
            lvl = get_list_level(para)

            if numbered:
                if not in_enumerate:
                    close_lists()
                    body_lines.append(r"\begin{enumerate}")
                    in_enumerate = True
                    in_itemize   = False
            else:
                if not in_itemize:
                    close_lists()
                    body_lines.append(r"\begin{itemize}")
                    in_itemize  = True
                    in_enumerate = False

            body_lines.append(r"  \item " + content)
            continue

        # Not a list — close any open list environment
        close_lists()

        # --- Normal paragraph ---
        content = paragraph_to_latex(para)
        if not content.strip():
            body_lines.append("")
            continue

        # Alignment hints
        align = para.alignment
        if align == WD_ALIGN_PARAGRAPH.CENTER:
            body_lines.append(r"\begin{center}")
            body_lines.append(content)
            body_lines.append(r"\end{center}")
        elif align == WD_ALIGN_PARAGRAPH.RIGHT:
            body_lines.append(r"\begin{flushright}")
            body_lines.append(content)
            body_lines.append(r"\end{flushright}")
        else:
            body_lines.append(content)
            body_lines.append("")

    # Close any trailing lists
    close_lists()

    # ---------------------------------------------------------------------------
    # Assemble LaTeX document
    # ---------------------------------------------------------------------------
    preamble = r"""\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=2.5cm]{geometry}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{enumitem}
\usepackage{parskip}
\usepackage{setspace}
\setstretch{1.15}
"""

    if title_text:
        preamble += f"\\title{{{title_text}}}\n"
        preamble += "\\date{}\n"

    preamble += "\n\\begin{document}\n"
    if title_text:
        preamble += "\\maketitle\n"

    body = "\n".join(body_lines)
    ending = "\n\\end{document}\n"

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(preamble + "\n" + body + ending)

    print(f"Done! LaTeX file written to: {tex_path}")


if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    convert(
        os.path.join(script_dir, DOCX_PATH),
        os.path.join(script_dir, TEX_PATH),
    )
