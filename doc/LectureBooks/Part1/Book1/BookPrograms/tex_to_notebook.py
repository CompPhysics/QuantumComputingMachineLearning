"""Generate the per-chapter Jupyter notebooks of the quantum-computing book
from the LaTeX chapters (doc/Book/chapterN.tex) and the chapter programs.

    python3 BookPrograms/tex_to_notebook.py Notebooks          # all chapters
    python3 BookPrograms/tex_to_notebook.py Notebooks 2,7      # a subset

Ported from the machine-learning book (doc/BookML/BookPrograms/tex_to_notebook.py).
Differences: the chapter title and abstract are read from book.tex (the
chapter files carry no \chapter line), the \be/\ee-style shorthands of
book.tex are expanded before parsing, the output is
Notebooks/chapterN/chapterN.ipynb, every notebook starts with a bootstrap
cell that puts BookPrograms/chapter*/ and BookPrograms/common/ on sys.path,
and TikZ figures are rendered to BookFigures/chapterN/<label>.png by
BookPrograms/render_tikz.py.  The notebooks are generated: never edit them
by hand.
"""
import re, json, os, sys, ast, uuid

# doc/Book, derived from this script's own location (BookPrograms/..) so
# that the script runs unchanged wherever the repository is checked out.
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHAPTERS = list(range(1, 12))

# Appendices are named rather than numbered; the second entry is the letter
# used when numbering their equations, tables and figures.
EXTRA = [("appendix", "B")]

# every source the generator knows about, as (key, number) pairs
SOURCES = [(c, c) for c in CHAPTERS] + EXTRA

# BookFigures subdirectory per source, used for TikZ schematics that have no
# \includegraphics and are rendered separately by BookPrograms/render_tikz.py.
DIRS = {c: f"chapter{c}" for c in CHAPTERS}
DIRS["appendix"] = "appendixB"

# shorthands defined in book.tex, expanded before any parsing
SHORTHANDS = [(r"\\be\b", r"\\begin{equation}"), (r"\\ee\b", r"\\end{equation}"),
              (r"\\beN\b", r"\\begin{equation*}"), (r"\\eeN\b", r"\\end{equation*}"),
              (r"\\bea\b", r"\\begin{eqnarray}"), (r"\\eea\b", r"\\end{eqnarray}"),
              (r"\\beaN\b", r"\\begin{eqnarray*}"), (r"\\eeaN\b", r"\\end{eqnarray*}"),
              (r"\\bdm\b", r"\\begin{displaymath}"), (r"\\edm\b", r"\\end{displaymath}"),
              (r"\\bsubeqs\b", r"\\begin{subequations}"), (r"\\esubeqs\b", r"\\end{subequations}")]

MATH_ENVS = ("equation","equation*","align","align*","eqnarray","eqnarray*")

PREAMBLE = r"""$$
\newcommand{\bm}[1]{\boldsymbol{#1}}
\newcommand{\matr}[1]{\boldsymbol{#1}}
\newcommand{\bfv}[1]{\boldsymbol{#1}}
\renewcommand{\vec}[1]{\boldsymbol{#1}}
\newcommand{\uvec}[1]{\boldsymbol{\hat{#1}}}
\newcommand{\op}[1]{\widehat{#1}}
\newcommand{\Op}[1]{\boldsymbol{\widehat{#1}}}
\newcommand{\OP}[1]{\boldsymbol{\widehat{#1}}}
\newcommand{\Det}[1]{|\boldsymbol{#1}|}
\newcommand{\bigO}{\mathcal{O}}
\newcommand{\var}{\mathrm{Var}}
\newcommand{\cov}{\mathrm{Cov}}
\newcommand{\real}[1]{\mathrm{Re}\left\{#1\right\}}
\newcommand{\ud}[1]{d#1}
\newcommand{\udd}[1]{d^2#1}
\newcommand{\mean}[1]{\langle #1 \rangle}
\newcommand{\overlap}[2]{\langle #1 | #2 \rangle}
\newcommand{\bracket}[3]{\langle #1 | #2 | #3 \rangle}
\newcommand{\projection}[2]{| #1 \rangle \langle #2 |}
\newcommand{\brab}[1]{\big\langle #1 \big|}
\newcommand{\ketb}[1]{\big| #1 \big\rangle}
\newcommand{\bket}[1]{\vert #1\rangle}
\newcommand{\bbra}[1]{\langle #1\vert}
\newcommand{\ketbra}[2]{\vert #1\rangle\langle #2\vert}
\newcommand{\braket}[2]{\langle #1\vert #2\rangle}
$$"""

def strip_comments(s):
    """Strip LaTeX %-comments, but never inside a verbatim listing, where a
    bare % is legitimate Python (a format specifier, a modulo, an f-string)."""
    out=[]; in_code=False
    for line in s.split("\n"):
        if re.match(r"\s*\\begin\{(Python|C\+\+)\}", line):
            in_code=True; out.append(line); continue
        if re.match(r"\s*\\end\{(Python|C\+\+)\}", line):
            in_code=False; out.append(line); continue
        if in_code:
            out.append(line); continue
        i=0; res=""
        while i < len(line):
            if line[i]=="%" and (i==0 or line[i-1]!="\\"):
                break
            res+=line[i]; i+=1
        out.append(res)
    return "\n".join(out)

def read(key):
    """Read a source by key: an int for chapterN.tex, a string for a named
    file such as appendix.tex (whose \\input{linalg} is followed)."""
    stem = f"chapter{key}" if isinstance(key, int) else key
    s = strip_comments(open(f"{SRC}/{stem}.tex").read())
    # follow one level of \input{file}
    s = re.sub(r"\\input\{([^}]*)\}",
               lambda m: strip_comments(open(f"{SRC}/{m.group(1)}.tex").read()), s)
    for a, b in SHORTHANDS:
        s = re.sub(a, b, s)
    return s

def _book_chapters():
    """(title, abstract) for every \\chapter of book.tex, main matter first,
    then the appendices; the author assignment after a comma is stripped."""
    book = strip_comments(open(f"{SRC}/book.tex").read())
    main, app = book.split("\\appendix", 1)
    out = {}
    for part, keys in ((main, CHAPTERS), (app, ["A", "B"])):
        found = re.findall(r"\\chapter\{(.*?)\}\s*\\abstract\{(.*?)\n\}", part, re.S)
        for key, (title, abstract) in zip(keys, found):
            title = re.sub(r",\s*(all|[A-Z][^,]*)(\+\+|\?)?\s*$", "", title).strip()
            out[key] = (title, abstract.strip())
    return out

BOOK = _book_chapters()

def chapter_title(s, key):
    """Chapter title from book.tex; the chapter file has no \\chapter line."""
    num = key if isinstance(key, int) else dict(SOURCES)[key]
    title = BOOK.get(num, (str(key), ""))[0]
    return title, None

# ---------------------------------------------------------------- pass 1
def build_maps():
    eqnum, secname, chapname, tabnum, fignum, thmnum = {}, {}, {}, {}, {}, {}
    for key, num in SOURCES:
        if not os.path.exists(f"{SRC}/{'chapter'+str(key) if isinstance(key,int) else key}.tex"):
            continue
        s = read(key)
        title, _ = chapter_title(s, key)
        # exercises: \item \label{ex:...} in document order
        for k, lm in enumerate(re.finditer(r"\\item\s*\\label\{(ex:[^}]*)\}", s)):
            thmnum[lm.group(1)] = f"{num}.{k+1}"
        # every chap: label, not only the first -- Chapter 7 carries both
        # \label{chap:trees} and \label{chap:ensemble}, and both are referenced.
        for lm in re.finditer(r"\\label\{(chap:[^}]*)\}", s):
            chapname[lm.group(1)] = (num, title)
        # section / subsection labels -> title
        for sm in re.finditer(r"\\(sub)?section\*?\{(.*?)\}\s*\n\s*\\label\{([^}]*)\}", s, re.S):
            secname[sm.group(3)] = sm.group(2).strip()
        if num is None:
            continue          # unnumbered source: nothing to number
        # table numbering
        tn = 0
        for tm2 in re.finditer(r"\\begin\{table\}(.*?)\\end\{table\}", s, re.S):
            tn += 1
            for l in re.findall(r"\\label\{(tab:[^}]*)\}", tm2.group(1)): tabnum[l] = f"{num}.{tn}"
        # figure numbering, in document order
        fn = 0
        for fm2 in re.finditer(r"\\begin\{figure\}(.*?)\\end\{figure\}", s, re.S):
            fn += 1
            for l in re.findall(r"\\label\{(fig:[^}]*)\}", fm2.group(1)): fignum[l] = f"{num}.{fn}"
        # theorem-like environments.  book.tex loads svmono with envcountsame
        # and envcountchap, so all six share a single counter per chapter.
        tk = 0
        for tm3 in re.finditer(r"\\begin\{(theorem|proposition|lemma|corollary|definition|example)\}"
                               r"(.*?)\\end\{\1\}", s, re.S):
            tk += 1
            for l in re.findall(r"\\label\{([^}]*)\}", tm3.group(2)): thmnum[l] = f"{num}.{tk}"
        # equation numbering, in document order
        n = 0
        for em in re.finditer(r"\\begin\{(equation|align|eqnarray)\}(.*?)\\end\{\1\}", s, re.S):
            body = em.group(2)
            labels = re.findall(r"\\label\{([^}]*)\}", body)
            if em.group(1) in ("align","eqnarray"):
                # one number per \\ separated row that carries a label
                rows = body.split(r"\\")
                for r in rows:
                    ls = re.findall(r"\\label\{([^}]*)\}", r)
                    if "\\nonumber" in r: continue
                    n += 1
                    for l in ls: eqnum[l] = f"{num}.{n}"
            else:
                n += 1
                for l in labels: eqnum[l] = f"{num}.{n}"
    return eqnum, secname, chapname, tabnum, fignum, thmnum

EQ, SEC, CHAP, TAB, FIG, THM = build_maps()

def resolve_ref(label, mathmode=False):
    """Return the bare reference text; the source already supplies
    'Eq.~(...)', 'Chapter~', 'Section~' and so on.  Inside math a section
    title is set plain, since markdown emphasis would be shown literally."""
    if label in EQ:   return EQ[label]
    if label in TAB:  return TAB[label]
    if label in FIG:  return FIG[label]
    if label in THM:  return THM[label]
    if label in CHAP:
        num, title = CHAP[label]
        # the introduction has no number; name it instead
        if num is not None: return str(num)
        return title if mathmode else f"*{title}*"
    if label in SEC:  return SEC[label] if mathmode else f"*{SEC[label]}*"
    return label

def resolve_refs_in_math(text):
    """Cross-references also occur inside display math, in \\text{...} labels."""
    return re.sub(r"\\ref\{([^}]*)\}",
                  lambda m: resolve_ref(m.group(1), True), text)

# ---------------------------------------------------------------- inline
def inline(s):
    _st = []
    def _stash(m):
        _st.append(m.group(0)); return f"\x01M{len(_st)-1}\x01"
    s = re.sub(r"\$\$.*?\$\$", _stash, s, flags=re.S)
    s = re.sub(r"\\begin\{equation\*\}.*?\\end\{equation\*\}", _stash, s, flags=re.S)
    s = re.sub(r"\$(?:[^$\n]|\n(?!\n))*?\$", _stash, s)
    s = re.sub(r"\\(?:Eq|Section|Chapter|Table|Figure)?~?\\?ref\{([^}]*)\}",
               lambda m: resolve_ref(m.group(1)), s)
    s = re.sub(r"\\ref\{([^}]*)\}", lambda m: resolve_ref(m.group(1)), s)
    s = re.sub(r"\\cite\{([^}]*)\}", lambda m: f"[{m.group(1)}]", s)
    s = re.sub(r"\\index\{[^}]*\}", "", s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\emph\{(.*?)\}", r"*\1*", s, flags=re.S)
    s = re.sub(r"\\textbf\{(.*?)\}", r"**\1**", s, flags=re.S)
    s = re.sub(r"\\textit\{(.*?)\}", r"*\1*", s, flags=re.S)
    s = re.sub(r"\\texttt\{(.*?)\}", lambda m: "`"+m.group(1).replace("\\_","_").replace("\\","")+"`", s, flags=re.S)
    s = re.sub(r"\\href\{([^}]*)\}\{([^}]*)\}", r"[\2](\1)", s)
    s = re.sub(r"\\url\{([^}]*)\}", r"<\1>", s)
    # a footnote whose whole content is a URL becomes a link on the word it is
    # attached to, which reads far better on screen than a parenthetical URL
    s = re.sub(r"([\w.\-)\]]+)([,.;:]?)\s*\\footnote\{<([^>]*)>\}",
               lambda m: f"[{m.group(1)}]({m.group(3)}){m.group(2)}", s)
    s = re.sub(r"\\footnote\{(.*?)\}", r" (\1)", s, flags=re.S)
    s = s.replace("\\%","%").replace("\\&","&").replace("\\#","#")
    s = re.sub(r"(?<!`)``(?!`)", '"', s)
    s = s.replace("''", '"')
    # text-mode LaTeX that legitimately appears outside math
    for a,b in [(r"\\ldots","..."), (r"\\dots","..."), (r"\\cdots","..."),
                (r"\\times"," x "), (r"\\ ", " "), (r"\\,", " "), (r"\\;", " "),
                (r"\\quad","  "), (r"\\qquad","    "), (r"\\hspace\{[^}]*\}"," "),
                (r"\\vspace\{[^}]*\}",""), (r"\\smallskip",""), (r"\\medskip",""),
                (r"\\bigskip",""), (r"\\/",""), (r"\\@","")]:
        s = re.sub(a, b, s)
    s = re.sub(r"\\-", "", s)
    s = s.replace("~"," ")
    s = re.sub(r"\\\\(?![a-zA-Z])", "  \n", s)
    s = re.sub(r"\x01M(\d+)\x01",
               lambda m: resolve_refs_in_math(_st[int(m.group(1))]), s)
    return s

MULTICOL = re.compile(r"\\multicolumn\{(\d+)\}\{.*?\}\{(.*)\}\s*$", re.S)

def tabular_to_md(body):
    body = re.sub(r"\\(top|mid|bottom)rule","",body)
    body = re.sub(r"\\addlinespace(\[[^\]]*\])?","",body)
    body = re.sub(r"\\cmidrule(\([^)]*\))?(\[[^\]]*\])?(\{[^}]*\})?","",body)
    body = re.sub(r"\\hline","",body)
    body = re.sub(r"\\noalign\{[^}]*\}","",body)
    raw=[r.strip() for r in body.split(r"\\") if r.strip()]
    # a row is either a \multicolumn banner spanning the table, or ordinary cells
    rows=[]
    for r in raw:
        m = MULTICOL.match(r)
        if m and "&" not in r:
            rows.append((int(m.group(1)), [m.group(2)]))
        else:
            rows.append((None, [c.strip() for c in r.split("&")]))
    ncols = max([len(c) for span,c in rows if span is None] or [1])
    out=[]
    for i,(span,cells) in enumerate(rows):
        # a markdown cell must be a single line, so fold the LaTeX line breaks
        cells = [re.sub(r"\s+", " ", inline(c)).strip() for c in cells]
        cells += [""]*(ncols-len(cells))            # pad banners and short rows
        out.append("| "+" | ".join(cells[:ncols])+" |")
        if i==0: out.append("|"+"|".join(["---"]*ncols)+"|")
    return "\n".join(out)

def fignumber(inner):
    """'Figure 17.3: ' for a figure environment whose label we numbered."""
    lm = re.search(r"\\label\{(fig:[^}]*)\}", inner)
    return f"Figure {FIG[lm.group(1)]}: " if lm and lm.group(1) in FIG else ""

def tabnumber(inner):
    lm = re.search(r"\\label\{(tab:[^}]*)\}", inner)
    return f"Table {TAB[lm.group(1)]}: " if lm and lm.group(1) in TAB else ""

def convert_prose(s, chnum):
    s = re.sub(r"\\index\{(?:[^{}]|\{[^{}]*\})*\}", "", s)

    # figures -> markdown image + italic caption.  The .pdf used by LaTeX has a
    # .png twin written by the same BookFigures script; point at the .png and
    # make the path relative to doc/LectureNotes.
    def fig_repl(m):
        inner = m.group(1)
        gm = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", inner)
        cm = re.search(r"\\caption\{(.*?)\}\s*(?:\\label|\\end)", inner, re.S)
        if not gm:
            # A TikZ schematic rather than a generated plot.  These are rendered
            # to PNG by BookFigures/render_tikz.py and named after their label.
            if "tikzpicture" in inner and DIRS.get(chnum):
                lm = re.search(r"\\label\{fig:([A-Za-z0-9_-]+)\}", inner)
                if lm:
                    cm2 = re.search(r"\\caption\{(.*?)\}\s*\\label", inner, re.S)
                    cap2 = inline(cm2.group(1)).strip().replace("\n", " ") if cm2 else ""
                    alt2 = re.sub(r"[^A-Za-z0-9 ,.-]", "",
                                  re.sub(r"\$[^$]*\$", "", cap2)).strip()[:70]
                    p = f"../../BookFigures/{DIRS[chnum]}/{lm.group(1)}.png"
                    return f"\n\n![{alt2}]({p})\n\n*{fignumber(inner)}{cap2}*\n\n"
            return ""
        path = "../../" + gm.group(1) + ".png"
        cap = inline(cm.group(1)).strip().replace("\n", " ") if cm else ""
        alt = re.sub(r"\\$[^$]*\\$", "", cap)          # no math in the alt text
        alt = re.sub(r"[^A-Za-z0-9 ,.-]", "", alt).strip()[:70]
        return f"\n\n![{alt}]({path})\n\n*{fignumber(inner)}{cap}*\n\n"
    s = re.sub(r"\\begin\{figure\}\[?[^\]]*\]?(.*?)\\end\{figure\}",
               fig_repl, s, flags=re.S)
    # tables
    def table_repl(m):
        inner=m.group(1)
        # the column specification may itself contain braces, as in {@{}clp{9cm}@{}}
        tm=re.search(r"\\begin\{tabular\}\{(?:[^{}]|\{[^{}]*\})*\}(.*?)\\end\{tabular\}",
                     inner, re.S)
        cm=re.search(r"\\caption\{(.*?)\}\s*(?:\\label|\Z)", inner, re.S)
        md = tabular_to_md(tm.group(1)) if tm else ""
        cap = ("\n\n*"+tabnumber(inner)+inline(cm.group(1)).strip()+"*") if cm else ""
        return "\n\n"+md+cap+"\n\n"
    s = re.sub(r"\\begin\{table\}\[?[^\]]*\]?(.*?)\\end\{table\}", table_repl, s, flags=re.S)
    s = re.sub(r"\\begin\{center\}(.*?)\\end\{center\}", lambda m: table_repl(m), s, flags=re.S)

    # notebox -> admonition
    def note_repl(m):
        inner = convert_prose(m.group(1), chnum).strip()
        title = "Note"
        tm = re.match(r"\*\*(.*?)\*\*\s*", inner)
        if tm:
            title = tm.group(1).rstrip(".")
            inner = inner[tm.end():]
        body = "\n".join("  "+l if l.strip() else "" for l in inner.split("\n"))
        return f"\n\n```{{admonition}} {title}\n:class: tip\n{inner}\n```\n\n"
    s = re.sub(r"\\begin\{notebox\}(.*?)\\end\{notebox\}", note_repl, s, flags=re.S)

    # theorem-like environments -> titled admonitions (Chapter 10 onwards)
    THMENV = {"theorem": "Theorem", "proposition": "Proposition",
              "lemma": "Lemma", "corollary": "Corollary",
              "definition": "Definition", "example": "Example"}
    def thm_repl(env, cls):
        def _f(m):
            opt, body = m.group(1), m.group(2)
            name = re.sub(r"^\[|\]$", "", opt or "").strip()
            lab = re.search(r"\\label\{([^}]*)\}", body)
            numtxt = f" {THM[lab.group(1)]}" if lab and lab.group(1) in THM else ""
            title = THMENV[env] + numtxt + (f" ({name})" if name else "")
            inner = convert_prose(body, chnum).strip()
            return f"\n\n```{{admonition}} {title}\n:class: {cls}\n{inner}\n```\n\n"
        return _f
    for env, cls in [(e, "important") for e in THMENV]:
        s = re.sub(r"\\begin\{" + env + r"\}(\[[^\]]*\])?(.*?)\\end\{" + env + r"\}",
                   thm_repl(env, cls), s, flags=re.S)
    def proof_repl(m):
        inner = convert_prose(m.group(1), chnum).strip()
        return f"\n\n```{{admonition}} Proof\n:class: note\n{inner}\n```\n\n"
    s = re.sub(r"\\begin\{proof\}(.*?)\\end\{proof\}", proof_repl, s, flags=re.S)

    # display math.  Bodies of theorem-like environments, noteboxes and
    # proofs were converted by a recursive convert_prose call above, so
    # their math is already in its final form; protect it, otherwise the
    # \begin{align} inside a finished $$...$$ block (or a bare tagged
    # equation*) would be matched and wrapped a second time.
    done = []
    def _keep(m):
        txt = m.group(0)
        if txt.startswith("\\begin{equation*}") and "\\tag{" not in txt:
            return txt          # a plain equation* from the LaTeX source
        done.append(txt); return f"\x02D{len(done)-1}\x02"
    s = re.sub(r"\$\$.*?\$\$", _keep, s, flags=re.S)
    s = re.sub(r"\\begin\{equation\*\}.*?\\end\{equation\*\}", _keep, s, flags=re.S)
    def math_repl(m):
        env, body = m.group(1), m.group(2)
        labels = re.findall(r"\\label\{([^}]*)\}", body)
        tag = ""
        if labels and labels[0] in EQ: tag = "\\tag{%s}" % EQ[labels[0]]
        body = re.sub(r"\\label\{[^}]*\}", "", body)
        if env.startswith("equation"):
            if tag and "\\\\" in body:
                # Sphinx's MathJax writer wraps any $$...$$ block that contains
                # a line break (\\ -- matrices, cases, split, aligned) in
                # \begin{split}...\end{split}, and MathJax then rejects the
                # \tag: "\tag not allowed in split environment".  A bare
                # starred amsmath environment goes through MyST's amsmath
                # extension instead, which passes it to MathJax untouched and
                # adds no Sphinx equation number of its own.
                return ("\n\n\\begin{equation*}\n"+body.strip()+tag
                        +"\n\\end{equation*}\n\n")
            return "\n\n$$\n"+body.strip()+tag+"\n$$\n\n"
        return "\n\n$$\n\\begin{"+env+"}\n"+body.strip()+"\n\\end{"+env+"}\n$$\n\n"
    s = re.sub(r"\\begin\{(equation\*?|align\*?|eqnarray\*?)\}(.*?)\\end\{\1\}", math_repl, s, flags=re.S)
    s = re.sub(r"(?<!\\)\\\[(.*?)(?<!\\)\\\]",
               lambda m: "\n\n$$\n"+m.group(1).strip()+"\n$$\n\n", s, flags=re.S)
    s = re.sub(r"\x02D(\d+)\x02", lambda m: done[int(m.group(1))], s)

    # lists, innermost first so that nesting is handled correctly
    def list_repl(body, ordered, indent):
        items = re.split(r"\\item\s", body)[1:]
        pad = "    " * indent
        out = []
        for i, t in enumerate(items):
            t = inline(t).strip()
            marker = f"{i+1}." if ordered else "-"
            lines = t.split("\n")
            first = pad + marker + " " + lines[0].strip()
            rest = [pad + "   " + l.strip() if l.strip() else "" for l in lines[1:]]
            out.append("\n".join([first] + rest))
        return "\n\n" + "\n".join(out) + "\n\n"

    def expand_lists(text, indent=0):
        pat = re.compile(r"\\begin\{(itemize|enumerate)\}"
                         r"((?:(?!\\begin\{(?:itemize|enumerate)\}).)*?)"
                         r"\\end\{\1\}", re.S)
        while True:
            m = pat.search(text)
            if not m: break
            text = text[:m.start()] + list_repl(m.group(2), m.group(1)=="enumerate",
                                                indent) + text[m.end():]
        return text
    s = expand_lists(s)
    # description lists: \item[label] text -> - **label** text
    def desc_repl(m):
        items = re.split(r"\\item\[", m.group(1))[1:]
        out = []
        for t in items:
            lab, rest = t.split("]", 1)
            out.append("- **" + inline(lab).strip() + "** " + re.sub(r"\s+", " ", inline(rest)).strip())
        return "\n\n" + "\n".join(out) + "\n\n"
    s = re.sub(r"\\begin\{description\}(.*?)\\end\{description\}", desc_repl, s, flags=re.S)
    s = re.sub(r"\\textsc\{(.*?)\}", lambda m: m.group(1).upper(), s)
    s = re.sub(r"\\sloppy\b", "", s)

    # sectioning.  The title may itself contain braces -- a heading such as
    # \subsection{Against Chapter~\ref{chap:lreg}} is common -- so the argument
    # is matched with one level of nesting allowed rather than non-greedily.
    ARG = r"\{((?:[^{}]|\{[^{}]*\})*)\}"
    s = re.sub(r"\\section\*?"+ARG, lambda m: "\n\n## "+inline(m.group(1))+"\n", s, flags=re.S)
    s = re.sub(r"\\subsection\*?"+ARG, lambda m: "\n\n### "+inline(m.group(1))+"\n", s, flags=re.S)
    s = re.sub(r"\\paragraph"+ARG, lambda m: "\n\n**"+inline(m.group(1)).rstrip(".")+".** ", s, flags=re.S)
    s = re.sub(r"\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}", "", s)
    s = re.sub(r"\\chaptermark\{[^}]*\}", "", s)
    # front-matter bookkeeping that carries no meaning in a notebook
    s = re.sub(r"\\markboth\{[^}]*\}\{[^}]*\}", "", s)
    s = re.sub(r"\\setcounter\{[^}]*\}\{[^}]*\}", "", s)
    s = re.sub(r"\\(clearemptydoublepage|newpage|noindent|centering|toprule|midrule|bottomrule|addlinespace|phantomsection)\b", "", s)
    s = re.sub(r"\\(small|footnotesize|scriptsize|normalsize|large|Large)\b", "", s)
    s = re.sub(r"\\renewcommand\{[^}]*\}\{[^}]*\}", "", s)
    s = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", s)

    # protect math so that the text-mode substitutions in inline() cannot
    # reach inside it
    store = []
    def stash(m):
        store.append(m.group(0)); return f"\x00MATH{len(store)-1}\x00"
    s = re.sub(r"\$\$.*?\$\$", stash, s, flags=re.S)
    # bare starred environments emitted by math_repl for tagged multi-line
    # equations (see the comment there) need the same protection
    s = re.sub(r"\\begin\{equation\*\}.*?\\end\{equation\*\}", stash, s, flags=re.S)
    s = re.sub(r"\$(?:[^$\n]|\n(?!\n))*?\$", stash, s)

    s = inline(s)

    s = re.sub(r"\x00MATH(\d+)\x00",
               lambda m: resolve_refs_in_math(store[int(m.group(1))]), s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def convert_chapter(key, num=None):
    """Convert one source.  `key` selects the file, `num` the chapter number
    used in the heading; a source with num=None gets an unnumbered heading."""
    if num is None and isinstance(key, int): num = key
    ch = key                                  # figure-directory key
    s = read(key)
    title, _ = chapter_title(s, key)
    s = re.sub(r"^\s*\\chaptermark\{[^}]*\}\s*\n","",s)
    s = re.sub(r"^\s*\\label\{[^}]*\}\s*\n","",s)
    kind = "Appendix" if isinstance(num, str) else "Chapter"
    head = f"# {kind} {num}: {title}"
    abstract = BOOK.get(num, ("", ""))[1]
    abstract_md = convert_prose(abstract, ch) if abstract else ""

    BOOTSTRAP = ("# Put every BookPrograms/chapter*/ and BookPrograms/common/ on sys.path\n"
                 "import sys, pathlib\n"
                 "root = pathlib.Path.cwd().resolve()\n"
                 "while not (root / 'BookPrograms').is_dir() and root.parent != root:\n"
                 "    root = root.parent\n"
                 "sys.path[:0] = [str(p) for p in (root / 'BookPrograms').iterdir() if p.is_dir()]\n"
                 "import numpy as np\n")
    cells=[{"cell_type":"markdown","metadata":{},
            "source":[head+"\n"] + (["\n", "*" + abstract_md.replace("\n", " ") + "*\n"] if abstract_md else [])},
           {"cell_type":"markdown","metadata":{},
            "source":["<!-- Macro definitions for MathJax, mirroring book.tex -->\n", PREAMBLE]},
           {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
            "source":[l+"\n" for l in BOOTSTRAP.split("\n") if l]}]

    parts = re.split(r"\\begin\{Python\}\{\}\n(.*?)\\end\{Python\}", s, flags=re.S)
    for i,part in enumerate(parts):
        if i % 2 == 0:
            md = convert_prose(part, ch)
            if md:
                # split into one cell per section/subsection so the notebook
                # stays navigable rather than one enormous cell
                chunks, cur = [], []
                for line in md.split("\n"):
                    if line.startswith("## ") and cur and any(x.strip() for x in cur):
                        chunks.append("\n".join(cur).strip()); cur=[line]
                    else:
                        cur.append(line)
                if cur: chunks.append("\n".join(cur).strip())
                for chunk in chunks:
                    if chunk:
                        cells.append({"cell_type":"markdown","metadata":{},
                                      "source":[l+"\n" for l in chunk.split("\n")]})
        else:
            code = part.rstrip("\n")
            try:
                ast.parse(code); is_output = False
            except SyntaxError:
                is_output = True          # captured program output, not source
            if is_output:
                cells.append({"cell_type":"markdown","metadata":{},
                              "source":["```\n"]+[l+"\n" for l in code.split("\n")]+["```\n"]})
            else:
                cells.append({"cell_type":"code","execution_count":None,"metadata":{},
                              "outputs":[], "source":[l+"\n" for l in code.split("\n")]})
    # one cell per program of the closing "Programs for this chapter" list,
    # so that the executed notebook shows every number and figure of the text
    progs = re.findall(r"\\item\[\\texttt\{([^}]*\.py)\}\]", s)
    progs = [p.replace("\\_", "_") for p in progs]
    if progs and isinstance(key, int):
        cells.append({"cell_type":"markdown","metadata":{},
                      "source":["## Running the programs of this chapter\n", "\n",
                                "Each cell below runs one of the programs listed above as a script and shows the figures it writes to `BookFigures/`.\n"]})
        for p in progs:
            rel = p if "/" in p else f"chapter{key}/{p}"
            code = ("import runpy, time\n"
                    "from IPython.display import Image, display\n"
                    "t0 = time.time()\n"
                    f"runpy.run_path(str(root / 'BookPrograms' / '{rel}'), run_name='__main__')\n"
                    f"for png in sorted((root / 'BookFigures' / 'chapter{key}').glob('*.png')):\n"
                    "    if png.stat().st_mtime >= t0:\n"
                    "        display(Image(str(png)))\n")
            cells.append({"cell_type":"code","execution_count":None,"metadata":{},
                          "outputs":[], "source":[l+"\n" for l in code.split("\n") if l]})
    for c in cells: c["id"] = uuid.uuid4().hex[:12]
    nb={"cells":cells,
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                    "language_info":{"name":"python","version":"3.11"}},
        "nbformat":4,"nbformat_minor":5}
    return nb

if __name__ == "__main__":
    out = sys.argv[1]
    # optional second argument: a comma separated list of sources to rebuild,
    # e.g. "17,introduction,conclusions".  Default is everything.
    wanted = None
    if len(sys.argv) > 2:
        wanted = {int(x) if x.strip().isdigit() else x.strip()
                  for x in sys.argv[2].split(",")}
    # render the TikZ figures of the requested chapters first
    try:
        import render_tikz
        for key, num in SOURCES:
            if (wanted is None or key in wanted) and isinstance(key, int):
                render_tikz.render_chapter(key)
    except Exception as e:                       # pdflatex missing, etc.
        print("TikZ rendering skipped:", e)
    os.makedirs(out, exist_ok=True)
    for key, num in SOURCES:
        if wanted is not None and key not in wanted: continue
        stem = f"chapter{key}" if isinstance(key, int) else key
        if not os.path.exists(f"{SRC}/{stem}.tex"): continue
        nb = convert_chapter(key, num)
        folder = f"{out}/{stem}" if isinstance(key, int) else f"{out}/appendix{num}"
        os.makedirs(folder, exist_ok=True)
        with open(f"{folder}/{stem}.ipynb","w") as f: json.dump(nb,f,indent=1)
        nmd=sum(1 for c in nb["cells"] if c["cell_type"]=="markdown")
        ncode=sum(1 for c in nb["cells"] if c["cell_type"]=="code")
        print(f"{stem}.ipynb: {nmd} markdown, {ncode} code cells")
