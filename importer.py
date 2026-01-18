import argparse
import json
import os
import re
import sqlite3
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup


STOPWORDS = {
    "introduction",
    "intro",
    "basics",
    "sauces",
    "stock",
    "stocks",
    "index",
    "contents",
    "toc",
    "acknowledgements",
    "acknowledgments",
    "about",
    "foreword",
    "preface",
    "glossary",
    "notes",
    "bibliography",
    "credits",
    "tips",
    "essentials",
    "equipment",
    "pantry",
    "pantry staples",
}

NON_RECIPE_FILE_HINTS = [
    "acknowledgement",
    "acknowledgment",
    "about",
    "about_the_author",
    "author",
    "chapter_",
    "note_on",
    "note_on_ovens",
    "note_on_tins",
    "note-on",
    "note",
    "intro",
    "introduction",
    "contents",
    "content",
    "title",
    "dedication",
    "map",
    "part",
    "part001_2",
    "part002_2",
    "part003_2",
    "part004_2",
    "recipe_pairings",
    "list_of_recipes",
    "recipe_list",
    "sides",
    "split_000",
    "split_002",
    "copyright",
]

BLOCKED_HREF_SUBSTRINGS = [
    "about_author",
    "about_author.xhtml",
    "about_book",
    "about_book.xhtml",
    "about_the_author",
    "note_on_ovens",
    "note_on_tins",
    "recipe_pairings",
    "sides.xhtml",
    "part004_2.xhtml",
    "part003_2.xhtml",
    "part002_2.xhtml",
    "part001_2.xhtml",
]

NON_RECIPE_TITLE_HINTS = [
    "worknight dinners",
    "weekend cooking",
    "family favourites",
    "family favorites",
    "make ahead lunchboxes",
    "date night",
    "sides",
    "recipe pairings",
    "picnic table",
    "light lunch",
    "autumnal dinners",
    "central south america",
    "usa the caribbean",
    "africa the middle east",
    "europe north asia",
    "asia",
    "acknowledgement",
    "acknowledgement s",
    "acknowledgments",
    "acknowledgments s",
    "copyright",
    "fruit plus",
    "note on tins",
    "note on ovens",
    "about the author",
    "about the book",
    "introduction",
]


@dataclass
class ManifestItem:
    href: str
    media_type: str
    properties: str


@dataclass
class RecipeCandidate:
    title: str
    href: str
    spine_index: Optional[int]
    image_paths: List[str]
    ingredients: Optional[str] = None
    steps: Optional[str] = None
    meta: Optional[str] = None
    score: int = 0


class EpubReader:
    def __init__(self, path: str):
        self.path = path
        self._zip = None
        self._is_dir = os.path.isdir(path)
        if not self._is_dir:
            self._zip = zipfile.ZipFile(path, "r")

    def close(self) -> None:
        if self._zip:
            self._zip.close()

    def open_bytes(self, relpath: str) -> bytes:
        if self._is_dir:
            full = os.path.join(self.path, relpath)
            with open(full, "rb") as f:
                return f.read()
        return self._zip.read(relpath)

    def exists(self, relpath: str) -> bool:
        if self._is_dir:
            return os.path.exists(os.path.join(self.path, relpath))
        try:
            self._zip.getinfo(relpath)
            return True
        except KeyError:
            return False


def scan_epub_paths(root: str) -> List[str]:
    if os.path.isfile(root) and root.lower().endswith(".epub"):
        return [root]
    if os.path.isdir(root) and root.lower().endswith(".epub"):
        return [root]

    paths: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Capture .epub directories and avoid descending into them
        epub_dirs = [d for d in dirnames if d.lower().endswith(".epub")]
        for d in epub_dirs:
            paths.append(os.path.join(dirpath, d))
        dirnames[:] = [d for d in dirnames if d not in epub_dirs]

        for fn in filenames:
            if fn.lower().endswith(".epub"):
                paths.append(os.path.join(dirpath, fn))
    return sorted(set(paths))


def _xml_text(elem: Optional[ET.Element]) -> str:
    if elem is None or elem.text is None:
        return ""
    return elem.text.strip()


def _safe_title(text: str) -> str:
    return " ".join(text.split()).strip()


def _normalize_line(text: str) -> str:
    return _safe_title(text.replace("\xa0", " "))


def _dedupe_lines(lines: List[str]) -> List[str]:
    seen = set()
    out = []
    for line in lines:
        key = _normalize_line(line).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def parse_container(reader: EpubReader) -> str:
    xml_bytes = reader.open_bytes("META-INF/container.xml")
    root = ET.fromstring(xml_bytes)
    # container.xml namespace can vary; match by localname
    for elem in root.iter():
        if elem.tag.endswith("rootfile") and "full-path" in elem.attrib:
            return elem.attrib["full-path"]
    raise ValueError("OPF not found in container.xml")


def parse_opf(reader: EpubReader, opf_path: str) -> Tuple[str, str, Dict[str, ManifestItem], List[str]]:
    xml_bytes = reader.open_bytes(opf_path)
    root = ET.fromstring(xml_bytes)

    ns = {"dc": "http://purl.org/dc/elements/1.1/"}
    title = ""
    author = ""
    for elem in root.iter():
        if elem.tag.endswith("title") and not title:
            title = _xml_text(elem)
        if elem.tag.endswith("creator") and not author:
            author = _xml_text(elem)

    manifest: Dict[str, ManifestItem] = {}
    spine: List[str] = []

    for elem in root.iter():
        if elem.tag.endswith("item"):
            item_id = elem.attrib.get("id", "")
            href = elem.attrib.get("href", "")
            media_type = elem.attrib.get("media-type", "")
            properties = elem.attrib.get("properties", "")
            if item_id and href:
                manifest[item_id] = ManifestItem(href=href, media_type=media_type, properties=properties)

    for elem in root.iter():
        if elem.tag.endswith("itemref"):
            idref = elem.attrib.get("idref")
            if idref:
                spine.append(idref)

    return title, author, manifest, spine


def _normalize_href(href: str) -> str:
    cleaned = href.strip()
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if cleaned.startswith("/"):
        cleaned = cleaned[1:]
    return cleaned


def _resolve_href(base_dir: str, href: str) -> str:
    norm = _normalize_href(href)
    if base_dir:
        return os.path.normpath(os.path.join(base_dir, norm))
    return norm


def find_toc(manifest: Dict[str, ManifestItem]) -> Optional[ManifestItem]:
    for item in manifest.values():
        if "nav" in (item.properties or ""):
            return item
    for item in manifest.values():
        if item.media_type == "application/x-dtbncx+xml":
            return item
    return None


def parse_nav_toc(reader: EpubReader, toc_path: str) -> List[Tuple[str, str]]:
    html = reader.open_bytes(toc_path).decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    nav = None
    for candidate in soup.find_all("nav"):
        if candidate.get("epub:type") == "toc" or candidate.get("type") == "toc" or candidate.get("role") == "doc-toc":
            nav = candidate
            break
    if nav is None:
        nav = soup.find("nav")
    entries = []
    if nav:
        for a in nav.find_all("a"):
            href = a.get("href") or ""
            text = _safe_title(a.get_text(" ", strip=True))
            if href and text:
                entries.append((text, href))
    return entries


def parse_ncx(reader: EpubReader, ncx_path: str) -> List[Tuple[str, str]]:
    xml_bytes = reader.open_bytes(ncx_path)
    root = ET.fromstring(xml_bytes)
    entries = []
    for nav_point in root.iter():
        if nav_point.tag.endswith("navPoint"):
            label = None
            content = None
            for child in nav_point:
                if child.tag.endswith("navLabel"):
                    for lbl in child:
                        if lbl.tag.endswith("text"):
                            label = _safe_title(_xml_text(lbl))
                if child.tag.endswith("content"):
                    content = child.attrib.get("src")
            if label and content:
                entries.append((label, content))
    return entries


def _normalize_toc_text(text: str) -> str:
    cleaned = text.replace("»", "").replace(">", "").strip()
    return _safe_title(cleaned)


def _anchor_from_href(href: str) -> Optional[str]:
    if "#" not in href:
        return None
    return href.split("#", 1)[1]


def _file_from_href(href: str) -> str:
    return href.split("#", 1)[0]


def _is_recipe_toc_entry(book_title: str, title: str, href: str) -> bool:
    lowered = title.lower()
    simplified = re.sub(r"[^a-z]+", " ", lowered).strip()
    if any(key in simplified for key in NON_RECIPE_TITLE_HINTS):
        return False
    if any(
        key in simplified
        for key in [
            "about the author",
            "about author",
            "about the book",
            "about book",
            "note on tins",
            "note on ovens",
            "note on",
            "introduction",
            "contents",
            "list of recipes",
            "recipe list",
            "dedication",
            "title page",
            "foreword",
            "preface",
            "acknowledgements",
            "acknowledgments",
            "map",
            "maps",
        ]
    ):
        return False
    if lowered in STOPWORDS:
        return False
    if "recipe list" in lowered or "list of recipes" in lowered:
        return False
    if "recipe list" in simplified or "list of recipes" in simplified:
        return False

    anchor = _anchor_from_href(href) or ""
    file_only = os.path.basename(_file_from_href(href))
    file_lower = file_only.lower()
    if any(key in file_lower for key in NON_RECIPE_FILE_HINTS):
        return False
    book_lower = book_title.lower()

    if "quick roasting tin" in book_lower:
        return anchor.startswith("qui") and not anchor.endswith(("a", "b", "c"))
    if "sweet roasting tin" in book_lower:
        return anchor.startswith("swe")
    if "green roasting tin" in book_lower:
        if not anchor:
            return False
        return bool(re.match(r"ch\\d+_rec\\d+", anchor))
    if "around the world" in book_lower:
        if file_lower.startswith("part") and "_2" in file_lower:
            return False
        return bool(re.match(r"chapter\\d+\\.xhtml$", file_only))
    if "simple one dish dinners" in book_lower:
        simplified_digits = re.sub(r"[^0-9 ]+", " ", title).strip()
        if re.match(r"^\\d+\\s", simplified_digits):
            return False
        return file_only.startswith("part") and file_only.endswith(".html")

    return True


def _resolve_relative(base_path: str, rel: str) -> str:
    if rel.startswith("http://") or rel.startswith("https://"):
        return rel
    base_dir = os.path.dirname(base_path)
    return os.path.normpath(os.path.join(base_dir, rel))


def _is_heading(tag) -> bool:
    return tag and tag.name in {"h1", "h2", "h3"}


def _heading_level(tag) -> int:
    if not tag or not getattr(tag, "name", None):
        return 7
    if tag.name == "h1":
        return 1
    if tag.name == "h2":
        return 2
    if tag.name == "h3":
        return 3
    if tag.name == "h4":
        return 4
    if tag.name == "h5":
        return 5
    if tag.name == "h6":
        return 6
    return 7


def _collect_section_elements(start_tag, max_elements: int = 400) -> List:
    elements = []
    base_level = _heading_level(start_tag) if _is_heading(start_tag) else 2
    for el in start_tag.find_all_next():
        if _is_heading(el) and _heading_level(el) <= base_level:
            break
        elements.append(el)
        if len(elements) >= max_elements:
            break
    return elements


def _extract_text_list(elements: List) -> List[str]:
    lines: List[str] = []
    for el in elements:
        if el.name in {"ul", "ol"}:
            for li in el.find_all("li"):
                txt = _normalize_line(li.get_text(" ", strip=True))
                if txt:
                    lines.append(txt)
        elif el.name == "table":
            for row in el.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
                line = _normalize_line(" ".join([c for c in cells if c]))
                if line:
                    lines.append(line)
        elif el.name == "dl":
            parts = []
            for child in el.find_all(["dt", "dd"]):
                txt = _safe_title(child.get_text(" ", strip=True))
                if txt:
                    parts.append(txt)
            if parts:
                lines.append(_normalize_line(" - ".join(parts)))
        elif el.name in {"p", "div", "span"}:
            txt = _normalize_line(el.get_text(" ", strip=True))
            if txt:
                lines.append(txt)
    return lines


def _extract_meta_from_lines(lines: List[str]) -> Tuple[List[str], List[str]]:
    meta = []
    remaining = []
    for line in lines:
        lower = _normalize_line(line).lower()
        if any(lower.startswith(key) for key in ["serves", "prep", "cook", "makes", "set"]):
            meta.append(line)
        else:
            remaining.append(line)
    return meta, remaining


def _section_after_heading(elements: List, keywords: List[str]) -> Optional[str]:
    capture = False
    captured: List = []
    for el in elements:
        if _is_heading(el) or el.name in {"strong", "b"}:
            label = _safe_title(el.get_text(" ", strip=True)).lower()
            if any(key in label for key in keywords):
                capture = True
                continue
            if capture:
                break
        if capture:
            captured.append(el)
    if captured:
        lines = _extract_text_list(captured)
        if lines:
            return "\n".join(lines)
    return None


def _section_after_element(elements: List, start_predicate, max_count: int = 200) -> Optional[str]:
    capture = False
    captured: List = []
    for el in elements:
        if start_predicate(el):
            capture = True
            continue
        if capture:
            if _is_heading(el):
                break
            captured.append(el)
            if len(captured) >= max_count:
                break
    if captured:
        lines = _extract_text_list(captured)
        if lines:
            return "\n".join(lines)
    return None


def _find_meta(section_elements: List) -> Optional[str]:
    meta_lines = []
    # Tables with serves/prep/cook
    for table in [el for el in section_elements if getattr(el, "name", None) == "table"]:
        classes = " ".join(table.get("class", []))
        if "serves" in classes or "serves" in table.get("class", []):
            for row in table.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
                if len(cells) >= 2:
                    meta_lines.append(_normalize_line(f"{cells[0]} {cells[1]}"))
    for el in section_elements:
        if el.name in {"ul", "ol"}:
            lines = _extract_text_list([el])
            meta, _ = _extract_meta_from_lines(lines)
            meta_lines.extend(meta)
        if el.name in {"aside", "section"}:
            for p in el.find_all("p"):
                txt = _safe_title(p.get_text(" ", strip=True))
                if not txt:
                    continue
                lower = txt.lower()
                if any(key in lower for key in ["serves", "prep", "cook", "makes", "set"]):
                    meta_lines.append(txt)
            for li in el.find_all("li"):
                li_class = " ".join(li.get("class", []))
                if "no_indent_grey" in li_class:
                    meta_lines.append(_normalize_line(li.get_text(" ", strip=True)))
                if li.find("span", class_="color_ing"):
                    meta_lines.append(_normalize_line(li.get_text(" ", strip=True)))
        if el.name == "p":
            txt = _safe_title(el.get_text(" ", strip=True))
            if not txt:
                continue
            lower = txt.lower()
            if any(key in lower for key in ["serves", "prep", "cook", "makes", "set"]):
                meta_lines.append(txt)
    if meta_lines:
        # de-dupe but preserve order
        out = _dedupe_lines(meta_lines)
        # Keep only short meta lines
        out = [line for line in out if len(line.split()) <= 6]
        return "\n".join(out)
    return None


def _find_ingredients(section_elements: List) -> Optional[str]:
    candidates = []
    for el in section_elements:
        attrs = " ".join([str(el.get("class", "")), str(el.get("id", ""))]).lower()
        if "ingredient" in attrs:
            candidates.extend(_extract_text_list([el]))
    if candidates:
        return "\n".join(_dedupe_lines(candidates))

    extracted = _section_after_heading(section_elements, ["ingredient", "ingredients"])
    if extracted:
        return "\n".join(_dedupe_lines(extracted.splitlines()))

    # Some books wrap ingredients in an aside/section with lists (e.g. sidebar_wrapper)
    for el in section_elements:
        if el.name in {"aside", "section"}:
            attrs = " ".join([str(el.get("class", "")), str(el.get("id", ""))]).lower()
            if "sidebar" in attrs or "ingredient" in attrs or el.find("ul"):
                lists = el.find_all("ul")
                if lists:
                    lines = _extract_text_list(lists)
                    meta, lines = _extract_meta_from_lines(lines)
                    lines = _filter_ingredient_lines(lines)
                    lines = _dedupe_lines(lines)
                    if lines:
                        return "\n".join(lines)
                # Fallback: ingredient lines are paragraphs inside the sidebar
                paras = []
                for p in el.find_all("p"):
                    txt = _safe_title(p.get_text(" ", strip=True))
                    if not txt:
                        continue
                    lower = txt.lower()
                    if any(key in lower for key in ["serves", "prep", "cook", "makes"]):
                        continue
                    # Heuristic: ingredient lines often contain numbers or units
                    if re.search(r"\\d", txt) or any(unit in lower for unit in ["g ", "kg", "ml", "l ", "tbsp", "tsp", "cup"]):
                        paras.append(txt)
                if paras:
                    return "\n".join(_dedupe_lines(paras))

    # Fallback: first substantial list in the section
    for el in section_elements:
        if el.name == "ul":
            lines = _extract_text_list([el])
            meta, lines = _extract_meta_from_lines(lines)
            lines = _filter_ingredient_lines(lines)
            lines = _dedupe_lines(lines)
            if len(lines) >= 3:
                return "\n".join(lines)
    return None


def _filter_ingredient_lines(lines: List[str]) -> List[str]:
    filtered = []
    for line in lines:
        lower = _normalize_line(line).lower()
        if any(prefix in lower for prefix in ["serves", "prep", "cook", "makes", "set"]):
            continue
        filtered.append(line)
    return filtered


def _find_steps(section_elements: List) -> Optional[str]:
    # Prefer explicit main content wrapper blocks
    for el in section_elements:
        if el.name == "div" and "maincontent_wrapper" in " ".join(el.get("class", [])):
            paras = []
            for p in el.find_all("p"):
                txt = _normalize_line(p.get_text(" ", strip=True))
                if txt:
                    paras.append(txt)
            if paras:
                return "\n".join(_dedupe_lines(paras))

    candidates = []
    for el in section_elements:
        attrs = " ".join([str(el.get("class", "")), str(el.get("id", ""))]).lower()
        if any(key in attrs for key in ["method", "direction", "step", "preparation", "instructions"]):
            candidates.extend(_extract_text_list([el]))
    if candidates:
        return "\n".join(_dedupe_lines(candidates))

    # Prefer ordered lists in the section
    for el in section_elements:
        if el.name == "ol":
            extracted = _extract_text_list([el])
            if extracted:
                return "\n".join(_dedupe_lines(extracted))

    extracted = _section_after_heading(
        section_elements,
        ["method", "directions", "direction", "steps", "preparation", "instructions"],
    )
    if extracted:
        return "\n".join(_dedupe_lines(extracted.splitlines()))

    # If we have ingredient lists, methods often follow them without a heading
    extracted = _section_after_element(
        section_elements,
        lambda el: "ingredient" in " ".join([str(el.get("class", "")), str(el.get("id", ""))]).lower(),
    )
    if extracted:
        return "\n".join(_dedupe_lines(extracted.splitlines()))

    # Fallback: collect consecutive paragraphs after ingredients block
    def _is_ingredient_block(el):
        if el.name in {"ul", "ol"} and el.find("li"):
            return True
        attrs = " ".join([str(el.get("class", "")), str(el.get("id", ""))]).lower()
        return "ingredient" in attrs

    captured = []
    capturing = False
    for el in section_elements:
        if _is_ingredient_block(el):
            capturing = True
            continue
        if capturing:
            if _is_heading(el):
                break
            if el.name in {"p", "div"}:
                txt = _safe_title(el.get_text(" ", strip=True))
                if txt:
                    captured.append(txt)
            if len(captured) >= 12:
                break
    if len(captured) >= 2:
        return "\n".join(_dedupe_lines(captured))
    return None


def _ensure_numbered_steps(steps: Optional[str]) -> Optional[str]:
    if not steps:
        return steps
    raw_lines = [line for line in steps.splitlines() if line.strip()]
    if not raw_lines:
        return steps
    norm_lines = []
    for line in raw_lines:
        cleaned = _normalize_line(line)
        m = re.match(r"^(\\d+)[\\.)]\\s+(\\d+)[\\.)]\\s+(.*)$", cleaned)
        if m and m.group(1) == m.group(2):
            cleaned = f"{m.group(1)}. {m.group(3)}"
        norm_lines.append(cleaned)
    # If already numbered, keep as-is after de-duping double numbers above
    if any(re.match(r"^\\d+[\\.)]", line) for line in norm_lines):
        return "\n".join(norm_lines)
    numbered = [f"{idx + 1}. {line}" for idx, line in enumerate(norm_lines)]
    return "\n".join(numbered)


def _find_images(section_elements: List, content_path: str) -> List[str]:
    images = []
    for el in section_elements:
        if el.name == "img" and el.get("src"):
            images.append(_resolve_relative(content_path, el.get("src")))
        img = el.find("img")
        if img and img.get("src"):
            images.append(_resolve_relative(content_path, img.get("src")))
    return list(dict.fromkeys(images))


def _lines_from_lis(lis: List) -> List[str]:
    lines = []
    for li in lis:
        txt = _normalize_line(li.get_text(" ", strip=True))
        if txt:
            lines.append(txt)
    return lines


def _format_ingredient_lines(lines: List[str]) -> str:
    formatted = []
    for line in lines:
        stripped = line.strip()
        if stripped.isupper() and len(stripped.split()) <= 4:
            formatted.append(f"[{stripped}]")
        else:
            formatted.append(line)
    return "\n".join(formatted)


def _extract_book_specific(
    book_title: str,
    section_elements: List,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    book_lower = (book_title or "").lower()

    # Quick Roasting Tin
    if "quick roasting tin" in book_lower:
        meta_lines = []
        ingredients_lines = []
        for el in section_elements:
            if el.name == "section" and "sidebar_wrapper" in " ".join(el.get("class", [])):
                for h in el.find_all("h5"):
                    txt = _normalize_line(h.get_text(" ", strip=True))
                    if txt:
                        lower = txt.lower()
                        if any(key in lower for key in ["serves", "prep", "cook", "makes", "set"]):
                            meta_lines.append(txt)
                for ul in el.find_all("ul"):
                    if "ingredient_items" in " ".join(ul.get("class", [])):
                        ingredients_lines.extend(_lines_from_lis(ul.find_all("li")))
                break
        meta = "\n".join(_dedupe_lines(meta_lines)) if meta_lines else None
        ingredients = _format_ingredient_lines(_dedupe_lines(ingredients_lines)) if ingredients_lines else None
        steps = None
        for el in section_elements:
            if el.name == "div" and "maincontent_wrapper" in " ".join(el.get("class", [])):
                paras = [_normalize_line(p.get_text(" ", strip=True)) for p in el.find_all("p")]
                paras = [p for p in paras if p]
                if paras:
                    steps = "\n".join(_dedupe_lines(paras))
                    break
        return meta, ingredients, steps

    # Green Roasting Tin
    if "green roasting tin" in book_lower:
        meta_lines = []
        serves_table = None
        for el in section_elements:
            if getattr(el, "name", None) == "table" and "serves" in " ".join(el.get("class", [])):
                serves_table = el
                break
        if serves_table:
            for row in serves_table.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
                if len(cells) >= 2:
                    meta_lines.append(_normalize_line(f"{cells[0]} {cells[1]}"))
        ingredients = None
        if serves_table:
            next_ul = serves_table.find_next("ul")
            if next_ul:
                ingredients = _format_ingredient_lines(_dedupe_lines(_lines_from_lis(next_ul.find_all("li"))))
        steps = None
        for el in section_elements:
            if el.name == "div" and "maincontent_wrapper" in " ".join(el.get("class", [])):
                paras = [_normalize_line(p.get_text(" ", strip=True)) for p in el.find_all("p")]
                paras = [p for p in paras if p]
                if paras:
                    steps = "\n".join(_dedupe_lines(paras))
                    break
        meta = "\n".join(_dedupe_lines(meta_lines)) if meta_lines else None
        return meta, ingredients, steps

    # Roasting Tin Around the World
    if "around the world" in book_lower:
        meta_lines = []
        ingredients_lines = []
        for el in section_elements:
            if el.name == "ul" and "ingredient_items" in " ".join(el.get("class", [])):
                for li in el.find_all("li"):
                    li_class = " ".join(li.get("class", []))
                    text = _normalize_line(li.get_text(" ", strip=True))
                    if not text:
                        continue
                    if "no_indent_grey" in li_class:
                        meta_lines.append(text)
                    else:
                        ingredients_lines.append(text)
                break
        steps = None
        for el in section_elements:
            if el.name == "div" and "maincontent_wrapper" in " ".join(el.get("class", [])):
                paras = [_normalize_line(p.get_text(" ", strip=True)) for p in el.find_all("p")]
                paras = [p for p in paras if p]
                if paras:
                    steps = "\n".join(_dedupe_lines(paras))
                    break
        meta = "\n".join(_dedupe_lines(meta_lines)) if meta_lines else None
        ingredients = _format_ingredient_lines(_dedupe_lines(ingredients_lines)) if ingredients_lines else None
        steps = _ensure_numbered_steps(steps)
        return meta, ingredients, steps

    # Simple One Dish Dinners
    if "simple one dish dinners" in book_lower:
        meta_lines = []
        ingredients_lines = []
        aside = None
        for el in section_elements:
            if el.name == "aside" and "sidebar_wrapper" in " ".join(el.get("class", [])):
                aside = el
                break
        if aside:
            for p in aside.find_all("p"):
                txt = _normalize_line(p.get_text(" ", strip=True))
                if not txt:
                    continue
                if "serves" in " ".join(p.get("class", [])):
                    meta_lines.append(txt)
                else:
                    ingredients_lines.append(txt)
            for ul in aside.find_all("ul"):
                ingredients_lines.extend(_lines_from_lis(ul.find_all("li")))
        steps = None
        for el in section_elements:
            if el.name == "div" and "maincontent_wrapper" in " ".join(el.get("class", [])):
                paras = [_normalize_line(p.get_text(" ", strip=True)) for p in el.find_all("p")]
                paras = [p for p in paras if p]
                if paras:
                    steps = "\n".join(_dedupe_lines(paras))
                    break
        meta = "\n".join(_dedupe_lines(meta_lines)) if meta_lines else None
        ingredients = _format_ingredient_lines(_dedupe_lines(ingredients_lines)) if ingredients_lines else None
        return meta, ingredients, steps

    # Sweet Roasting Tin
    if "sweet roasting tin" in book_lower:
        meta_lines = []
        ingredients_lines = []
        for el in section_elements:
            if el.name == "ul" and "ingredient_items" in " ".join(el.get("class", [])):
                for li in el.find_all("li"):
                    text = _normalize_line(li.get_text(" ", strip=True))
                    if not text:
                        continue
                    if li.find("span", class_="color_ing"):
                        meta_lines.append(text)
                    else:
                        ingredients_lines.append(text)
                break
        steps = None
        for el in section_elements:
            if el.name == "div" and "maincontent_wrapper" in " ".join(el.get("class", [])):
                paras = [_normalize_line(p.get_text(" ", strip=True)) for p in el.find_all("p")]
                paras = [p for p in paras if p]
                if paras:
                    steps = "\n".join(_dedupe_lines(paras))
                    break
        meta = "\n".join(_dedupe_lines(meta_lines)) if meta_lines else None
        ingredients = _format_ingredient_lines(_dedupe_lines(ingredients_lines)) if ingredients_lines else None
        steps = _ensure_numbered_steps(steps)
        return meta, ingredients, steps

    # Fallback: use generic heuristics
    return None, None, None


def extract_from_anchor(
    reader: EpubReader,
    content_path: str,
    spine_index: int,
    anchor: Optional[str],
    fallback_title: Optional[str] = None,
    book_title: Optional[str] = None,
) -> Optional[RecipeCandidate]:
    html = reader.open_bytes(content_path).decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    target = None
    if anchor:
        target = soup.find(id=anchor) or soup.find(attrs={"name": anchor})
    if target is None and fallback_title:
        normalized = _safe_title(fallback_title).lower()
        if normalized:
            for h in soup.find_all(["h1", "h2", "h3"]):
                if _safe_title(h.get_text(" ", strip=True)).lower() == normalized:
                    target = h
                    break
    if target is None:
        target = soup.find(["h1", "h2", "h3"])
    if target is None:
        return None

    heading = target if _is_heading(target) else target.find_previous(["h1", "h2", "h3"])
    if heading is None:
        heading = target

    if _is_heading(heading):
        title = _safe_title(heading.get_text(" ", strip=True))
    else:
        title = _safe_title(fallback_title or "")
    if not title:
        return None
    lowered = title.lower()
    if lowered in STOPWORDS or len(title) < 3:
        return None

    href = content_path
    if anchor:
        href = f"{content_path}#{anchor}"

    section_elements = _collect_section_elements(heading)
    image_paths = _find_images(section_elements, content_path)

    meta, ingredients, steps = _extract_book_specific(book_title or "", section_elements)
    if meta is None:
        meta = _find_meta(section_elements)
    if ingredients is None:
        ingredients = _find_ingredients(section_elements)
    if steps is None:
        steps = _ensure_numbered_steps(_find_steps(section_elements))
    score = 0
    if ingredients:
        score += 2
    if steps:
        score += 2
    if image_paths:
        score += 1

    return RecipeCandidate(
        title=title,
        href=href,
        spine_index=spine_index,
        image_paths=image_paths,
        ingredients=ingredients,
        steps=steps,
        meta=meta,
        score=score,
    )


def extract_headings(
    reader: EpubReader,
    content_path: str,
    spine_index: int,
    book_title: Optional[str] = None,
) -> List[RecipeCandidate]:
    html = reader.open_bytes(content_path).decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    candidates: List[RecipeCandidate] = []

    for tag in soup.find_all(["h1", "h2", "h3"]):
        title = _safe_title(tag.get_text(" ", strip=True))
        if not title:
            continue
        lowered = title.lower()
        if "recipe list" in lowered or "list of recipes" in lowered:
            continue
        if lowered in STOPWORDS:
            continue
        if len(title) < 3:
            continue

        anchor = tag.get("id")
        href = content_path
        if anchor:
            href = f"{content_path}#{anchor}"

        section_elements = _collect_section_elements(tag)
        image_paths = _find_images(section_elements, content_path)
        meta, ingredients, steps = _extract_book_specific(book_title or "", section_elements)
        if meta is None:
            meta = _find_meta(section_elements)
        if ingredients is None:
            ingredients = _find_ingredients(section_elements)
        if steps is None:
            steps = _ensure_numbered_steps(_find_steps(section_elements))
        score = 0
        if ingredients:
            score += 2
        if steps:
            score += 2
        if image_paths:
            score += 1

        candidates.append(
            RecipeCandidate(
                title=title,
                href=href,
                spine_index=spine_index,
                image_paths=image_paths,
                ingredients=ingredients,
                steps=steps,
                meta=meta,
                score=score,
            )
        )

    return candidates


def init_db(db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY,
                title TEXT,
                author TEXT,
                source_path TEXT UNIQUE
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recipes (
                id INTEGER PRIMARY KEY,
                book_id INTEGER,
                title TEXT,
                href TEXT,
                spine_index INTEGER,
                image_paths TEXT,
                ingredients TEXT,
                steps TEXT,
                score INTEGER,
                meta TEXT,
                UNIQUE(book_id, title, href),
                FOREIGN KEY(book_id) REFERENCES books(id)
            )
            """
        )
        cur.execute("PRAGMA table_info(recipes)")
        existing = {row[1] for row in cur.fetchall()}
        if "ingredients" not in existing:
            cur.execute("ALTER TABLE recipes ADD COLUMN ingredients TEXT")
        if "steps" not in existing:
            cur.execute("ALTER TABLE recipes ADD COLUMN steps TEXT")
        if "score" not in existing:
            cur.execute("ALTER TABLE recipes ADD COLUMN score INTEGER")
        if "meta" not in existing:
            cur.execute("ALTER TABLE recipes ADD COLUMN meta TEXT")
        conn.commit()
    finally:
        conn.close()


def upsert_book(conn: sqlite3.Connection, title: str, author: str, source_path: str) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO books (title, author, source_path)
        VALUES (?, ?, ?)
        ON CONFLICT(source_path) DO UPDATE SET
            title=excluded.title,
            author=excluded.author
        """
        ,
        (title, author, source_path),
    )
    conn.commit()
    cur.execute("SELECT id FROM books WHERE source_path = ?", (source_path,))
    row = cur.fetchone()
    return int(row[0])


def insert_recipes(conn: sqlite3.Connection, book_id: int, recipes: Iterable[RecipeCandidate]) -> int:
    cur = conn.cursor()
    cur.execute("DELETE FROM recipes WHERE book_id = ?", (book_id,))
    inserted = 0
    for r in recipes:
        cur.execute(
            """
            INSERT OR IGNORE INTO recipes
                (book_id, title, href, spine_index, image_paths, ingredients, steps, score, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                book_id,
                r.title,
                r.href,
                r.spine_index,
                json.dumps(r.image_paths) if r.image_paths else None,
                r.ingredients,
                r.steps,
                r.score,
                r.meta,
            ),
        )
        if cur.rowcount:
            inserted += 1
    conn.commit()
    return inserted


def build_recipes_for_book(reader: EpubReader) -> Tuple[str, str, List[RecipeCandidate]]:
    opf_path = parse_container(reader)
    opf_dir = os.path.dirname(opf_path)
    title, author, manifest, spine = parse_opf(reader, opf_path)

    # Normalize manifest hrefs relative to OPF dir
    manifest_paths: Dict[str, str] = {}
    for item_id, item in manifest.items():
        href = _normalize_href(item.href)
        full = os.path.join(opf_dir, href) if opf_dir else href
        manifest_paths[item_id] = full

    toc_item = find_toc(manifest)
    toc_entries: List[RecipeCandidate] = []
    if toc_item:
        toc_path = _resolve_href(opf_dir, toc_item.href)
        toc_base_dir = os.path.dirname(toc_path)
        if toc_item.media_type == "application/x-dtbncx+xml":
            entries = parse_ncx(reader, toc_path)
        else:
            entries = parse_nav_toc(reader, toc_path)
        for text, href in entries:
            text = _normalize_toc_text(text)
            if not text:
                continue
            full = _resolve_href(toc_base_dir, href)
            if not _is_recipe_toc_entry(title, text, full):
                continue
            spine_index = None
            file_only = full.split("#", 1)[0]
            for idx, item_id in enumerate(spine):
                if manifest_paths.get(item_id) == file_only:
                    spine_index = idx
                    break
            anchor = None
            if "#" in full:
                file_only, anchor = full.split("#", 1)
            else:
                file_only = full
            candidate = None
            if reader.exists(file_only):
                try:
                    candidate = extract_from_anchor(reader, file_only, spine_index, anchor, fallback_title=text, book_title=title)
                except Exception:
                    candidate = None
            if candidate is None:
                candidate = RecipeCandidate(title=text, href=full, spine_index=spine_index, image_paths=[])
            toc_entries.append(candidate)

    if toc_entries:
        # De-dupe by title+href
        seen = set()
        combined: List[RecipeCandidate] = []
        for entry in toc_entries:
            key = (entry.title.lower(), entry.href)
            if key in seen:
                continue
            seen.add(key)
            combined.append(entry)
        # Hard filter for known non-recipe paths
        filtered = []
        for entry in combined:
            href_lower = (entry.href or "").lower()
            if any(block in href_lower for block in BLOCKED_HREF_SUBSTRINGS):
                continue
            filtered.append(entry)
        return title or "Unknown Title", author, filtered

    heading_entries: List[RecipeCandidate] = []
    for idx, item_id in enumerate(spine):
        path = manifest_paths.get(item_id)
        if not path:
            continue
        if not reader.exists(path):
            continue
        try:
            heading_entries.extend(extract_headings(reader, path, idx, book_title=title))
        except Exception:
            continue

    filtered = []
    for entry in heading_entries:
        href_lower = (entry.href or "").lower()
        if any(block in href_lower for block in BLOCKED_HREF_SUBSTRINGS):
            continue
        filtered.append(entry)
    return title or "Unknown Title", author, filtered


def import_library(root: str, db_path: str) -> Tuple[int, int]:
    init_db(db_path)
    epub_paths = scan_epub_paths(root)
    conn = sqlite3.connect(db_path)
    try:
        book_count = 0
        recipe_count = 0
        for path in epub_paths:
            try:
                reader = EpubReader(path)
                try:
                    title, author, recipes = build_recipes_for_book(reader)
                finally:
                    reader.close()
                book_id = upsert_book(conn, title, author, os.path.abspath(path))
                inserted = insert_recipes(conn, book_id, recipes)
                book_count += 1
                recipe_count += inserted
            except Exception:
                continue
        return book_count, recipe_count
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import recipes from EPUB cookbooks")
    parser.add_argument("path", help="Folder (or single .epub file) to scan")
    parser.add_argument("--db", default="data/library.sqlite", help="SQLite database path")
    args = parser.parse_args()

    books, recipes = import_library(args.path, args.db)
    print(f"Imported {books} books, {recipes} recipes into {args.db}")


if __name__ == "__main__":
    main()
