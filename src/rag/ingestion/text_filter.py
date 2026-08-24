"""Text filtering: remove site-wide boilerplate injected by PHASE 1 scraping.

PHASE 1 scraped the site template into every page, so many records carry
repeated header (contact bar / language switcher) and footer (location,
newsletter, visitor counters) blocks. These pollute retrieval and waste
LLM context, so they are stripped here.

Only the extracted text is filtered; ``data/documents.jsonl`` is never
modified.
"""

from __future__ import annotations

import re

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------
# Footer markers (EN / AR). Everything from the first marker onward is
# treated as boilerplate tail unless a later real-content marker exists.
# --------------------------------------------------------------------------

_EN_FOOTER_MARKERS = (
    "Our university location",
    "FOLLOW US ON",
    "Subscribe to our newsletter",
    "All Rights Reserved",
    "Get in Touch",
)

_AR_FOOTER_MARKERS = (
    "موقع الجامعة",
    "روابط سريعة",
    "اشترك في نشرتنا البريدية",
    "جميع الحقوق محفوظة",
    "ابقى على تواصل",
)

# Headers / nav noise that can appear at the very start of a page.
_EN_HEADER_MARKERS = (
    "01070004148 - 01070004149",
    "Sat - Thu: 9 AM - 4 PM",
    "FOLLOW US",
)
_AR_HEADER_MARKERS = (
    "اتصل بنا في مواعيد عملنا على",
    "تواصل معنا بريدياً 24/7",
    "01070004148 - 01070004149",
)

# Lines that are pure noise regardless of language.
_NOISE_LINES = (
    "FOLLOW US",
    "English",
    "العربية",
    "Home",
    "الرئيسية",
    "View All",
    "عرض الكل",
    "Learn More",
    "قراءة المزيد",
    "Read More Details",
    "More About Us",
)


class TextFilter:
    """Strip repeated site boilerplate from document text."""

    def clean(self, text: str) -> str:
        """Return filtered text (header + footer boilerplate removed)."""
        if not text:
            return text
        text = self._strip_footer(text)
        text = self._strip_header(text)
        text = self._strip_noise_lines(text)
        return text.strip()

    def _strip_footer(self, text: str) -> str:
        """Cut everything from the first recognised footer marker onward.

        Markers are chosen to be unambiguous footer starts (they do not
        appear in the header/nav). A marker is only honoured when it sits in
        the trailing portion of the document, which prevents accidental
        truncation when a phrase appears mid-content.
        """
        markers = _EN_FOOTER_MARKERS + _AR_FOOTER_MARKERS
        positions = []
        for marker in markers:
            idx = text.find(marker)
            if idx != -1:
                positions.append(idx)
        if not positions:
            return text

        # Only cut at markers located in the last 60% of the document.
        guard = int(len(text) * 0.4)
        late_positions = [p for p in positions if p >= guard]
        if not late_positions:
            return text
        cut = min(late_positions)
        return text[:cut].rstrip()

    def _strip_header(self, text: str) -> str:
        """Remove a leading contact/nav block by dropping consecutive noise lines."""
        lines = text.splitlines()
        start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if _is_header_noise(stripped):
                start = i + 1
            else:
                break
        return "\n".join(lines[start:]).strip()

    def _strip_noise_lines(self, text: str) -> str:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        out = [ln for ln in lines if ln.strip() not in _NOISE_LINES]
        return "\n".join(out)


_TITLE_LIKE_RE = re.compile(r"^[A-Z][^a-z]{4,}$")

_HEADER_NOISE_PATTERNS = (
    # phone bar
    r"^0107000\d{4}",
    r"^info@nmu\.edu\.eg$",
    r"^Sat - Thu: 9 AM - 4 PM$",
    r"^السبت - الخميس: 9 ص - 4 م$",
    r"^FOLLOW US$",
    r"^تابعنا على$",
    r"^English$",
    r"^العربية$",
    r"^Home$",
    r"^الرئيسية$",
    r"^اتصل بنا في مواعيد عملنا على$",
    r"^تواصل معنا بريدياً 24/7$",
)

_HEADER_NOISE_RE = [re.compile(p) for p in _HEADER_NOISE_PATTERNS]


def _is_header_noise(line: str) -> bool:
    return any(rx.match(line) for rx in _HEADER_NOISE_RE)


def _is_title_like(line: str) -> bool:
    """Heuristic: an all-caps or heading-ish standalone line."""
    s = line.strip()
    if not s:
        return False
    if s.isupper() and len(s) > 5:
        return True
    return False