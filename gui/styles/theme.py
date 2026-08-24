"""Central color palette and Qt stylesheet for the NMU AI Assistant GUI."""

from __future__ import annotations

COLORS: dict[str, str] = {
    "background": "#F6F7FB",
    "surface": "#FFFFFF",
    "surface_secondary": "#EEF1F5",
    "text_primary": "#0B1220",
    "text_secondary": "#475467",
    "text_tertiary": "#667085",
    "border": "#E5E7EB",
    "border_strong": "#D0D5DD",
    "input_border": "#D0D5DD",
    "input_bg": "#FFFFFF",
    "input_text": "#0B1220",
    "placeholder": "#98A2B3",
    "accent": "#D97706",
    "accent_hover": "#B45309",
    "accent_pressed": "#92400E",
    "accent_soft": "#FFF7ED",
    "button_bg": "#D97706",
    "button_hover": "#B45309",
    "button_pressed": "#92400E",
    "button_text": "#FFFFFF",
    "disabled_button": "#FED7AA",
    "disabled_button_text": "#FFFFFF",
    "user_bg": "#FFF4E5",
    "user_border": "#FCC58C",
    "assistant_bg": "#FFFFFF",
    "assistant_border": "#DCDFE4",
    "assistant_accent": "#D97706",
    "link": "#1D4ED8",
    "error": "#991B1B",
    "error_bg": "#FEF2F2",
    "error_border": "#FECACA",
    "success": "#15803D",
    "scrollbar": "#CBD2DA",
    "scrollbar_hover": "#94A3B8",
}

# Font stack: prefer Noto Sans Arabic for its excellent connected Arabic
# glyphs + Latin coverage; fall back through system Arabic UI faces. Qt's
# QFont will fall through the comma-separated list automatically.
FONT_FAMILY = "Noto Sans Arabic, Segoe UI Arabic, Segoe UI, Tahoma, Arial"
ANSWER_POINT_SIZE = 11  # ~15 px, readable for long answers

def _build_answer_stylesheet(colors: dict[str, str]) -> str:
    return f"""
    body {{
        color: {colors['text_primary']};
        font-family: {FONT_FAMILY};
        font-size: {ANSWER_POINT_SIZE}pt;
        line-height: 1.65;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }}
    p {{
        margin: 0 0 12px 0;
        line-height: 1.65;
    }}
    p:empty {{ display: none; }}
    p:last-child {{
        margin-bottom: 0;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {colors['text_primary']};
        line-height: 1.32;
        margin: 16px 0 10px 0;
        font-weight: 700;
    }}
    /* Expanded individually: Qt's QTextDocument CSS subset is stricter than
       QSS and handles grouped selectors less reliably. */
    h1 {{ color: {colors['text_primary']}; line-height: 1.32; margin: 16px 0 10px 0; font-weight: 700; font-size: 1.45em; }}
    h2 {{ color: {colors['text_primary']}; line-height: 1.32; margin: 16px 0 10px 0; font-weight: 700; font-size: 1.30em; }}
    h3 {{ color: {colors['text_primary']}; line-height: 1.32; margin: 16px 0 10px 0; font-weight: 700; font-size: 1.18em; }}
    h4 {{ color: {colors['text_primary']}; line-height: 1.32; margin: 16px 0 10px 0; font-weight: 700; font-size: 1.08em; }}
    h5, h6 {{ color: {colors['text_primary']}; line-height: 1.32; margin: 16px 0 10px 0; font-weight: 700; font-size: 1.00em; }}
    h1:first-child, h2:first-child, h3:first-child,
    h4:first-child, h5:first-child, h6:first-child {{
        margin-top: 0;
    }}
    ul, ol {{
        margin: 8px 0 12px 0;
        padding-left: 22px;
        padding-right: 22px;
    }}
    li {{
        margin: 4px 0;
        line-height: 1.65;
    }}
    li > p:first-child {{
        display: inline;
        margin: 0;
    }}
    li > p:not(:first-child) {{
        margin-top: 4px;
        margin-bottom: 0;
    }}
    a, a:visited {{
        color: {colors['link']};
        text-decoration: none;
        word-break: break-all;
    }}
    a {{ color: {colors['link']}; text-decoration: none; word-break: break-all; }}
    a:visited {{ color: {colors['link']}; text-decoration: none; word-break: break-all; }}
    a:hover {{
        text-decoration: underline;
    }}
    strong, b {{
        color: {colors['text_primary']};
        font-weight: 700;
    }}
    strong {{ color: {colors['text_primary']}; font-weight: 700; }}
    b {{ color: {colors['text_primary']}; font-weight: 700; }}
    em, i {{
        color: {colors['text_primary']};
    }}
    em {{ color: {colors['text_primary']}; }}
    i {{ color: {colors['text_primary']}; }}
    code {{
        background: {colors['surface_secondary']};
        border: 1px solid {colors['border']};
        border-radius: 5px;
        padding: 1px 6px;
        font-size: 0.92em;
        word-break: break-word;
    }}
    pre {{
        background: {colors['surface_secondary']};
        border: 1px solid {colors['border']};
        border-radius: 10px;
        padding: 12px 14px;
        overflow-x: hidden;
        word-break: break-word;
    }}
    pre code {{
        background: transparent;
        border: none;
        padding: 0;
        word-break: break-word;
    }}
    blockquote {{
        border-left: 3px solid {colors['accent']};
        border-right: 3px solid {colors['accent']};
        background: {colors['surface_secondary']};
        margin: 10px 0 12px 0;
        padding: 8px 14px;
        color: {colors['text_secondary']};
        border-radius: 6px;
    }}
    hr {{
        border: none;
        border-top: 1px solid {colors['border']};
        margin: 14px 0;
    }}
    table {{
        border-collapse: collapse;
        margin: 8px 0 12px 0;
        width: 100%;
    }}
    th, td {{
        border: 1px solid {colors['border']};
        padding: 6px 10px;
        text-align: left;
    }}
    th {{
        background: {colors['surface_secondary']};
        font-weight: 700;
    }}
    """

ANSWER_STYLESHEET = _build_answer_stylesheet(COLORS)


def _c(key: str, colors: dict[str, str]) -> str:
    return colors[key]


def build_qss(colors: dict[str, str] | None = None) -> str:
    """Build the full application stylesheet from the palette."""
    c = colors or COLORS
    return f"""
    QWidget {{
        font-family: "{FONT_FAMILY}";
        font-size: 14px;
        color: {_c('text_primary', c)};
    }}
    QMainWindow, #root {{
        background: {_c('background', c)};
    }}

    /* ---------- header ---------- */
    #header {{
        background: {_c('surface', c)};
        border-bottom: 1px solid {_c('border', c)};
    }}
    #appTitle {{
        font-size: 17px;
        font-weight: 700;
        color: {_c('text_primary', c)};
    }}
    #appSubtitle {{
        font-size: 12px;
        color: {_c('text_secondary', c)};
    }}
    #statusDot {{
        font-size: 12px;
        font-weight: 600;
        color: {_c('success', c)};
    }}
    #clearButton {{
        background: transparent;
        color: {_c('text_secondary', c)};
        border: 1px solid {_c('border', c)};
        border-radius: 14px;
        padding: 5px 14px;
        font-size: 13px;
    }}
    #clearButton:hover {{
        background: {_c('surface_secondary', c)};
        color: {_c('text_primary', c)};
    }}
    #clearButton:pressed {{ background: {_c('border', c)}; }}

    /* ---------- chat scroll ---------- */
    #chatScroll {{
        background: {_c('background', c)};
        border: none;
    }}
    #chatContainer {{
        background: {_c('background', c)};
    }}

    /* ---------- bubbles ---------- */
    QFrame#bubble_user {{
        background: {_c('user_bg', c)};
        border: 1px solid {_c('user_border', c)};
        border-radius: 16px;
        border-top-right-radius: 4px;
        color: {_c('text_primary', c)};
    }}
    QFrame#bubble_assistant {{
        background: {_c('assistant_bg', c)};
        border: 1px solid {_c('assistant_border', c)};
        border-left: 3px solid {_c('assistant_accent', c)};
        border-right: 3px solid transparent;
        border-radius: 16px;
        border-top-left-radius: 4px;
        color: {_c('text_primary', c)};
    }}
    QFrame#bubble_welcome {{
        background: {_c('surface', c)};
        border: 1px solid {_c('border', c)};
        border-radius: 16px;
        color: {_c('text_secondary', c)};
    }}
    QFrame#bubble_error {{
        background: {_c('error_bg', c)};
        border: 1px solid {_c('error_border', c)};
        border-left: 3px solid {_c('error', c)};
        border-right: 3px solid transparent;
        border-radius: 16px;
        border-top-left-radius: 4px;
        color: {_c('error', c)};
    }}
    QFrame#bubble_loading {{
        background: {_c('assistant_bg', c)};
        border: 1px solid {_c('assistant_border', c)};
        border-left: 3px solid {_c('assistant_accent', c)};
        border-right: 3px solid transparent;
        border-radius: 16px;
        border-top-left-radius: 4px;
        color: {_c('text_secondary', c)};
    }}
    #loadingLabel {{
        color: {_c('text_secondary', c)};
        font-size: 14px;
    }}
    #answerText {{
        background: transparent;
        border: none;
        color: {_c('text_primary', c)};
    }}

    /* ---------- sources ---------- */
    #sourcesToggle {{
        background: transparent;
        color: {_c('text_secondary', c)};
        border: 1px solid {_c('border', c)};
        border-radius: 13px;
        padding: 4px 14px;
        font-size: 12px;
        max-width: 160px;
    }}
    #sourcesToggle:hover {{
        background: {_c('surface_secondary', c)};
        color: {_c('accent', c)};
        border-color: {_c('accent', c)};
    }}
    #sourcesPanel {{
        background: {_c('surface_secondary', c)};
        border: 1px solid {_c('border', c)};
        border-radius: 10px;
    }}
    #sourcesHeading {{
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.4px;
        color: {_c('text_secondary', c)};
    }}
    #sourceRow {{
        font-size: 13px;
        color: {_c('text_primary', c)};
    }}

    /* ---------- feedback ---------- */
    #feedbackPrompt {{
        font-size: 11px;
        color: {_c('text_secondary', c)};
    }}
    #feedbackButton {{
        background: transparent;
        color: {_c('text_secondary', c)};
        border: 1px solid {_c('border', c)};
        border-radius: 12px;
        padding: 3px 12px;
        font-size: 11px;
    }}
    #feedbackButton:hover {{
        background: {_c('surface_secondary', c)};
        color: {_c('accent', c)};
        border-color: {_c('accent', c)};
    }}
    #feedbackButton:disabled {{
        color: {_c('text_secondary', c)};
        background: transparent;
        border-color: {_c('border', c)};
    }}
    #feedbackThanks {{
        font-size: 11px;
        color: {_c('success', c)};
    }}

    /* ---------- welcome ---------- */
    #welcome {{
        background: transparent;
    }}
    #welcomeTitle {{
        font-size: 24px;
        font-weight: 700;
        color: {_c('text_primary', c)};
    }}
    #welcomeSubtitle {{
        font-size: 15px;
        color: {_c('text_secondary', c)};
        line-height: 1.5;
    }}
    #suggestionButton {{
        background: {_c('surface', c)};
        color: {_c('text_primary', c)};
        border: 1px solid {_c('border', c)};
        border-radius: 14px;
        padding: 10px 18px;
        text-align: left;
        font-size: 14px;
        min-width: 260px;
        max-width: 480px;
    }}
    #suggestionButton:hover {{
        border-color: {_c('accent', c)};
        color: {_c('accent', c)};
        background: {_c('accent_soft', c)};
    }}

    /* ---------- input ---------- */
    #inputArea {{
        background: {_c('surface', c)};
        border-top: 1px solid {_c('border', c)};
    }}
    #chatInput {{
        background: {_c('input_bg', c)};
        color: {_c('input_text', c)};
        border: 1px solid {_c('input_border', c)};
        border-radius: 12px;
        padding: 10px 14px;
        font-size: 15px;
        placeholder-text-color: {_c('placeholder', c)};
        selection-background-color: {_c('accent', c)};
        selection-color: #FFFFFF;
        min-height: 28px;
        max-height: 160px;
    }}
    #chatInput:focus {{
        border: 2px solid {_c('accent', c)};
        padding: 9px 13px;
    }}
    #sendButton {{
        background: {_c('button_bg', c)};
        color: {_c('button_text', c)};
        border: none;
        border-radius: 12px;
        padding: 10px 22px;
        font-size: 15px;
        font-weight: 600;
        min-width: 76px;
        max-height: 52px;
    }}
    #sendButton:hover {{ background: {_c('button_hover', c)}; }}
    #sendButton:pressed {{ background: {_c('button_pressed', c)}; }}
    #sendButton:disabled {{
        background: {_c('disabled_button', c)};
        color: {_c('disabled_button_text', c)};
    }}
    #inputHint {{
        font-size: 11px;
        color: {_c('text_secondary', c)};
    }}

    /* ---------- jump to latest ---------- */
    #jumpButton {{
        background: {_c('surface', c)};
        color: {_c('text_primary', c)};
        border: 1px solid {_c('border_strong', c)};
        border-radius: 14px;
        padding: 7px 18px;
        font-size: 13px;
        font-weight: 500;
    }}
    #jumpButton:hover {{
        background: {_c('surface_secondary', c)};
        border-color: {_c('accent', c)};
        color: {_c('accent', c)};
    }}

    /* ---------- scrollbar ---------- */
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 3px;
    }}
    QScrollBar::handle:vertical {{
        background: {_c('scrollbar', c)};
        border-radius: 5px;
        min-height: 36px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {_c('scrollbar_hover', c)}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
    QScrollBar:horizontal {{ height: 0; }}

    /* ---------- focus / accessibility ---------- */
    QPushButton:focus {{ border: 2px solid {_c('accent', c)}; }}
    """