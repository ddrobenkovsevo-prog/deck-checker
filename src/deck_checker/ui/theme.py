"""
ui/theme.py — Design tokens and Qt stylesheet for the Deck Checker kiosk UI.

Aesthetic direction: Industrial precision — dark background, sharp amber/green
accents, monospace typography. Like a high-end airport departure board meets
casino back-office equipment. Serious, readable at a glance, zero decoration
that doesn't serve a function.

Designed for 1024×600 touchscreen in a dimly lit casino back-office.
"""

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "bg_primary":    "#1E1040",   # near-black background
    "bg_secondary":  "#261650",   # card/panel background
    "bg_tertiary":   "#2E1E60",   # input/row background
    "border":        "#3A2870",   # subtle border
    "border_bright": "#5040A0",   # focused border

    "amber":         "#F5A623",   # primary accent — scanning, active
    "amber_dim":     "#7A5312",   # dimmed amber
    "green":         "#2ECC71",   # success
    "green_dim":     "#1A5C35",   # dimmed green
    "red":           "#E74C3C",   # error
    "red_dim":       "#6B1F1A",   # dimmed red
    "blue":          "#3498DB",   # info / manual validation
    "blue_dim":      "#1A3D5C",   # dimmed blue

    "text_primary":  "#E8EAF0",   # main text
    "text_secondary":"#8890A4",   # secondary / labels
    "text_dim":      "#4A5266",   # disabled / placeholder

    "white":         "#FFFFFF",
    "black":         "#000000",
}

# ── Typography ────────────────────────────────────────────────────────────────
FONTS = {
    "mono":    "Courier New",     # digits, card names, counts
    "sans":    "Arial",           # labels, buttons
    "size_xl": 48,
    "size_lg": 32,
    "size_md": 20,
    "size_sm": 14,
    "size_xs": 11,
}

# ── Dimensions ────────────────────────────────────────────────────────────────
SCREEN_W = 1024
SCREEN_H = 600
CORNER_R  = 8    # border radius px

STYLESHEET = f"""
/* ── Global ─────────────────────────────────────────────────────────────── */
QWidget {{
    background-color: {COLORS['bg_primary']};
    color: {COLORS['text_primary']};
    font-family: {FONTS['sans']};
    font-size: {FONTS['size_sm']}px;
    border: none;
    outline: none;
}}

QMainWindow {{
    background-color: {COLORS['bg_primary']};
}}

/* ── Labels ─────────────────────────────────────────────────────────────── */
QLabel {{
    background: transparent;
    color: {COLORS['text_primary']};
}}

QLabel#label_title {{
    font-family: {FONTS['mono']};
    font-size: {FONTS['size_xl']}px;
    font-weight: bold;
    color: {COLORS['white']};
    letter-spacing: 4px;
}}

QLabel#label_status {{
    font-family: {FONTS['mono']};
    font-size: {FONTS['size_lg']}px;
    font-weight: bold;
    letter-spacing: 2px;
}}

QLabel#label_count {{
    font-family: {FONTS['mono']};
    font-size: 72px;
    font-weight: bold;
    color: {COLORS['amber']};
}}

QLabel#label_subtitle {{
    font-size: {FONTS['size_md']}px;
    color: {COLORS['text_secondary']};
    letter-spacing: 1px;
}}

QLabel#label_info {{
    font-size: {FONTS['size_sm']}px;
    color: {COLORS['text_secondary']};
}}

QLabel#label_card {{
    font-family: {FONTS['mono']};
    font-size: {FONTS['size_md']}px;
    font-weight: bold;
    color: {COLORS['white']};
    background-color: {COLORS['bg_tertiary']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 4px 8px;
    min-width: 48px;
    qproperty-alignment: AlignCenter;
}}

/* ── Progress bar ───────────────────────────────────────────────────────── */
QProgressBar {{
    background-color: {COLORS['bg_tertiary']};
    border: 1px solid {COLORS['border']};
    border-radius: {CORNER_R}px;
    height: 24px;
    text-align: center;
    font-family: {FONTS['mono']};
    font-size: {FONTS['size_sm']}px;
    color: {COLORS['text_primary']};
}}

QProgressBar::chunk {{
    background-color: {COLORS['amber']};
    border-radius: {CORNER_R - 1}px;
}}

QProgressBar#progress_success::chunk {{
    background-color: {COLORS['green']};
}}

QProgressBar#progress_error::chunk {{
    background-color: {COLORS['red']};
}}

/* ── Buttons ────────────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {COLORS['bg_tertiary']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border_bright']};
    border-radius: {CORNER_R}px;
    font-size: {FONTS['size_md']}px;
    font-family: {FONTS['sans']};
    font-weight: bold;
    padding: 12px 28px;
    min-height: 56px;
    letter-spacing: 1px;
}}

QPushButton:hover {{
    background-color: {COLORS['border_bright']};
    border-color: {COLORS['amber']};
    color: {COLORS['amber']};
}}

QPushButton:pressed {{
    background-color: {COLORS['amber_dim']};
    border-color: {COLORS['amber']};
}}

QPushButton:disabled {{
    background-color: {COLORS['bg_secondary']};
    color: {COLORS['text_dim']};
    border-color: {COLORS['border']};
}}

QPushButton#btn_primary {{
    background-color: {COLORS['amber_dim']};
    border-color: {COLORS['amber']};
    color: {COLORS['amber']};
}}

QPushButton#btn_primary:hover {{
    background-color: {COLORS['amber']};
    color: {COLORS['black']};
}}

QPushButton#btn_success {{
    background-color: {COLORS['green_dim']};
    border-color: {COLORS['green']};
    color: {COLORS['green']};
}}

QPushButton#btn_success:hover {{
    background-color: {COLORS['green']};
    color: {COLORS['black']};
}}

QPushButton#btn_danger {{
    background-color: {COLORS['red_dim']};
    border-color: {COLORS['red']};
    color: {COLORS['red']};
}}

QPushButton#btn_danger:hover {{
    background-color: {COLORS['red']};
    color: {COLORS['white']};
}}

/* ── Frames / panels ────────────────────────────────────────────────────── */
QFrame#panel {{
    background-color: {COLORS['bg_secondary']};
    border: 1px solid {COLORS['border']};
    border-radius: {CORNER_R}px;
}}

QFrame#panel_success {{
    background-color: {COLORS['green_dim']};
    border: 2px solid {COLORS['green']};
    border-radius: {CORNER_R}px;
}}

QFrame#panel_error {{
    background-color: {COLORS['red_dim']};
    border: 2px solid {COLORS['red']};
    border-radius: {CORNER_R}px;
}}

QFrame#panel_scanning {{
    background-color: {COLORS['bg_secondary']};
    border: 2px solid {COLORS['amber']};
    border-radius: {CORNER_R}px;
}}

QFrame#divider {{
    background-color: {COLORS['border']};
    max-height: 1px;
    min-height: 1px;
}}

/* ── List widget (missing/extra cards) ──────────────────────────────────── */
QListWidget {{
    background-color: {COLORS['bg_tertiary']};
    border: 1px solid {COLORS['border']};
    border-radius: {CORNER_R}px;
    font-family: {FONTS['mono']};
    font-size: {FONTS['size_sm']}px;
    color: {COLORS['text_primary']};
}}

QListWidget::item {{
    padding: 6px 12px;
    border-bottom: 1px solid {COLORS['border']};
}}

QListWidget::item:selected {{
    background-color: {COLORS['blue_dim']};
    color: {COLORS['blue']};
}}

/* ── Scroll bar ─────────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {COLORS['bg_tertiary']};
    width: 8px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical {{
    background: {COLORS['border_bright']};
    border-radius: 4px;
    min-height: 20px;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* ── Combo box ──────────────────────────────────────────────────────────── */
QComboBox {{
    background-color: {COLORS['bg_tertiary']};
    border: 1px solid {COLORS['border_bright']};
    border-radius: {CORNER_R}px;
    padding: 8px 12px;
    font-size: {FONTS['size_sm']}px;
    color: {COLORS['text_primary']};
    min-height: 40px;
}}

QComboBox:hover {{
    border-color: {COLORS['amber']};
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['bg_secondary']};
    border: 1px solid {COLORS['border_bright']};
    selection-background-color: {COLORS['amber_dim']};
    color: {COLORS['text_primary']};
}}

/* ── Tool tip ───────────────────────────────────────────────────────────── */
QToolTip {{
    background-color: {COLORS['bg_secondary']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border_bright']};
    padding: 4px 8px;
    font-size: {FONTS['size_xs']}px;
}}
"""
