"""The app's look, in one place.

Streamlit themes globally: `.streamlit/config.toml` sets one accent, a ground, a
widget fill, the borders and the two fonts, and every widget inherits from them.
That config is most of the design. What it cannot express is here, and it is
deliberately short - each rule below exists because the default is wrong for a
document-generation tool, not to re-skin Streamlit's widgets one by one:

  * a reading measure. `layout="wide"` is needed for the 5-column weighting
    grids, but a form at 2000px wide is unusable, so the content is capped and
    centred instead of filling the monitor.
  * numbered sections. The flow is a sequence - source documents, unit,
    generate - and the trainer needs to see where they are in it. `st.header`
    gives a bare line; `section()` gives the number its own mark.
  * the footer, lifted out of app.py unchanged.

Anything here that reaches into Streamlit's own class names is avoided: those
change between releases. The selectors used are the stable ones (`.stAppHeader`,
`.block-container`, `[data-testid]`), and if one stops matching, the app loses a
margin and nothing else.
"""

from __future__ import annotations

import streamlit as st

# Kept in step with .streamlit/config.toml - the few places CSS has to name a
# colour itself.
INK = "#1C2320"
MUTED = "#6B7066"
ACCENT = "#1F5C4A"
BORDER = "#E8E2D4"
PAPER = "#FAF8F4"

_CSS = f"""
<style>
  /* A reading measure. Wide layout is for the weighting grids, not for forms. */
  .block-container {{
    max-width: 1320px;
    padding-top: 2.5rem;
    padding-bottom: 5rem;
  }}

  /* Streamlit's own toolbar strip sits above the title; it needs no ground. */
  .stAppHeader {{ background: transparent; }}

  /* The page title block. */
  .lpg-title {{
    text-align: center;
    margin: 0 0 0.25rem;
    font-weight: 600;
    letter-spacing: -0.01em;
  }}
  .lpg-subtitle {{
    text-align: center;
    color: {MUTED};
    font-size: 0.9rem;
    margin: 0 0 1.75rem;
  }}

  /* A numbered section heading: the step number carries its own mark. */
  .lpg-section {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 2.25rem 0 0.35rem;
  }}
  .lpg-section-num {{
    flex: none;
    width: 1.55rem;
    height: 1.55rem;
    border-radius: 50%;
    background: {ACCENT};
    color: #fff;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    justify-content: center;
  }}
  .lpg-section-title {{
    margin: 0;
    font-size: 1.3rem;
    font-weight: 600;
    color: {INK};
  }}
  /* A step inside a section (the assessment path's eight) sits a level down. */
  .lpg-step .lpg-section-num {{
    width: 1.35rem;
    height: 1.35rem;
    font-size: 0.72rem;
  }}
  .lpg-step .lpg-section-title {{ font-size: 1.1rem; }}

  /* Metrics report the unit's size; boxed, they read as four tiles. */
  [data-testid="stMetric"] {{
    background: #fff;
    border: 1px solid {BORDER};
    border-radius: 0.6rem;
    padding: 0.85rem 1rem;
  }}
  [data-testid="stMetricLabel"] p {{
    font-size: 0.72rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: {MUTED};
  }}

  /* Bordered containers are the app's cards; lift them off the ground. */
  [data-testid="stVerticalBlockBorderWrapper"] {{ background: #fff; }}

  /* Expanders are used for the long extraction dumps - quieten the closed state. */
  details[data-testid="stExpander"] summary {{ font-size: 0.9rem; }}

  .app-footer {{
    position: fixed; left: 0; bottom: 0; width: 100%;
    text-align: center; color: {MUTED}; font-size: 0.8rem; padding: 8px 0;
    border-top: 1px solid {BORDER};
    background-color: rgba(250, 248, 244, 0.85);
    -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px);
    z-index: 1000;
  }}
</style>
"""


def style() -> None:
    """Install the stylesheet. Called once, immediately after set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    """The centred page title, and the line of provenance under it."""
    st.markdown(f"<h1 class='lpg-title'>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p class='lpg-subtitle'>{subtitle}</p>",
                    unsafe_allow_html=True)


def section(number: int | str, title: str, *, step: bool = False) -> None:
    """A numbered heading: `section(1, "Source documents")`.

    `step=True` for a step within a path (the assessment tool's eight), which
    sits a level below a section in the hierarchy.
    """
    cls = "lpg-section lpg-step" if step else "lpg-section"
    st.markdown(
        f"<div class='{cls}'>"
        f"<span class='lpg-section-num'>{number}</span>"
        f"<h2 class='lpg-section-title'>{title}</h2>"
        f"</div>",
        unsafe_allow_html=True)


def footer(text: str = "Made with &#10084;&#65039; by Musumbi &#128081;") -> None:
    """The fixed strip at the foot of the window."""
    st.markdown(f"<div class='app-footer'>{text}</div>",
                unsafe_allow_html=True)
