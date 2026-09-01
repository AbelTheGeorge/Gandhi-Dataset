# The Collected Works of Mahatma Gandhi (CWMG) Dataset

A structured JSON dataset of the *Collected Works of Mahatma Gandhi*, extracted from the 98-volume electronic edition published by the Publications Division, Government of India. The original CWMG project ran from September 1956 to October 2, 1994. This release converts it into a document-level corpus with metadata for dates, document types, footnotes, language provenance, and archival sources.

## Dataset Overview

| Statistic | Value |
| :--- | :--- |
| Total volumes | 98 |
| Total documents | 45,458 |
| Format | Structured JSON |
| Character count | 88,394,471 |
| Word count | 15,713,477 |
| Token count (NLTK) | 17,925,599 |
| Token count (BPE, tiktoken) | 20,436,331 |
| Time period | 1884 – 1948 |
| Document types | 5 |
| Original languages | 9 |
| Documents with an identifiable date | 45,141 (99.3%) |

Counts above cover the `contents`, `title`, `source`, and `footnotes` fields. The `contents` field alone holds 79,976,031 characters and 16,268,005 NLTK tokens.

Documents range from 9 to 958,739 characters, with a median of 740 and a mean of 1,759. The distribution is strongly right-skewed: 77% of documents are under 2,000 characters, while the longest 1,000 hold 22.7% of all characters. Sentences run from 1 to 488 words, median 13, mean 16.0. Mean Flesch-Kincaid grade level is 7.03, median 6.38.

The text holds 14,352,960 alphabetic word tokens and 78,599 distinct word types. Because the type-token ratio falls as text length grows, lexical diversity is reported as the moving-average type-token ratio over a 500-token window, which is 48.74%.

## Dataset Structure

The dataset is a single JSON file containing a list of objects, each representing one document.

### Schema

```jsonc
[
  {
    "volume": "75",
    "section": "277",
    "document_type": "LETTER",
    "document_date": "31-03-1939",
    "title": "LETTER TO NARAYANI DEVI",
    "contents": "NEW DELHI, March 31, 1939 DEAR SISTER, Keeping in view the present situation I think the satyagraha, or the preparations for it, going on in the Indian States should be suspended. It would be good, therefore, if in Mewar too the satyagraha was suspended.{1} Constructive work must of course go on and what I write these days should be studied. Yours, M. K. GANDHI SHRIMATI NARAYANI DEVI{2} MEWAR",
    "footnotes": {
      "1": "According to The Indian Annual Register, 1939, Vol I, this was done on April 4.",
      "2": "Secretary, Mewar Praja Parishad"
    },
    "original_language": "English",
    "source": "From a photostat of the Hindi: G.N. 9132"
  }
]
```

### Field Descriptions

| Field | Type | Description |
| :--- | :--- | :--- |
| `volume` | String | The volume number (1–98) in which the document appears. |
| `section` | String | The section number within the volume. Volume and section together form a stable, zero-padded unique identifier for cross-reference with the printed edition. |
| `document_type` | String | One of five categories: `LETTER`, `ARTICLE`, `TRANSCRIPT`, `TELEGRAM`, `PROSE`. See the assignment method below. |
| `document_date` | String | Normalized, zero-padded date at whatever resolution the source supports: `DD-MM-YYYY`, `MM-YYYY`, or `YYYY`. Uncertain dates carry a prefix (see below). 317 documents have no identifiable date. |
| `title` | String | The document title as given by the CWMG project. |
| `contents` | String | The main body of the text. Footnote references are embedded inline as `{1}`, `{2}`, and so on. |
| `footnotes` | Object | Maps each renumbered marker to its editorial note. Footnote numbering restarts on every page in the source; markers here are renumbered to be unique within each document. |
| `original_language` | String | The language in which the document was originally composed. All non-English documents appear here only in the English translation supplied by the print volumes. |
| `source` | String | Where the CWMG project obtained the original document. 225 documents have an empty or unusable source field. Abbreviations are explained below. |

## Data Distribution

### Document categories

| Category | Count | Category | Count |
| :--- | ---: | :--- | ---: |
| LETTER | 30,077 | ARTICLE | 6,748 |
| TRANSCRIPT | 4,677 | TELEGRAM | 2,360 |
| PROSE | 1,596 | | |

Letters account for roughly 66% of the corpus.

### Original language

All text in the dataset is in English. This field records the language of composition, not the language of the released text.

| Language | Count | Language | Count | Language | Count |
| :--- | ---: | :--- | ---: | :--- | ---: |
| English | 39,455 | Urdu | 18 | Bengali | 1 |
| Gujarati | 5,054 | Marathi | 2 | Oriya | 1 |
| Hindi | 925 | Tamil | 1 | French | 1 |

## `document_type` assignment

The five categories classify documents by communicative function:

- **LETTER** — personal and formal correspondence authored by Gandhi
- **ARTICLE** — texts written for publication or formal presentation
- **TRANSCRIPT** — records of speeches, discussions, interviews, and conversations
- **TELEGRAM** — cables and similar brief, formal communications
- **PROSE** — a residual category for texts that fit none of the above

Assignment is hybrid:

1. Titles that state the type, such as those beginning with "LETTER TO" or "CABLE TO", are handled by deterministic rules.
2. Remaining titles are passed to an open-source reasoning model (DeepSeek V4 Flash, via the LangChain framework), which reads the title and full text to infer the category.

This classification is reliable for clear-cut categories but imperfect at the boundaries, and PROSE in particular is heterogeneous by construction. Use with care if your research depends on exact genre labels.

## `document_date` conventions

Dates are drawn from the consolidated contents PDF and from document headers, then normalized to zero-padded numerals. Of the dated documents, 95.55% resolve to an exact day, 0.33% to a year alone, and 0.23% to a month.

Where the source records uncertainty, a prefix is attached to the normalized string rather than discarding the ambiguity:

- `ON OR BEFORE` — consolidates "Before" and "Prior to"
- `ON OR AFTER` — composition following a stated event
- `ON OR AROUND` — approximations, month ranges, and phrases such as "About" or "End of"

Prefixed exact dates account for 3.86% of the dated corpus. Dates appearing inside the `contents` field are left exactly as printed.

## `source` abbreviations

- **S. N.** — Sabarmati Sangrahalaya, Ahmedabad
- **G. N.** — Gandhi National Museum and Library, New Delhi
- **M.M.U.** — Reels of the Mobile Microfilm Unit
- **S. G.** — Photostats of the Sevagram Collection, held at the Gandhi National Museum and Library, New Delhi
- **C.W.** — Documents secured by the Collected Works of Mahatma Gandhi project
- **N. A. I.** — National Archives of India
- **C.S.O.** — Colonial Secretary's Office
- **C.O.** — Colonial Office
- **J&P** — Judicial and Public Records
- **Lt. G.** / **L.G.** — Lieutenant Governor

## Intended uses

- NLP research in social and historical contexts
- Historical text analysis and computational humanities
- Political discourse and rhetoric studies
- Stylometric and authorship-style analysis across genres
- Diachronic language study across a 64-year span
- Teaching and experimentation with historical text preprocessing

## Known limitations

The 1999 electronic edition contains errors and inconsistencies relative to the print volumes, and page and volume numbers do not always correspond. It was
chosen as the source because its publisher-generated text layer preserves the per-span typographic attributes — font, size, and style — that the parsing
pipeline relies on to separate primary text from editorial matter. The later scanned edition carries an OCR layer over page images, in which those
attributes are estimated rather than authentic and are not usable for this purpose.

The parsing pipeline relies on typographic and layout heuristics, so the corpus inherits text artifacts and pagination differences from the source. Rule-based
cleaning and manual verification were applied, but inspecting all 45,458 documents individually is not feasible and residual transcription errors
remain. Corrections submitted through the issue tracker will be incorporated into future releases.

## About the source material

The Government of India's CWMG project ran from 1956 to 1994 and produced 100 printed volumes.

Two digital editions exist. A 2005 digital version of the print collection is credited as:

> *The Collected Works of Mahatma Gandhi (digital)*, New Delhi, Publications Division, Government of India, 2005, 100 vols.

This dataset is derived from the earlier 1999 electronic edition:

> *The Collected Works of Mahatma Gandhi (Electronic Book)*, New Delhi, Publications Division, Government of India, 1999, 98 vols.

This dataset would not exist without the Publications Division's original CWMG project, the creators and maintainers of the 1999 electronic edition, and the historians, researchers, and archivists who preserved these writings.

## Source and terms

Gandhi's own writings are in the public domain in India. The volumes also contain editorial material that is not: document titles, footnotes, source statements, and the English translations of documents composed in other languages, which account for roughly 13% of the corpus. That material originates with the Publications Division and is attributed to it throughout the dataset.

The dataset is released for non-commercial research and educational use. Please cite the source edition above in any work that uses it. If you hold rights in any portion of this material and object to its inclusion, open an issue and it will be removed.

## Citation

Please cite the source edition:

> *The Collected Works of Mahatma Gandhi (Electronic Book)*, New Delhi, Publications Division, Government of India, 1999, 98 vols.

A link back to this repository is appreciated.

## Contact and contributions

For questions or to report errors, open an issue. Pull requests improving parsing, correcting metadata, or adding derived files such as cleaned-text variants or CSV exports are welcome.