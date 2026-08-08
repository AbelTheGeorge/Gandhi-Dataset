# The Collected Works of Mahatma Gandhi (CWMG) Dataset

A structured JSON dataset of the *Collected Works of Mahatma Gandhi*, extracted from the 98-volume electronic edition published by the Publications Division, Government of India. The original CWMG project ran from September 1956 to October 2, 1994. This release converts it into a document-level corpus with metadata for dates, document types, footnotes, language provenance, and archival sources.

## Dataset Overview

| Statistic | Value |
| :--- | :--- |
| Total volumes | 98 |
| Total documents | 45,458 |
| Corpus size | 100.57 MB |
| Format | Structured JSON |
| Character count | 90,002,340 |
| Token count (NLTK) | 18,419,112 |
| Token count (BPE, tiktoken) | 21,094,055 |
| Time period | 1884 – 1948 |
| Document types | 5 |
| Original languages | 9 |
| Documents with an identifiable date | 45,141 (99.3%) |

Documents range from 9 to 958,739 characters, with a median of 678 and a mean of 1,573. Sentences run from 2 to 100 words, median 13, mean 15.7. Mean Flesch-Kincaid grade level is 7.01.

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
| `source` | String | Where the CWMG project obtained the original document. 6,224 documents have a missing or unknown source. Abbreviations are explained below. |

## Data Distribution

### Document categories

| Category | Count | Category | Count |
| :--- | ---: | :--- | ---: |
| LETTER | 30,076 | ARTICLE | 6,748 |
| TRANSCRIPT | 4,677 | TELEGRAM | 2,361 |
| PROSE | 1,596 | | |

Letters account for roughly 66% of the corpus.

### Original language

All text in the dataset is in English. This field records the language of composition, not the language of the released text.

| Language | Count | Language | Count | Language | Count |
| :--- | ---: | :--- | ---: | :--- | ---: |
| English | 39,465 | Urdu | 18 | Bengali | 1 |
| Gujarati | 5,048 | Marathi | 2 | Oriya | 1 |
| Hindi | 921 | Tamil | 1 | French | 1 |

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

Dates are drawn from the consolidated contents PDF and from document headers, then normalized to zero-padded numerals. Of the dated documents, 95.55% resolve to an exact day, 0.34% to a year alone, and 0.23% to a month.

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

The 1999 electronic edition was produced by OCR and was later found to contain errors and inconsistencies relative to the print volumes; page and volume numbers do not always match. It was chosen as the source because the 2005 digital edition does not contain machine-readable text.

The parsing pipeline relies on typographic and layout heuristics, so the corpus inherits OCR artifacts and pagination differences from the source. Rule-based cleaning and manual verification were applied, but inspecting all 45,458 documents individually is not feasible and residual transcription errors remain. Corrections submitted through the issue tracker will be incorporated into future releases.

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