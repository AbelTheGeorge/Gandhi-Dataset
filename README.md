# The Collected Works of Mahatma Gandhi (CWMG) Dataset

A structured JSON dataset of the *Collected Works of Mahatma Gandhi*, extracted from the 98-volume electronic edition published by the Publications Division, Government of India. The original CWMG project ran from September 1956 to October 2, 1994. This release converts it into a document-level corpus with metadata for dates, document types, footnotes, language provenance, and archival sources.

## Paper

The dataset is described in *Collected Works of Mahatma Gandhi (CWMG) Dataset: A Metadata-Dense Dataset for Computational Humanities and Social Sciences*, accepted at CODS 2026. The full version, with an appendix covering the curation pipeline, representative records, and reproducibility details, is in [`Paper/CWMG_Dataset_Full_Paper.pdf`](Paper/CWMG_Dataset_Full_Paper.pdf). A citation will be added once the proceedings are published.

## Dataset Overview

| Statistic | Value |
| :--- | :--- |
| Total volumes | 98 |
| Total documents | 45,458 |
| Format | Structured JSON |
| Character count | 88,775,422 |
| Word count | 15,756,865 |
| Token count (NLTK) | 18,011,138 |
| Token count (BPE, tiktoken `o200k_base`) | 20,532,358 |
| Time period | 1884 – 1948 |
| Document types | 5 |
| Original languages | 9 |
| Documents with an identifiable date | 45,160 (99.3%) |

Counts above cover the `contents`, `title`, `source`, and `footnotes` fields. The `contents` field alone holds 80,279,629 characters.

Documents range from 30 to 957,864 characters, with a median of 744 and a mean of 1,766. The distribution is strongly right-skewed: 77% of documents are under 2,000 characters, while the longest 1,000 hold 22.7% of all characters. Sentences run from 1 to 488 words, median 13, mean 16.1. Mean Flesch-Kincaid grade level is 7.07, median 6.42.

The text holds 14,378,073 alphabetic word tokens and 72,032 distinct word types. Because the type-token ratio falls as text length grows, lexical diversity is reported as the moving-average type-token ratio over a 500-token window, which is 48.63%.

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
    "contents": "NEW DELHI, March 31, 1939 DEAR SISTER, Keeping in view the present situation I think the satyagraha, or the preparations for it, going on in the Indian States should be suspended. It would be good, therefore, if in Mewar too the satyagraha was suspen-ded.{1} Constructive work must of course go on and what I write these days should be studied. Yours, M. K. GANDHI SHRIMATI NARAYANI DEVI{2} MEWAR",
    "footnotes": {
      "1": "According to The Indian Annual Register, 1939, Vol I, this was done on April 4.",
      "2": "Secretary, Mewar Praja Parishad"
    },
    "original_language": "Hindi",
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
| `document_date` | String | Normalized, zero-padded date at whatever resolution the source supports: `DD-MM-YYYY`, `MM-YYYY`, or `YYYY`. Uncertain dates carry a prefix (see below). 298 documents have no identifiable date. |
| `title` | String | The document title as given by the CWMG project. |
| `contents` | String | The main body of the text. Footnote references are embedded inline as `{1}`, `{2}`, and so on. |
| `footnotes` | Object | Maps each renumbered marker to its editorial note. Footnote numbering restarts on every page in the source; markers here are renumbered to be unique within each document. |
| `original_language` | String | The language in which the document was originally composed. All non-English documents appear here only in the English translation supplied by the print volumes. |
| `source` | String | Where the CWMG project obtained the original document. 130 documents have no source statement. Abbreviations are explained below. |

## Data Distribution

### Document categories

| Category | Count | Category | Count |
| :--- | ---: | :--- | ---: |
| LETTER | 29,875 | ARTICLE | 6,877 |
| TRANSCRIPT | 4,735 | TELEGRAM | 2,303 |
| PROSE | 1,668 | | |

Letters account for roughly 66% of the corpus.

### Original language

The text is in English throughout, apart from short Devanagari passages in 14 documents that account for less than 0.01% of all characters. This field records the language of composition, not the language of the released text.

| Language | Count | Language | Count | Language | Count |
| :--- | ---: | :--- | ---: | :--- | ---: |
| English | 23,710 | Urdu | 170 | Tamil | 3 |
| Gujarati | 16,451 | Marathi | 4 | Oriya | 1 |
| Hindi | 5,114 | Bengali | 4 | French | 1 |

The language is taken from the editors' translation notices, such as "From Gujarati", and, where no notice is given, from source statements that name the language of the original, such as "From a microfilm of the Gujarati". Weaker cues, such as the name of a vernacular periodical, were reviewed by hand.

## `document_type` assignment

The five categories classify documents by communicative function:

- **LETTER** — personal and formal correspondence authored by Gandhi
- **ARTICLE** — texts written for publication or formal presentation
- **TRANSCRIPT** — records of speeches, discussions, interviews, and conversations
- **TELEGRAM** — cables and similar brief, formal communications
- **PROSE** — a residual category for texts that fit none of the above

Gandhi's writings span a wide range of forms and states of completeness, so sorting them into fixed categories is difficult. We therefore reserve LETTER, ARTICLE, TRANSCRIPT and TELEGRAM for documents whose type is clear, and assign everything else to PROSE by default. PROSE is the smallest category, but also the most mixed.

Labels are assigned by an ordered set of deterministic rules over each document's title, source, contents and footnotes, with written policies for ambiguous forms: for example, a title beginning "LETTER TO" or "CABLE TO", a salutation and signature, a third-person report of a speech, or publication in one of Gandhi's own periodicals.

As a check, a language model (DeepSeek V4 Flash) suggested labels for the 12,887 documents whose title does not state a type, and agreed with the rules on 87.8% of the 12,670 the rules labeled. Of the 1,545 disagreements (3.4% of the corpus), 225 were labeled individually by hand and 1,320 were settled by written policies decided after reviewing the cases. In all, 456 documents (1.0%), including the 217 no rule covered, were labeled individually by hand. Language-model suggestions are never used as final labels.

Readers should bear this in mind when using the dataset: the labels are reliable for clear-cut documents but less certain at the boundaries between categories. Contributions toward a more detailed classification scheme are welcome.

## `document_date` conventions

Dates are drawn from the consolidated contents PDF and from document headers, then normalized to zero-padded numerals. Of the dated documents, 95.41% resolve to an exact day, 0.33% to a month, and 0.31% to a year alone.

Where the source records uncertainty, a prefix is attached to the normalized string rather than discarding the ambiguity:

- `ON OR BEFORE` — consolidates "Before" and "Prior to"
- `ON OR AFTER` — composition following a stated event
- `ON OR AROUND` — approximations, month ranges, and phrases such as "About" or "End of"

Prefixed exact dates account for 3.89% of the dated corpus, and prefixed partial dates for the remaining 0.07%. Dates appearing inside the `contents` field are left exactly as printed.

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

## Analysis code

The `Analysis` folder holds the code and selected results for the analyses reported in the accompanying paper. All scripts read `The-Collected-Works-Mahatma-Gandhi.json` from the repository root. Package versions are listed in `Analysis/requirements.txt`.

### `Analysis/Supervised`

Three classification tasks: diachronic era prediction (five periods, dates in the text masked), translation detection (translated vs. composed in English), and document type classification. Each uses the `contents` field only and a stratified 80/10/10 split fixed by seed 42.

| File | Purpose |
| :--- | :--- |
| `cwmg_common.py` | Shared code: data loading, date masking, chunking, training and evaluation |
| `train_era.py`, `train_translation.py`, `train_type.py` | Fine-tune ModernBERT-base (8,192-token context) with seeds 42, 43 and 44 |
| `tfidf_baseline.py` | TF-IDF and logistic regression baseline on the same splits |
| `confusion_mat.py` | Combined confusion-matrix figure, summed over seeds |

Results are in `Supervised/results`:

- `<task>/` for each of `era`, `translation`, `type`: run configuration, data diagnostics, software environment, the train/validation/test split, a summary across seeds, and the confusion matrix summed over seeds. Each `seed_<n>/` folder holds the classification report, test metrics, test predictions and the per-epoch training log.
- `tfidf/`: the baseline's configuration, summary, per-task metrics and test predictions. The baseline is deterministic, so one seed's predictions are included.
- `combined_confusion_matrices.pdf`.

| Task | ModernBERT Macro F1 (mean ± sd, 3 seeds) | TF-IDF Macro F1 |
| :--- | :---: | :---: |
| Era prediction | 0.848 ± 0.003 | 0.811 |
| Translation detection | 0.972 ± 0.001 | 0.961 |
| Document type | 0.929 ± 0.005 | 0.835 |

Run the scripts from `Analysis/Supervised`, for example `python train_type.py --smoke` for a short check, then `python train_type.py`. Each task trains three seeds on one GPU; `python tfidf_baseline.py` runs on CPU and uses the split files in `results/<task>/`.

### `Analysis/Unsupervised`

Topic analysis over document embeddings.

| File | Purpose |
| :--- | :--- |
| `embed_documents.py` | Embeds every document with Qwen3-Embedding-8B; documents longer than 8,192 tokens are split into overlapping windows and averaged |
| `topic_analysis.py` | Nine clustering settings (HDBSCAN, and k-means with 20, 40 or 60 clusters on UMAP-reduced or full embeddings), each over ten seeds; topic terms, NPMI coherence, topic diversity and seed agreement |
| `TopicFigures.py` | Topic summary, Ward grouping of topics and the intertopic distance map for the reported setting |

Results are in `Unsupervised/results`:

- `embeddings_qwen3-8B/summary_by_setting.csv`: the comparison of the nine settings, averaged over seeds.
- `embeddings_qwen3-8B/topics_kmeans_umap_k40.csv`: topics of the reported setting, k-means with 40 clusters on UMAP-reduced embeddings.
- `kmeans_umap_k40/`: per-topic summary, the grouping of topics into three groups and its report, and the intertopic distance map.

The embeddings (about 750 MB) are not included; `embed_documents.py` regenerates them on one GPU in roughly an hour. Run `python embed_documents.py`, then `python topic_analysis.py --embeddings embeddings_qwen3-8B`, then `python TopicFigures.py`, from `Analysis/Unsupervised`.

## Intended uses

- NLP research in social and historical contexts
- Historical text analysis and computational humanities
- Political discourse and rhetoric studies
- Stylometric and authorship-style analysis across genres
- Diachronic language study across a 64-year span
- Teaching and experimentation with historical text preprocessing

## Known limitations

The dataset is a computational resource rather than a critical edition. The 1999 electronic edition on which it is based contains omissions, transcription errors and missing pages relative to the print volumes, and its page and volume numbers do not always correspond. It was chosen as the source because its publisher-generated text layer preserves the per-span typographic attributes — font, size, and style — that the parsing pipeline relies on to separate primary text from editorial matter. Scanned releases of the print series carry an OCR layer over page images, in which those attributes are estimated rather than recorded and are not usable for this purpose.

Rather than inherit the errors of the electronic text, we worked to bring it closer to the print series: recurring transcription errors were corrected, and documents missing from the electronic edition were added by hand, with the page images of the print series treated as the authority where the two differ. A complete collation against the print series was beyond the scope of this work, so residual errors and extraction artifacts remain, and work that requires exact correspondence with the print text should consult the printed volumes. Corrections submitted through the issue tracker will be incorporated into future releases, and work is underway on a release built directly from the page images of the print series.

Documents composed in other languages, 47.8% of the corpus, appear only in the English translations supplied by the volumes, so the dataset records much of Gandhi's writing as mediated by his editors.

## About the source material

The Government of India's CWMG project ran from 1956 to 1994 and produced 100 printed volumes.

Two digital editions exist. This dataset is derived from the 1999 electronic edition, a revised version of the print series rearranged into 98 volumes:

> *The Collected Works of Mahatma Gandhi (Electronic Book)*, New Delhi, Publications Division, Government of India, 1999, 98 vols.

Omissions and alterations relative to the print series were reported after its release, and the Government of India withdrew it in 2005. The digital version of the original print series is credited as:

> *The Collected Works of Mahatma Gandhi (digital)*, New Delhi, Publications Division, Government of India, 2005, 100 vols.

This dataset would not exist without the Publications Division's original CWMG project, the creators and maintainers of the 1999 electronic edition, and the historians, researchers, and archivists who preserved these writings.

## Source and terms

Gandhi's own writings are in the public domain in India. The volumes also contain editorial material that is not: document titles, footnotes, source statements, and the English translations of documents composed in other languages, which account for 47.8% of the corpus. That material originates with the Publications Division and is attributed to it throughout the dataset.

The dataset is released for non-commercial research and educational use. Please cite the source edition above in any work that uses it. If you hold rights in any portion of this material and object to its inclusion, open an issue and it will be removed.

## Citation

Please cite the source edition:

> *The Collected Works of Mahatma Gandhi (Electronic Book)*, New Delhi, Publications Division, Government of India, 1999, 98 vols.

A link back to this repository is appreciated.

## Contact and contributions

For questions or to report errors, open an issue. Pull requests improving parsing, correcting metadata, or adding derived files such as cleaned-text variants or CSV exports are welcome.
