# How Matter Learns to Continue

The reader companion to **Piotr Mikołajczyk's** book.

Start with a small example, change one assumption, and see what follows. This repository contains the programs, frozen data and worked checks behind the book's optional practical route. The book itself is not included.

## Download the companion

**[Download the checked v48 companion ZIP](https://github.com/pmikola/how-matter-learns-to-continue-companion/releases/download/companion-v48/how-matter-learns-to-continue-v48-companion.zip)** (about 1.1 MB).

Alternatively, browse the same files in [`companion/`](companion/). The release ZIP is the simplest way to keep the directory structure expected by the examples. Do not download only an individual script if it needs the accompanying data.

This first public release is bound to the v48 PDF. The calculations may remain applicable to later copyedited editions, but page references and PDF links are edition-specific. Check the release notes before using them with another edition.

## Start here

1. Extract the release ZIP.
2. Open the extracted folder and read `README.md`.
3. To explore without programming, open `where-to-return.html`, `reader-exercise.html` or `reader-variations.html` in a browser. The answer keys are separate.
4. To run the checks, install Python 3.11 or later and the supplied requirements.

For example, from a terminal in the extracted folder:

```text
python -m venv ../book-companion-venv
```

On Windows PowerShell:

```text
../book-companion-venv/Scripts/python.exe -m pip install -r requirements.txt
../book-companion-venv/Scripts/python.exe -B verify_companion.py
```

On macOS or Linux:

```text
../book-companion-venv/bin/python -m pip install -r requirements.txt
../book-companion-venv/bin/python -B verify_companion.py
```

The main verifier runs 35 portable checks. Further commands and their limitations are explained in the [companion README](companion/README.md). The portable requirements do not install PyTorch, TensorFlow, JAX or Keras.

## What you can investigate

- Nine-pixel pictures, adjustable connections, and the difference between making a prediction and learning from an error.
- Memory, noisy evidence, propagation and distinguishability.
- Pattern continuation, maintenance, coding and intervention.
- Simplified resource models that let us calculate when learning helps continuation and when it does not.
- The limits of transferring a prediction to a changed rule or environment, including preserved negative results.

These are specified mathematical and computational examples. They are not proof of a universal equation, autonomous life, or successful prediction in every environment.

## Integrity and history

The release ZIP has SHA-256:

```text
117eaa8c640f1a60b8e4810812413f3698858993081bb8b8663960a9ca67a00d
```

All 152 payload files here match that archive byte for byte. Its 151 manifest entries have been checked, with the manifest itself accounting for the remaining file. See [publication-integrity.json](publication-integrity.json) and [VALIDATION.md](VALIDATION.md).

Historical filenames, failed tests and provenance have not been renamed to make them appear new. Some archived documents describe an earlier local-only delivery. This repository and the `companion-v48` release now provide the public distribution route, without changing those historical records.

The archive includes selected Chapter 1 technical notes as evidence for code-snippet checks, not the full manuscript. Guide links point to PDF files supplied separately. Downloading this companion does not download the book.

## Report a problem

Use [Issues](https://github.com/pmikola/how-matter-learns-to-continue-companion/issues). Include the book and companion versions, chapter or example, command used, and relevant error message. Please do not upload the full book, private files, credentials or unrelated system logs.

No additional open-source license has been selected for this initial release.
