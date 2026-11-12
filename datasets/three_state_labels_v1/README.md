# Three-state pitch-contour labels for SMC2016

Version 1.0 · Released 12 November 2026 · Annotation author: Yukun Li

This dataset supplies additional **transitory, steady and modulating labels** for
3,123 selected segments in 12 Jingju recordings from Gong, Yang and Serra's
SMC2016 dataset. It is the reviewed label version used by the Chapter 3 study,
*Characterising Basic Elements in Sung Pitch Contours: A Flexible Framework for
Segmentation and Labelling*.

The CSV files contain our labels and the corresponding original segment row
numbers. Obtain the original segment boundaries and F0 tracks from the
[SMC2016 repository](https://github.com/jingju-SMC2016-PCS/SMC2016).
Audio recordings, F0 values and original annotation files are not redistributed
in this release.

## Labels

| Value | Name | Meaning |
| --- | --- | --- |
| `-1` | transitory | Voiced pitch movement that meets neither the steady nor the modulating definition. |
| `0` | steady | Approximately constant pitch, allowing small fluctuations without sustained quasi-periodic oscillation. |
| `1` | modulating | Recurrent, approximately periodic rises and falls around a local pitch centre. |

These labels describe pitch-contour regions. They are not the numerical values
in the middle column of Gong's original `*_refinedSeg.csv` files, and they are
not labels for automatically extracted peak-to-trough pitch contour units
(PCUs).

## Files and format

- `labels/train/`: label assignments for eight training recordings.
- `labels/development_test/`: label assignments for four development-test recordings.
- `recordings.csv`: recording-name mapping, split membership, source filenames,
  file checksums and counts.
- `restore_annotations.py`: combines the labels with your local copy of the
  original segmentation files. Python 3.9 or later; no external packages.

Each label CSV has a header and two columns:

| Column | Meaning |
| --- | --- |
| `source_row` | One-based row number in the corresponding original `*_refinedSeg.csv` file, which has no header. |
| `label` | One of `-1`, `0`, `1`, with the meanings above. |

Use `recordings.csv` to identify the correct source file. In particular,
`bcn_001`, `bcn_003`, `bcn_007` and `london_Laosheng_01` have different names in
the original repository. The source files must not be sorted or have rows
removed before joining them to these labels.

## Obtain the original data and restore timed annotations

The source revision verified for this release is
[`d965c28bd3266d764d9dd2ef663c7f494ee5fc3d`](https://github.com/jingju-SMC2016-PCS/SMC2016/tree/d965c28bd3266d764d9dd2ef663c7f494ee5fc3d).
Download that revision or clone the repository and check out this commit. The
needed files are under `dataset/groundtruth/`. Each source file's SHA-256 digest
is recorded in `recordings.csv`; the helper checks the segmentation files before
producing any output.

From this dataset directory, run:

```sh
python3 restore_annotations.py \
  --source-dir /path/to/SMC2016/dataset/groundtruth \
  --output-dir /path/to/new/restored_labels
```

The destination must not already exist. The helper writes 12 headerless CSVs
with the columns `start_time,label,duration`, in seconds, under `Train/` and
`Test/`. `Test` is the original directory name for the development-test split.
Every restored file must match its recorded checksum; the helper stops if an
input or output does not match. Original files are read only.

## Dataset size and provenance

| Split | Recordings | Transitory | Steady | Modulating | Total labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 8 | 1,441 | 722 | 185 | 2,348 |
| Development-test | 4 | 440 | 226 | 109 | 775 |
| Total | 12 | 1,881 | 948 | 294 | 3,123 |

The labels originate from one author's manual annotations. Labels were
transferred to Gong's segment intervals by overlapping duration. A targeted
single-author review of 14 boundary-sensitive assignments changed five labels;
this release includes those decisions. This was not a multiple-annotator
agreement study. The development-test recordings were used in earlier
exploratory work and are not an untouched independent test set.

Of the 3,148 original segment rows in these recordings, 25 had no overlap with
a valid manual label and are omitted. An absent row is unlabelled, not a
negative example. Four inherited overlaps were already cleaned by truncating
the earlier selected segment at the next selected segment's start. The helper
reproduces these four truncations. The August 2026 label review and this export
make no further boundary changes.

The 3,123 assignments are the input annotations, before the paper's F0-support
preparation and merging of adjacent equal labels. The C3-V16 analysis splits
intervals at unvoiced gaps, trims them to retained F0 support, closes internal
gaps at their midpoints, and merges touching equal labels within each voiced
stretch. That preparation produces **2,228 reference regions**: 1,688 in Train
and 540 in development-test. The reconstruction helper restores the input
annotations; it does not perform this later F0-based preparation or generate
the fitting PCUs.

## Attribution and licence

The original contributions in this directory, including the added labels,
documentation and reconstruction helper, are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See `LICENSE`.
This licence does not apply to the original SMC2016 data or to other parts of
the parent repository.

Suggested attribution for these labels:

> Li, Yukun (2026). Three-state pitch-contour labels for SMC2016. Version 1.0.

Also acknowledge the original segmentation dataset in research that uses it:

> Gong, Rong, Yang, Yile, and Serra, Xavier (2016). Pitch Contour Segmentation
> for Computer-aided Jingju Singing Training. Proceedings of the 13th Sound
> and Music Computing Conference, pp. 172–178.

Please identify the release version or Git commit when citing the labels.
