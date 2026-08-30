# Obtaining the raw Freddie Mac SFLLD sample files

The raw loan-level files are **not committed to this repository**. They are distributed by
Freddie Mac under a licence that requires each user to register and accept terms directly,
and redistributing them would breach that licence — which the problem statement lists as a
disqualification condition (section 13, "Uses public data in violation of source terms").

Everything in `data/` is regenerated from these files by the pipeline, so the repository is
fully reproducible once you have them locally.

## What you need

Five vintage sample folders, placed under `dataset/`:

```
dataset/
  sample_2019/  sample_orig_2019.txt  sample_perf_2019.txt
  sample_2020/  sample_orig_2020.txt  sample_perf_2020.txt
  sample_2021/  sample_orig_2021.txt  sample_perf_2021.txt
  sample_2022/  sample_orig_2022.txt  sample_perf_2022.txt
  sample_2023/  sample_orig_2023.txt  sample_perf_2023.txt
```

Roughly 1.2 GB in total: 50,000 loans per vintage (250,000 loans) and 10,482,492 monthly
performance rows.

## Steps

1. Go to the Single-Family Loan-Level Dataset page:
   <https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset>
2. Register for a free account and accept the terms of use.
3. Sign in to the download portal:
   <https://claritydownload.fmapps.freddiemac.com/>
4. Download the **sample** files (not the full dataset) for vintages 2019 through 2023.
   Each vintage downloads as a zip containing one origination and one performance file.
5. Unzip each into `dataset/sample_<year>/`, matching the layout above.

## Verifying the download

The loader refuses to run against an unexpected layout rather than silently mis-mapping
columns. Check your files with:

```bash
python -c "from pathlib import Path; from src.data import sflld; \
print(sflld.verify_layout(Path('dataset')))"
```

This asserts **31 columns** in every origination file and **35** in every performance file.

## Why 31/35 and not 32/32

Freddie Mac's published layout (`file_layout.xlsx`, and the January 2026 General User Guide)
specifies 32 fields for both files. The sample files carry a different, undocumented
arrangement, verified empirically across all five vintages:

- **Origination has 31 columns.** `Servicer Name` (official position 25) is absent, so
  positions 25–31 correspond to official 26–32.
- **Performance has 35 columns.** Official positions 1–32 are unchanged, followed by
  33 = `MI Cancellation Indicator`, 34 = `Servicer Name`, 35 = an empty filler column.

Both relocated fields are time-varying, which is why they moved to the monthly file. The
full evidence for this mapping is documented in `src/data/sflld.py`.

## Macro series

The three macroeconomic series are separate, freely redistributable, and **are** vendored in
this repository under `data/external/` so the pipeline reproduces without network access:

| File | FRED series | Description |
|---|---|---|
| `fred_MORTGAGE30US.csv` | `MORTGAGE30US` | Freddie Mac PMMS 30-year fixed rate, weekly |
| `fred_UNRATE.csv` | `UNRATE` | BLS civilian unemployment rate, monthly |
| `fred_CSUSHPINSA.csv` | `CSUSHPINSA` | Case-Shiller U.S. National Home Price Index, monthly |

To refresh them:

```bash
for id in MORTGAGE30US UNRATE CSUSHPINSA; do
  curl -s "https://fred.stlouisfed.org/graph/fredgraph.csv?id=$id" \
    -o "data/external/fred_$id.csv"
done
```

## Running against real data

```bash
python -m src.data.build_from_sflld   # writes the data pack from dataset/
python -m src.data.macro_real         # writes real macro history + scenarios
python -m src.pipeline --skip-data    # runs Tasks 1-8 over the pack
```

If you do not have the raw files, the synthetic generator remains in the repository and
produces the same 33-column panel contract:

```bash
python -m src.pipeline                # builds the synthetic pack, then runs everything
```
