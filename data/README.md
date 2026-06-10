# NASA MDP Dataset

Place your raw NASA MDP CSV file here.

## Supported Dataset Files

The pipeline is compatible with all standard NASA MDP modules:

| File     | Module        | Rows  | Defect Rate |
|----------|---------------|-------|-------------|
| KC1.csv  | KC1 (C++)     | 2,109 | ~15.5%      |
| KC2.csv  | KC2           |   522 | ~20.5%      |
| PC1.csv  | PC1           | 1,109 | ~6.9%       |
| MC2.csv  | MC2           |   127 | ~35.4%      |
| JM1.csv  | JM1 (C)       | 10885 | ~19.3%      |

## Download Sources

1. **PROMISE Repository** (recommended):
   https://github.com/klainfo/NASADefectDataset

2. **PROMISE Data Repository** (direct CSV downloads):
   http://promise.site.uottawa.ca/SERepository/datasets-page.html

3. **Kaggle** – search "NASA PROMISE defect dataset"

## Expected Column Format

The CSV must contain numeric static code metrics and a `defects` column
with string values `'true'` / `'false'`.

Example columns:
    loc, v(g), ev(g), iv(g), n, v, l, d, i, e, b, t,
    lOCode, lOComment, lOBlank, lOCodeAndComment,
    uniq_Op, uniq_Opnd, total_Op, total_Opnd, branchCount, defects
