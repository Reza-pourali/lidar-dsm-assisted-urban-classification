# Documented Python Results

## Input configuration

The final supervised classification experiment used:

- image size: 2100 x 2100 pixels;
- 7 input bands without DSM;
- 8 input bands with DSM;
- 5 target classes plus background label 0;
- 139,420 training pixels;
- 59,754 test pixels;
- 30% test split.

Target classes:

1. road
2. building
3. water
4. vegetation
5. work field

## Accuracy comparison

| Model | Features | Overall Accuracy | Macro F1 | Weighted F1 |
| --- | --- | ---: | ---: | ---: |
| Decision Tree | Without DSM | 92.46% | 0.91 | 0.92 |
| Random Forest | Without DSM | 95.00% | 0.94 | 0.95 |
| Decision Tree | With DSM | 98.24% | 0.97 | 0.98 |
| Random Forest | With DSM | 98.88% | 0.98 | 0.99 |

Documented improvement after adding DSM:

- Decision Tree: +5.78 percentage points
- Random Forest: +3.88 percentage points

## Random Forest feature importance with DSM

| Feature | Importance |
| --- | ---: |
| Band 1 | 16.85% |
| Band 2 | 16.44% |
| Band 3 | 26.30% |
| Band 4 | 1.99% |
| Band 5 | 3.92% |
| Band 6 | 1.35% |
| Band 7 | 0.84% |
| DSM | 32.31% |

DSM was the highest-importance individual feature in the documented Random
Forest model.

## Methodological note

The documented evaluation used a random pixel-level train/test split. In remote
sensing, nearby pixels can be spatially correlated, so random pixel splitting
may produce more optimistic accuracy estimates than a spatially separated
validation design.

The repository reproduces the original evaluation design for comparability, but
a spatial block split is recommended for future research-grade evaluation.
