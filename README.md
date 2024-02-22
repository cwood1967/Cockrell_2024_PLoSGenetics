# Cockrell_2024_PLoSGenetics

The repository contains code used for image processing in the paper by Lexy
Cockrell: "Regulators of rDNA morphology in fission yeast."

All code is written as python files and jupyter (python) notebooks.

- `analysis` : code to process folders of images.
- `InferTimeLapse` : code for measuring properties while tracking divisions over time.
- `cellfinder` : code for inferring and training a torch mask rcnn model.
- `CellProps` : code to find cell length of segented pombe cells.
- `Notebooks` : jupyer notebooks used to combine image processing measrements with other data.

