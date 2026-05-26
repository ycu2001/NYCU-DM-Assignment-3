# How to Run
Download all files (or at least those marked as [required] below) and place `data.zip` in the same directory as `har.ipynb`.

## har.ipynb
* **How to train:** Run the cell under the **Train/LGBM/Best** section.
* **How to get predictions:** Run the cell under the **Predict_and_Output/Best_LGBM** section.
* Other cells include experimental methods and data exploration.
* Depending on your environment, you may need to adjust the file paths in the cells. If you are running this on Colab, you can place all the files in the `/Data_Mining/HW3` directory to avoid modifying any paths.

## File Discriptions
project-root/
├── requirements.txt                   
├── har.ipynb                          [required]  main
├── train_lgb.py                       [required]  train the best LGBM pipeline
├── predict_ensemble.py                [required]  LGBM pipeline inference
├── train_catboost.py                  [optional]  CatBoost experiments
├── predict_v2.py                      [optional]  LGBM + CatBoost ensemble
├── train.py                           [optional]  neural-network training
├── predict.py                         [optional]  neural-network ensemble
├── src/
│   └── har/
│       ├── __init__.py                [required]  package entry
│       ├── data.py                    [required]  data loading and cache building
│       ├── features.py                [required]  handcrafted feature extraction
│       ├── train_utils.py             [required]  grouped CV and evaluation helpers
│       └── model.py                   [optional]  neural-network model definition
├── data.zip                           

* [required] means needed to reproduce the current best method: Only LGBM.
* [optional] means useful for reference or previous experiments, but not needed for the main result.
