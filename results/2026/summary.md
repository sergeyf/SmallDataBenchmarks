106 datasets (17 with fewer than 250 rows).

|                                |   mean AUROC |   median AUROC |   mean AUROC (n < 250) |   best or within 0.5% |   worst or within 0.5% |   median seconds per dataset |
|:-------------------------------|-------------:|---------------:|-----------------------:|----------------------:|-----------------------:|-----------------------------:|
| TabPFN-3.5                     |        0.911 |          0.942 |                  0.932 |                    87 |                      0 |                          2.8 |
| TabICLv2                       |        0.908 |          0.940 |                  0.930 |                    79 |                      0 |                          5.6 |
| MLJAR (2020, 300s)             |        0.895 |          0.928 |                  0.915 |                    36 |                      4 |                       1258.2 |
| AutoGluon (2020, 300s)         |        0.893 |          0.929 |                  0.916 |                    32 |                      2 |                       1194.1 |
| AutoGluon 1.6 (60s)            |        0.892 |          0.929 |                  0.916 |                    41 |                      0 |                        264.3 |
| MLJAR 1.3 (60s)                |        0.890 |          0.921 |                  0.902 |                    27 |                      7 |                        247.9 |
| LightGBM (hyperopt, 50 trials) |        0.883 |          0.920 |                  0.905 |                    28 |                      8 |                        134.4 |
| CatBoost (default)             |        0.879 |          0.923 |                  0.914 |                    28 |                      8 |                         14.3 |
| Random Forest                  |        0.878 |          0.912 |                  0.908 |                    19 |                     12 |                          2.1 |
| LightGBM (default, 1000 trees) |        0.859 |          0.913 |                  0.869 |                    19 |                     44 |                          4.0 |
| SVC                            |        0.847 |          0.886 |                  0.860 |                    14 |                     36 |                          4.9 |
| Logistic Regression            |        0.841 |          0.883 |                  0.857 |                    17 |                     43 |                          1.1 |

Timing: one outer fold of 30 datasets, medians.

| model                          |   cpu median fit_seconds |   gpu median fit_seconds |   cpu median predict_seconds |   gpu median predict_seconds |   cpu median predict_ms_per_row |   gpu median predict_ms_per_row |
|:-------------------------------|-------------------------:|-------------------------:|-----------------------------:|-----------------------------:|--------------------------------:|--------------------------------:|
| TabPFN-3.5                     |                   0.2027 |                   0.2538 |                    8.53      |                     0.3619   |                       37.38     |                         2.126   |
| Logistic Regression            |                   0.264  |                 nan      |                    0.0005446 |                   nan        |                        0.001919 |                       nan       |
| TabICLv2                       |                   0.5613 |                   0.5423 |                    2.623     |                     0.3491   |                       11.96     |                         1.167   |
| Random Forest                  |                   0.5635 |                 nan      |                    0.01012   |                   nan        |                        0.03788  |                       nan       |
| LightGBM (default, 1000 trees) |                   0.744  |                 nan      |                    0.0034    |                   nan        |                        0.01932  |                       nan       |
| SVC                            |                   0.9392 |                 nan      |                    0.002816  |                   nan        |                        0.01516  |                       nan       |
| CatBoost (default)             |                   2.578  |                  26.35   |                    0.004322  |                     0.005054 |                        0.01671  |                         0.01758 |
| LightGBM (hyperopt, 50 trials) |                  23.04   |                 nan      |                    0.003849  |                   nan        |                        0.01887  |                       nan       |
| AutoGluon 1.6 (60s)            |                  65.27   |                  82.23   |                    1.047     |                     0.7919   |                        4.453    |                         2.68    |
