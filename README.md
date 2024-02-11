# RC-for-circadian-rhythm

このGithub レポジトリでは，2023年度卒業論文研究において行なった実験のコードと結果を公開しています．
ディレクトリ構造は次の通りです．

```text
RC-for-circadian-rhythm
├── codes
└── results
    ├── forecasts
    │   ├── data
    │   └── plot
    ├── hyperparamters
    │   ├── result_VDP_-7_4020_offline
    │   ├── result_VDP_0_4020_offline
    │   ├── result_VDP_7_4020_offline
    │   └── result_VDP_random_4020_offline
    └── simulations
```

`codes`には実験に用いたコードのうち主要なものを，`results`には実験の主要な結果を示しています．
特に，`codes/opt_analysis.ipynb`は，`codes/optuna.py`で生成する`result`ディレクトリで実行してください．

なお，Pythonモジュールのnoldsを用いたデータのサンプルエントロピーと散布図・ヒートマップに関するコードと結果は，ここでは公開していません．