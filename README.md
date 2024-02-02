# RC-for-circadian-rhythm

このGithub レポジトリでは，2023年度卒業論文研究において行なった実験のコードと結果を公開しています．
ディレクトリ構造は次の通りです．

```text
RC-for-circadian-rhythm
├── Other works
│   └── 2010
├── determinitstic
│   ├── forecast
│   ├── generate_data
│   └── opt_hypparams
├── nolds
│   ├── plot
│   └── self_plot
├── random
│   ├── forecast
│   ├── generate_data
│   └── opt_hypparams
└── stats
    ├── nrmse_plot
    ├── opt_-7_VDP_gen
    ├── opt_-7_VDP_val
    ├── opt_0_VDP_gen
    ├── opt_0_VDP_val
    ├── opt_7_VDP_gen
    ├── opt_7_VDP_val
    ├── opt_random_VDP_gen
    ├── opt_random_VDP_val
    ├── plot
    ├── self_gen_-7
    ├── self_gen_0
    ├── self_gen_7
    ├── self_gen_random
    ├── self_plot
    ├── self_test_-7
    ├── self_test_0
    ├── self_test_7
    ├── self_test_random
    └── stddev
```

`deterministic`と`random`には、各時系列データの生成、Reservoir ComputerのHyperparametersの最適化、学習データの続きの予測のコードと結果が含まれています。
`stats`には、Reservoir Computerの学習データごとの異なる位相シフトのデータに対する予測結果と、統計的な量についてのコードと結果が含まれています。
`nolds`には、Pythonモジュールのnoldsを用いた結果が含まれています。
