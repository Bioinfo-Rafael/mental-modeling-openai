# 概要

[AirCombat-WEZ](https://github.com/andrekuros/AirCombat-WEZ)、commit `5ae42f4e56362e4df2bf29c5c01cbfb50c5795ed`。CC0 1.0。4 input CSVに加えて8 Paper_Results CSVも全件解析した。

2026-09-09のdownload済み実ファイル全走査。元データは変更していない。

# ファイル構成

| format | file数 |
| --- | --- |
| .csv | 12 |
| .ipynb | 1 |
| .md | 1 |
| .py | 1 |
| .txt | 1 |
| no_extension | 1 |

全filenameとbytesは `outputs/dataset_inventory/candidate_files.csv`（datasetでfilter）に記録。schema、file別missing/unique/type・shape、archive treeは `candidate_schema.json`。support file（code/model/media/PDF）はmetadata inventoryのみで、flight dataとして実行/deserializeしていない。上流READMEは原文保存し、このSCHEMA.mdを日本語の調査結果とする。

# データ量

| 量 | 値 |
| --- | --- |
| file_count | 17 |
| disk_bytes_excluding_git_cache | 462715 |
| compressed_bytes_all_zip_including_nested | 0 |
| compressed_bytes_top_level | 0 |
| extracted_bytes_excluding_nested_zip | 462715 |
| format_counts | {'.csv': 12, 'no_extension': 1, '.md': 1, '.ipynb': 1, '.py': 1, '.txt': 1} |
| schema_group_count | 12 |
| errors | [] |

bytesはファイル内容のサイズ合計で、filesystem allocation（du）ではない。`.git`, `.cache` と自作wrapper文書を除外。top-level ZIP、nested ZIP、展開物を区別しており、ZIPの展開後サイズを現在のdisk bytesへ重複加算しない。Git取得でZIPが0なのは圧縮配布物を保存していない意味で、Git pack size=0の主張ではない。

# Schema

schema fingerprintは列名順序＋reader dtype＋delimiter/header等の構造からSHA256の先頭16桁を計算。CSVの行数・missing数をfingerprintへ混ぜない。MATはvariable shapeを含む。dtypeはCSVにnative保存された型ではなくreaderの推定結果。空fileは未知schemaとして独立。文字列/全欠損/数値型の差も分離する。

| schema group | file数 | 物理rows合計 | kind | 例filename |
| --- | --- | --- | --- | --- |
| aircombat_wez:15ed781ce07cec18 | 1 | 864 | delimited | repository/Data/FactorialExperiment.csv |
| aircombat_wez:3400173d0f20724e | 1 | 1000 | delimited | repository/Data/RandomExperiment_1000_NEZ.csv |
| aircombat_wez:616c9b29c95caed1 | 1 | 1000 | delimited | repository/Data/RandomExperiment_1000_RMAX.csv |
| aircombat_wez:b2484680644a27b8 | 1 | 1000 | delimited | repository/Data/RandomExperiment_1000_WEZ.csv |
| aircombat_wez:8372931050a8036b | 2 | 127 | unstructured_text | repository/LICENSE |
| aircombat_wez:cf7d844fac86530a | 2 | 61 | delimited | repository/Paper_Results/SummaryRMax_Lasso_Ridge_50_Reduction.csv |
| aircombat_wez:f2520c736d775ba3 | 1 | 1 | delimited | repository/Paper_Results/SummaryRNez_RF_50_.csv |
| aircombat_wez:3f2420d4132708e6 | 4 | 88 | delimited | repository/Paper_Results/SummaryWez_Final_Results.csv |
| aircombat_wez:5e10945a47b3304f | 1 | 36 | delimited | repository/Paper_Results/SummaryWez_Ridge_Lasso_PR_1to12.csv |
| aircombat_wez:a726919e93d47d7f | 1 | 0 | support_file | repository/README.md |
| aircombat_wez:591b49bdaee332d0 | 1 | 0 | support_file | repository/WEZ_Model_Generation.ipynb |
| aircombat_wez:a9f1126fd5da30d3 | 1 | 0 | support_file | repository/WEZ_Model_Generation.py |

# 1 record / timestep の意味

Data内の1行は独立した射手(BL)と標的(RD)の初期交戦条件に対する射程計算case。時刻順のflight sampleではない。`Case` はcase番号、`Unnamed: 0` はjoined CSVの保存indexでありtimestampではない。Paper_Resultsの1行はregressor・前処理・学習条件別の誤差/実行時間の集計で、航空機状態ではない。

# trajectory構造

FactorialExperimentの864行はfactorial cases。3個のRandomExperiment_1000_* はそれぞれ1,000行。NEZ/RMAX/WEZは関連する実験表なので独立した3,000 flightと数えない。Caseと7入力の対応検査は後掲。12 CSVすべてにflight timestampや永続agent ID、episode境界はない。

# state候補

input variables=`BL_Speed, RD_Speed, rad, BL_Hdg, RD_Hdg, BL_Alt, RD_Alt`。両機の速度・heading・altitudeと相対幾何の初期条件。raw headerは単位を明記しない。repository READMEは速度NM/hour・角度degree・高度ftと説明するが、factorialのBL_Alt実値は304.8/7620/13716でありREADMEの高度範囲1,000〜45,000 ftとスケールが異なる。raw高度の単位を一意に確認できず、今回勝手に換算していない。連続flightの状態列は存在しない。

# action候補

なし。操舵、throttle、時系列maneuverは記録されていない。

# reward候補

なし。`maxRange`, `RMax`, `RNez` は射程等のtarget/output、`minErrorTry` は計算結果の補助値。RL rewardではない。Paper_ResultsのMAE/RMSEは学習器の誤差指標でありflight rewardではない。

# LLM-Xavierとの互換性

**C**。時系列のstate-action-reward historyは作れない。静的なWEZ回帰taskのデータとして利用できる。

# 不明点

Case順を飛行時刻順に読み替える根拠は存在しない。制御・rewardを復元することはできない。重複やcase対応の実測値は後掲に保存。

## 全CSVのrows・shape・sampling

| filename | bytes | rows | columns | time開始/終了 | median delta (s) |
| --- | --- | --- | --- | --- | --- |
| repository/Data/FactorialExperiment.csv | 49720 | 864 | 10 | (None, None) | None |
| repository/Data/RandomExperiment_1000_NEZ.csv | 52430 | 1000 | 10 | (None, None) | None |
| repository/Data/RandomExperiment_1000_RMAX.csv | 53865 | 1000 | 10 | (None, None) | None |
| repository/Data/RandomExperiment_1000_WEZ.csv | 76481 | 1000 | 13 | (None, None) | None |
| repository/Paper_Results/SummaryRMax_Lasso_Ridge_50_Reduction.csv | 36823 | 60 | 39 | (None, None) | None |
| repository/Paper_Results/SummaryRMax_RF_50_.csv | 1215 | 1 | 39 | (None, None) | None |
| repository/Paper_Results/SummaryRNez_RF_50_.csv | 1218 | 1 | 39 | (None, None) | None |
| repository/Paper_Results/SummaryWez_Final_Results.csv | 7030 | 7 | 53 | (None, None) | None |
| repository/Paper_Results/SummaryWez_Lasso_50_Preprocessing_Comparison.csv | 35689 | 40 | 53 | (None, None) | None |
| repository/Paper_Results/SummaryWez_MLP128_50_Preprocessing_Comparison.csv | 35579 | 40 | 53 | (None, None) | None |
| repository/Paper_Results/SummaryWez_MLP256_50_.csv | 1755 | 1 | 53 | (None, None) | None |
| repository/Paper_Results/SummaryWez_Ridge_Lasso_PR_1to12.csv | 32322 | 36 | 54 | (None, None) | None |

# Schema group別の全column / variable

同じschemaは全columnを一度掲載。少数groupはfilenameも記録し、多数groupの全対応はcandidate_files.csvで検索できる。missingはgroup内合計、uniqueのfile別値はJSON内 `unique`。空文字やNA/NaN等を欠損とし、weatherのみMも欠損扱い。XLSXは全OOXML cellを読み、formulaを実行せずcached typeとtext cellを調査する。

## aircombat_wez:15ed781ce07cec18

1 files。

- `repository/Data/FactorialExperiment.csv`

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| Case | int64 | 0 |
| BL_Speed | int64 | 0 |
| RD_Speed | int64 | 0 |
| rad | int64 | 0 |
| BL_Hdg | int64 | 0 |
| RD_Hdg | int64 | 0 |
| BL_Alt | float64 | 0 |
| RD_Alt | float64 | 0 |
| maxRange | float64 | 0 |
| minErrorTry | float64 | 0 |

## aircombat_wez:3400173d0f20724e

1 files。

- `repository/Data/RandomExperiment_1000_NEZ.csv`

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| Case | int64 | 0 |
| BL_Speed | int64 | 0 |
| RD_Speed | int64 | 0 |
| rad | int64 | 0 |
| BL_Hdg | int64 | 0 |
| RD_Hdg | int64 | 0 |
| BL_Alt | int64 | 0 |
| RD_Alt | int64 | 0 |
| RNez | float64 | 0 |
| minErrorTry | float64 | 0 |

## aircombat_wez:616c9b29c95caed1

1 files。

- `repository/Data/RandomExperiment_1000_RMAX.csv`

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| Case | int64 | 0 |
| BL_Speed | int64 | 0 |
| RD_Speed | int64 | 0 |
| rad | int64 | 0 |
| BL_Hdg | int64 | 0 |
| RD_Hdg | int64 | 0 |
| BL_Alt | int64 | 0 |
| RD_Alt | int64 | 0 |
| RMax | float64 | 0 |
| minErrorTry | float64 | 0 |

## aircombat_wez:b2484680644a27b8

1 files。

- `repository/Data/RandomExperiment_1000_WEZ.csv`

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| Unnamed: 0 | int64 | 0 |
| Case | int64 | 0 |
| BL_Speed | int64 | 0 |
| RD_Speed | int64 | 0 |
| rad | int64 | 0 |
| BL_Hdg | int64 | 0 |
| RD_Hdg | int64 | 0 |
| BL_Alt | int64 | 0 |
| RD_Alt | int64 | 0 |
| maxRange | float64 | 0 |
| minErrorTry | float64 | 0 |
| RMax | float64 | 0 |
| RNez | float64 | 0 |

## aircombat_wez:8372931050a8036b

2 files。

- `repository/LICENSE`
- `repository/requirements.txt`

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |

## aircombat_wez:cf7d844fac86530a

2 files。

- `repository/Paper_Results/SummaryRMax_Lasso_Ridge_50_Reduction.csv`
- `repository/Paper_Results/SummaryRMax_RF_50_.csv`

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| type | str | 0 |
| regressor | str | 0 |
| preprocess_params | str | 0 |
| interaction_degree_mean | float64 | 0 |
| interaction_degree_std | float64 | 0 |
| reduction_factors_mean | float64 | 0 |
| reduction_factors_std | float64 | 0 |
| test_param_mean | float64 | 0 |
| test_param_std | float64 | 0 |
| training_size_mean | float64 | 0 |
| training_size_std | float64 | 0 |
| RMax_MAE_mean | float64 | 0 |
| RMax_MAE_std | float64 | 0 |
| RMax_MAE_bs_ci_low | float64 | 0 |
| RMax_MAE_bs_ci_up | float64 | 0 |
| RMax_RMSE_mean | float64 | 0 |
| RMax_RMSE_std | float64 | 0 |
| RMax_RMSE_bs_ci_low | float64 | 0 |
| RMax_RMSE_bs_ci_up | float64 | 0 |
| RMax_time_err_mean | float64 | 0 |
| RMax_time_err_std | float64 | 0 |
| RMax_relative_err_mean | float64 | 0 |
| RMax_relative_err_std | float64 | 0 |
| RMax_max_error_mean | float64 | 0 |
| RMax_max_error_std | float64 | 0 |
| MAE_mean | float64 | 0 |
| MAE_std | float64 | 0 |
| MAE_bs_ci_low | float64 | 0 |
| MAE_bs_ci_up | float64 | 0 |
| RMSE_mean | float64 | 0 |
| RMSE_std | float64 | 0 |
| RMSE_bs_ci_low | float64 | 0 |
| RMSE_bs_ci_up | float64 | 0 |
| pred_time_mean | float64 | 0 |
| pred_time_std | float64 | 0 |
| eval_reps_mean | float64 | 0 |
| eval_reps_std | float64 | 0 |
| fit_time_mean | float64 | 0 |
| fit_time_std | float64 | 0 |

## aircombat_wez:f2520c736d775ba3

1 files。

- `repository/Paper_Results/SummaryRNez_RF_50_.csv`

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| type | str | 0 |
| regressor | str | 0 |
| preprocess_params | str | 0 |
| interaction_degree_mean | float64 | 0 |
| interaction_degree_std | float64 | 0 |
| reduction_factors_mean | float64 | 0 |
| reduction_factors_std | float64 | 0 |
| test_param_mean | float64 | 0 |
| test_param_std | float64 | 0 |
| training_size_mean | float64 | 0 |
| training_size_std | float64 | 0 |
| RNez_MAE_mean | float64 | 0 |
| RNez_MAE_std | float64 | 0 |
| RNez_MAE_bs_ci_low | float64 | 0 |
| RNez_MAE_bs_ci_up | float64 | 0 |
| RNez_RMSE_mean | float64 | 0 |
| RNez_RMSE_std | float64 | 0 |
| RNez_RMSE_bs_ci_low | float64 | 0 |
| RNez_RMSE_bs_ci_up | float64 | 0 |
| RNez_time_err_mean | float64 | 0 |
| RNez_time_err_std | float64 | 0 |
| RNez_relative_err_mean | float64 | 0 |
| RNez_relative_err_std | float64 | 0 |
| RNez_max_error_mean | float64 | 0 |
| RNez_max_error_std | float64 | 0 |
| MAE_mean | float64 | 0 |
| MAE_std | float64 | 0 |
| MAE_bs_ci_low | float64 | 0 |
| MAE_bs_ci_up | float64 | 0 |
| RMSE_mean | float64 | 0 |
| RMSE_std | float64 | 0 |
| RMSE_bs_ci_low | float64 | 0 |
| RMSE_bs_ci_up | float64 | 0 |
| pred_time_mean | float64 | 0 |
| pred_time_std | float64 | 0 |
| eval_reps_mean | float64 | 0 |
| eval_reps_std | float64 | 0 |
| fit_time_mean | float64 | 0 |
| fit_time_std | float64 | 0 |

## aircombat_wez:3f2420d4132708e6

4 files。

- `repository/Paper_Results/SummaryWez_Final_Results.csv`
- `repository/Paper_Results/SummaryWez_Lasso_50_Preprocessing_Comparison.csv`
- `repository/Paper_Results/SummaryWez_MLP128_50_Preprocessing_Comparison.csv`
- `repository/Paper_Results/SummaryWez_MLP256_50_.csv`

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| type | str | 0 |
| regressor | str | 0 |
| preprocess_params | str | 0 |
| interaction_degree_mean | float64 | 0 |
| interaction_degree_std | float64 | 0 |
| reduction_factors_mean | float64 | 0 |
| reduction_factors_std | float64 | 0 |
| test_param_mean | float64 | 0 |
| test_param_std | float64 | 0 |
| training_size_mean | float64 | 0 |
| training_size_std | float64 | 0 |
| RNez_MAE_mean | float64 | 0 |
| RNez_MAE_std | float64 | 0 |
| RNez_MAE_bs_ci_low | float64 | 0 |
| RNez_MAE_bs_ci_up | float64 | 0 |
| RNez_RMSE_mean | float64 | 0 |
| RNez_RMSE_std | float64 | 0 |
| RNez_RMSE_bs_ci_low | float64 | 0 |
| RNez_RMSE_bs_ci_up | float64 | 0 |
| RNez_time_err_mean | float64 | 0 |
| RNez_time_err_std | float64 | 0 |
| RNez_relative_err_mean | float64 | 0 |
| RNez_relative_err_std | float64 | 0 |
| RNez_max_error_mean | float64 | 0 |
| RNez_max_error_std | float64 | 0 |
| RMax_MAE_mean | float64 | 0 |
| RMax_MAE_std | float64 | 0 |
| RMax_MAE_bs_ci_low | float64 | 0 |
| RMax_MAE_bs_ci_up | float64 | 0 |
| RMax_RMSE_mean | float64 | 0 |
| RMax_RMSE_std | float64 | 0 |
| RMax_RMSE_bs_ci_low | float64 | 0 |
| RMax_RMSE_bs_ci_up | float64 | 0 |
| RMax_time_err_mean | float64 | 0 |
| RMax_time_err_std | float64 | 0 |
| RMax_relative_err_mean | float64 | 0 |
| RMax_relative_err_std | float64 | 0 |
| RMax_max_error_mean | float64 | 0 |
| RMax_max_error_std | float64 | 0 |
| MAE_mean | float64 | 0 |
| MAE_std | float64 | 0 |
| MAE_bs_ci_low | float64 | 0 |
| MAE_bs_ci_up | float64 | 0 |
| RMSE_mean | float64 | 0 |
| RMSE_std | float64 | 0 |
| RMSE_bs_ci_low | float64 | 0 |
| RMSE_bs_ci_up | float64 | 0 |
| pred_time_mean | float64 | 0 |
| pred_time_std | float64 | 0 |
| eval_reps_mean | float64 | 0 |
| eval_reps_std | float64 | 0 |
| fit_time_mean | float64 | 0 |
| fit_time_std | float64 | 0 |

## aircombat_wez:5e10945a47b3304f

1 files。

- `repository/Paper_Results/SummaryWez_Ridge_Lasso_PR_1to12.csv`

| 全column/field名（順序保持） | dtype | missing合計 |
| --- | --- | --- |
| Unnamed: 0 | int64 | 0 |
| type | str | 0 |
| regressor | str | 0 |
| preprocess_params | str | 0 |
| interaction_degree_mean | float64 | 0 |
| interaction_degree_std | float64 | 0 |
| reduction_factors_mean | float64 | 0 |
| reduction_factors_std | float64 | 0 |
| test_param_mean | float64 | 0 |
| test_param_std | float64 | 0 |
| training_size_mean | float64 | 0 |
| training_size_std | float64 | 0 |
| RNez_MAE_mean | float64 | 0 |
| RNez_MAE_std | float64 | 0 |
| RNez_MAE_bs_ci_low | float64 | 0 |
| RNez_MAE_bs_ci_up | float64 | 0 |
| RNez_RMSE_mean | float64 | 0 |
| RNez_RMSE_std | float64 | 0 |
| RNez_RMSE_bs_ci_low | float64 | 0 |
| RNez_RMSE_bs_ci_up | float64 | 0 |
| RNez_time_err_mean | float64 | 0 |
| RNez_time_err_std | float64 | 0 |
| RNez_relative_err_mean | float64 | 0 |
| RNez_relative_err_std | float64 | 0 |
| RNez_max_error_mean | float64 | 0 |
| RNez_max_error_std | float64 | 0 |
| RMax_MAE_mean | float64 | 0 |
| RMax_MAE_std | float64 | 0 |
| RMax_MAE_bs_ci_low | float64 | 0 |
| RMax_MAE_bs_ci_up | float64 | 0 |
| RMax_RMSE_mean | float64 | 0 |
| RMax_RMSE_std | float64 | 0 |
| RMax_RMSE_bs_ci_low | float64 | 0 |
| RMax_RMSE_bs_ci_up | float64 | 0 |
| RMax_time_err_mean | float64 | 0 |
| RMax_time_err_std | float64 | 0 |
| RMax_relative_err_mean | float64 | 0 |
| RMax_relative_err_std | float64 | 0 |
| RMax_max_error_mean | float64 | 0 |
| RMax_max_error_std | float64 | 0 |
| MAE_mean | float64 | 0 |
| MAE_std | float64 | 0 |
| MAE_bs_ci_low | float64 | 0 |
| MAE_bs_ci_up | float64 | 0 |
| RMSE_mean | float64 | 0 |
| RMSE_std | float64 | 0 |
| RMSE_bs_ci_low | float64 | 0 |
| RMSE_bs_ci_up | float64 | 0 |
| pred_time_mean | float64 | 0 |
| pred_time_std | float64 | 0 |
| eval_reps_mean | float64 | 0 |
| eval_reps_std | float64 | 0 |
| fit_time_mean | float64 | 0 |
| fit_time_std | float64 | 0 |

# ファイル間対応の全件照合

`tools/audit_dataset_relationships.py` の実測。sourceコードは実行せず比較のみ。

```json
{
  "factorial_rows": 864,
  "factorial_input_levels": {
    "BL_Speed": [
      450,
      600,
      750
    ],
    "RD_Speed": [
      450,
      600,
      750
    ],
    "rad": [
      -60,
      -30,
      0
    ],
    "BL_Hdg": [
      0
    ],
    "RD_Hdg": [
      -150,
      -120,
      -90,
      -60,
      -30,
      0,
      30,
      60,
      90,
      120,
      150,
      180
    ],
    "BL_Alt": [
      304.8,
      7620.0,
      13716.0
    ],
    "RD_Alt": [
      304.8,
      1828.8,
      6096.0,
      7620.0,
      9144.0,
      12192.0,
      13716.0,
      15240.0
    ]
  },
  "factorial_unique_input_combinations": 864,
  "random_case_correspondence": {
    "RandomExperiment_1000_NEZ": {
      "unique_keys": true,
      "left_rows": 1000,
      "right_rows": 1000,
      "common_keys": 1000,
      "matching_rows": 1000,
      "columns": [
        "BL_Speed",
        "RD_Speed",
        "rad",
        "BL_Hdg",
        "RD_Hdg",
        "BL_Alt",
        "RD_Alt"
      ],
      "columns_in_source": [
        "Case",
        "BL_Speed",
        "RD_Speed",
        "rad",
        "BL_Hdg",
        "RD_Hdg",
        "BL_Alt",
        "RD_Alt",
        "RNez",
        "minErrorTry"
      ],
      "output_correspondence": {
        "unique_keys": true,
        "left_rows": 1000,
        "right_rows": 1000,
        "common_keys": 1000,
        "matching_rows": 1000,
        "columns": [
          "RNez"
        ]
      }
    },
    "RandomExperiment_1000_RMAX": {
      "unique_keys": true,
      "left_rows": 1000,
      "right_rows": 1000,
      "common_keys": 1000,
      "matching_rows": 1000,
      "columns": [
        "BL_Speed",
        "RD_Speed",
        "rad",
        "BL_Hdg",
        "RD_Hdg",
        "BL_Alt",
        "RD_Alt"
      ],
      "columns_in_source": [
        "Case",
        "BL_Speed",
        "RD_Speed",
        "rad",
        "BL_Hdg",
        "RD_Hdg",
        "BL_Alt",
        "RD_Alt",
        "RMax",
        "minErrorTry"
      ],
      "output_correspondence": {
        "unique_keys": true,
        "left_rows": 1000,
        "right_rows": 1000,
        "common_keys": 1000,
        "matching_rows": 1000,
        "columns": [
          "RMax"
        ]
      }
    }
  }
}
```

WEZ/NEZ/RMAXの全1,000 Caseで7入力が一致し、RNez/RMaxのoutputも対応表で全件一致（rtol=1e-9, atol=1e-10）。1,000 random casesの関連exportであり3,000独立caseではない。864 factorialの入力tupleは全件unique。周辺level集合の直積を全組合せ採用したと仮定しない。

