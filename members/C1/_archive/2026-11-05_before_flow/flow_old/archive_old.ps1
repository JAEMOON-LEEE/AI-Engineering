# 예전 흐름 파일을 _archive 로 옮깁니다 (삭제하지 않음).
# 사용법: 저장소 최상위(AI-Engineering)에서  powershell -ExecutionPolicy Bypass -File members\C1\flow\archive_old.ps1
# final_model\, predict_mlp.py, make_final_split.py, experiment_log_C1.xlsx, figures\fonts 는 옮기지 않습니다.
$ErrorActionPreference = "Stop"
$c1 = "members\C1"; $c2 = "members\C2"
$a1 = "$c1\_archive\2026-11-05_before_flow"; $a2 = "$c2\_archive\2026-11-05_before_flow"
New-Item -ItemType Directory -Force -Path $a1, "$a1\figures", $a2, "$c1\flow\fonts" | Out-Null
# 그림 폰트를 새 흐름 폴더로 복사
Copy-Item "$c1\figures\fonts\*" "$c1\flow\fonts\" -Force -ErrorAction SilentlyContinue
if (Test-Path "$c1\c1_cw_curve.py") { Move-Item "$c1\c1_cw_curve.py" "$a1\" -Force; Write-Host "이동 C1: c1_cw_curve.py" }
if (Test-Path "$c1\c1_flow_common.py") { Move-Item "$c1\c1_flow_common.py" "$a1\" -Force; Write-Host "이동 C1: c1_flow_common.py" }
if (Test-Path "$c1\c1_freeze_model.py") { Move-Item "$c1\c1_freeze_model.py" "$a1\" -Force; Write-Host "이동 C1: c1_freeze_model.py" }
if (Test-Path "$c1\c1_initial_curve.py") { Move-Item "$c1\c1_initial_curve.py" "$a1\" -Force; Write-Host "이동 C1: c1_initial_curve.py" }
if (Test-Path "$c1\c1_lr_sgd.py") { Move-Item "$c1\c1_lr_sgd.py" "$a1\" -Force; Write-Host "이동 C1: c1_lr_sgd.py" }
if (Test-Path "$c1\c1_lr_sgd_summary.py") { Move-Item "$c1\c1_lr_sgd_summary.py" "$a1\" -Force; Write-Host "이동 C1: c1_lr_sgd_summary.py" }
if (Test-Path "$c1\c1_opt_lr.py") { Move-Item "$c1\c1_opt_lr.py" "$a1\" -Force; Write-Host "이동 C1: c1_opt_lr.py" }
if (Test-Path "$c1\c1_opt_lr_cw.py") { Move-Item "$c1\c1_opt_lr_cw.py" "$a1\" -Force; Write-Host "이동 C1: c1_opt_lr_cw.py" }
if (Test-Path "$c1\c1_slide_figures.py") { Move-Item "$c1\c1_slide_figures.py" "$a1\" -Force; Write-Host "이동 C1: c1_slide_figures.py" }
if (Test-Path "$c1\c1_step2_opt_lr.py") { Move-Item "$c1\c1_step2_opt_lr.py" "$a1\" -Force; Write-Host "이동 C1: c1_step2_opt_lr.py" }
if (Test-Path "$c1\c1_step2_print.py") { Move-Item "$c1\c1_step2_print.py" "$a1\" -Force; Write-Host "이동 C1: c1_step2_print.py" }
if (Test-Path "$c1\c1_step3_overfit.py") { Move-Item "$c1\c1_step3_overfit.py" "$a1\" -Force; Write-Host "이동 C1: c1_step3_overfit.py" }
if (Test-Path "$c1\c1_step3_overfit_curve.py") { Move-Item "$c1\c1_step3_overfit_curve.py" "$a1\" -Force; Write-Host "이동 C1: c1_step3_overfit_curve.py" }
if (Test-Path "$c1\c1_step3_pr_curve.py") { Move-Item "$c1\c1_step3_pr_curve.py" "$a1\" -Force; Write-Host "이동 C1: c1_step3_pr_curve.py" }
if (Test-Path "$c1\c1_step4_batch.py") { Move-Item "$c1\c1_step4_batch.py" "$a1\" -Force; Write-Host "이동 C1: c1_step4_batch.py" }
if (Test-Path "$c1\c1_step4_batch_figure.py") { Move-Item "$c1\c1_step4_batch_figure.py" "$a1\" -Force; Write-Host "이동 C1: c1_step4_batch_figure.py" }
if (Test-Path "$c1\c1_step5_structure.py") { Move-Item "$c1\c1_step5_structure.py" "$a1\" -Force; Write-Host "이동 C1: c1_step5_structure.py" }
if (Test-Path "$c1\c1_step6_structure_lr.py") { Move-Item "$c1\c1_step6_structure_lr.py" "$a1\" -Force; Write-Host "이동 C1: c1_step6_structure_lr.py" }
if (Test-Path "$c1\c1_structure_experiment.py") { Move-Item "$c1\c1_structure_experiment.py" "$a1\" -Force; Write-Host "이동 C1: c1_structure_experiment.py" }
if (Test-Path "$c1\c1_ver2_adam.py") { Move-Item "$c1\c1_ver2_adam.py" "$a1\" -Force; Write-Host "이동 C1: c1_ver2_adam.py" }
if (Test-Path "$c1\c1_flow_runs.csv") { Move-Item "$c1\c1_flow_runs.csv" "$a1\" -Force; Write-Host "이동 C1: c1_flow_runs.csv" }
if (Test-Path "$c1\c1_overfit_history.csv") { Move-Item "$c1\c1_overfit_history.csv" "$a1\" -Force; Write-Host "이동 C1: c1_overfit_history.csv" }
if (Test-Path "$c1\ver2_adam_runs.csv") { Move-Item "$c1\ver2_adam_runs.csv" "$a1\" -Force; Write-Host "이동 C1: ver2_adam_runs.csv" }
if (Test-Path "$c1\ver2_adam_steps.csv") { Move-Item "$c1\ver2_adam_steps.csv" "$a1\" -Force; Write-Host "이동 C1: ver2_adam_steps.csv" }
if (Test-Path "$c1\ver2_adam_story.md") { Move-Item "$c1\ver2_adam_story.md" "$a1\" -Force; Write-Host "이동 C1: ver2_adam_story.md" }
if (Test-Path "$c1\experiment_log_C1_backup.xlsx") { Move-Item "$c1\experiment_log_C1_backup.xlsx" "$a1\" -Force; Write-Host "이동 C1: experiment_log_C1_backup.xlsx" }
Get-ChildItem "$c1\figures" -File | ForEach-Object { Move-Item $_.FullName "$a1\figures\" -Force; Write-Host ("이동 C1 그림: " + $_.Name) }
if (Test-Path "$c2\c2_experiments.py") { Move-Item "$c2\c2_experiments.py" "$a2\" -Force; Write-Host "이동 C2: c2_experiments.py" }
if (Test-Path "$c2\c2_final_experiment.py") { Move-Item "$c2\c2_final_experiment.py" "$a2\" -Force; Write-Host "이동 C2: c2_final_experiment.py" }
if (Test-Path "$c2\c2_final_results.csv") { Move-Item "$c2\c2_final_results.csv" "$a2\" -Force; Write-Host "이동 C2: c2_final_results.csv" }
if (Test-Path "$c2\c2_flow_experiment.py") { Move-Item "$c2\c2_flow_experiment.py" "$a2\" -Force; Write-Host "이동 C2: c2_flow_experiment.py" }
if (Test-Path "$c2\c2_flow_print.py") { Move-Item "$c2\c2_flow_print.py" "$a2\" -Force; Write-Host "이동 C2: c2_flow_print.py" }
if (Test-Path "$c2\c2_flow_runs.csv") { Move-Item "$c2\c2_flow_runs.csv" "$a2\" -Force; Write-Host "이동 C2: c2_flow_runs.csv" }
if (Test-Path "$c2\c2_regularization_experiment.py") { Move-Item "$c2\c2_regularization_experiment.py" "$a2\" -Force; Write-Host "이동 C2: c2_regularization_experiment.py" }
if (Test-Path "$c2\c2_regularization_results.csv") { Move-Item "$c2\c2_regularization_results.csv" "$a2\" -Force; Write-Host "이동 C2: c2_regularization_results.csv" }
if (Test-Path "$c2\C2정규화 실험결과.xlsx") { Move-Item "$c2\C2정규화 실험결과.xlsx" "$a2\" -Force; Write-Host "이동 C2: C2정규화 실험결과.xlsx" }
if (Test-Path "$c2\C2최종실험_결과.xlsx") { Move-Item "$c2\C2최종실험_결과.xlsx" "$a2\" -Force; Write-Host "이동 C2: C2최종실험_결과.xlsx" }
Write-Host "완료. 남은 C1 파일:"; Get-ChildItem $c1 -Name
