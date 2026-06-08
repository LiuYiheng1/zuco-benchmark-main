@echo off
REM ZuCo 2.0 TGCR Experiment Runner

echo ====================================
echo ZuCo 2.0 Experiment Pipeline
echo ====================================

cd /d %~dp0\src

echo.
echo ====================================
echo Step 1: Running Official Baseline
echo ====================================
python run_official_baseline.py
if %errorlevel% neq 0 (
    echo Error running official baseline
)

echo.
echo ====================================
echo Step 2: Running PyTorch Baselines
echo ====================================
python neuro_web_reader\train_tgcr.py --model baselines --seeds 0 1 2 3 4
if %errorlevel% neq 0 (
    echo Error running baselines
)

echo.
echo ====================================
echo Step 3: Running TGCR v1
echo ====================================
python neuro_web_reader\train_tgcr.py --model tgcr --seeds 0 1 2 3 4
if %errorlevel% neq 0 (
    echo Error running TGCR
)

echo.
echo ====================================
echo Step 4: Running Ablation Experiments
echo ====================================
python neuro_web_reader\train_tgcr.py --model all --seeds 0 1 2 3 4
if %errorlevel% neq 0 (
    echo Error running ablation
)

echo.
echo ====================================
echo Step 5: Generating Reports
echo ====================================
python neuro_web_reader\evaluate.py --results_dir results

echo.
echo ====================================
echo Experiments Complete!
echo ====================================
echo Results saved in: src\results
echo Reports saved in: src\reports
pause