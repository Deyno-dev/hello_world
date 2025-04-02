@echo off
echo 🚀 Starting XGBoost AI Training...

:loop
echo 🔄 Retraining XGBoost model...

:: Run XGBoost training and capture errors
python ai_custom\train_xgboost.py
if %errorlevel% neq 0 (
    echo ❌ XGBoost Training Failed! Check logs for details.
    exit /b 1
) else (
    echo ✅ XGBoost Training Completed Successfully!
)

echo 🕒 Training cycle completed, waiting before next cycle...
timeout /t 5 /nobreak >nul

goto loop
