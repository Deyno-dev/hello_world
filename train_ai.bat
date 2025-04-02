@echo off
echo Starting AI Training...

:loop
echo 🔄 Retraining AI models...

:: Run XGBoost training and capture errors
python ai_custom\train_xgboost.py
if %errorlevel% neq 0 (
    echo ❌ XGBoost Training Failed! Check logs for details.
    exit /b 1
) else (
    echo ✅ XGBoost Training Completed Successfully!
)

:: Run Prophet training and capture errors
python ai_custom\train_prophet.py
if %errorlevel% neq 0 (
    echo ❌ Prophet Training Failed! Check logs for details.
    exit /b 1
) else (
    echo ✅ Prophet Training Completed Successfully!
)

echo 🕒 Training cycle completed, waiting before next cycle...
timeout /t 60 /nobreak >nul

goto loop
