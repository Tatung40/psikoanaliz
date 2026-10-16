@echo off
chcp 65001 > nul
title PsikoAnaliz — Klasik Test Kuramı Analiz Platformu
color 1F

echo ===============================================================================
echo                PSİKOANALİZ — KLASİK TEST KURAMI ANALİZ PLATFORMU
echo                            Doç. Dr. Mehmet Taha Eser
echo ===============================================================================
echo.
echo [1/3] Yerel ortam ve sanal çevre kontrol ediliyor...

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
    set "STREAMLIT_EXE=.venv\Scripts\streamlit.exe"
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% equ 0 (
        set "PYTHON_EXE=python"
        set "STREAMLIT_EXE=streamlit"
    ) else (
        echo [HATA] Python veya yerel sanal ortam (.venv) bulunamadı!
        echo Lütfen Python 3.11'in kurulu olduğundan ve gereksinimlerin yüklendiğinden emin olun.
        echo.
        pause
        exit /b 1
    )
)

echo [2/3] Tarayıcı ve Streamlit sunucusu hazırlanıyor...
echo.
echo -------------------------------------------------------------------------------
echo Uygulama açılıyor: http://localhost:8501
echo Çıkış yapmak için bu pencereyi kapatabilir veya Ctrl+C tuşlarına basabilirsiniz.
echo -------------------------------------------------------------------------------
echo.

"%STREAMLIT_EXE%" run app.py --server.headless=false --server.port=8501

if %ERRORLEVEL% neq 0 (
    echo.
    echo [BİLGİ] Sunucu durduruldu.
    pause
)
