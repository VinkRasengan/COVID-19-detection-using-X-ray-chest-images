@echo off
echo 🚀 Starting COVID-19 X-Ray Detection App...
echo.

echo 📦 Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo 🔧 Starting Flask Backend (Port 5000)...
start "Flask Backend" cmd /k "python app.py"

echo.
echo ⏳ Waiting for backend to start...
timeout /t 3 /nobreak > nul

echo.
echo ⚛️ Starting React Frontend (Port 3000)...
cd frontend
start "React Frontend" cmd /k "npm start"

echo.
echo ✅ Both servers are starting...
echo.
echo 🌐 Access URLs:
echo    Frontend: http://localhost:3000
echo    Backend:  http://localhost:5000
echo.
echo 📝 Press any key to close this window...
pause > nul 