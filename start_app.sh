#!/bin/bash

echo "🚀 Starting COVID-19 X-Ray Detection App..."
echo

echo "📦 Activating virtual environment..."
source venv/bin/activate

echo
echo "🔧 Starting Flask Backend (Port 5000)..."
python app.py &
BACKEND_PID=$!

echo
echo "⏳ Waiting for backend to start..."
sleep 3

echo
echo "⚛️ Starting React Frontend (Port 3000)..."
cd frontend
npm start &
FRONTEND_PID=$!

echo
echo "✅ Both servers are starting..."
echo
echo "🌐 Access URLs:"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:5000"
echo
echo "📝 Press Ctrl+C to stop both servers..."

# Wait for user to stop
trap "echo '🛑 Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait 