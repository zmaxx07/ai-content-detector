import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

export default function Dashboard() {
    const [progress, setProgress] = useState("Idle");
    const [theme, setTheme] = useState("dark");
    const [files, setFiles] = useState([]);

    // WebSocket implementation
    useEffect(() => {
        const wsUrl = process.env.REACT_APP_WS_URL || `ws://${window.location.hostname}:8000/ws`;
        const ws = new WebSocket(wsUrl);
        ws.onmessage = (event) => setProgress(event.data);
        return () => {
            if (ws.readyState === 1) {
                ws.close();
            }
        };
    }, []);

    const toggleTheme = () => setTheme(theme === 'dark' ? 'light' : 'dark');

    const handleDrop = (e) => {
        e.preventDefault();
        setFiles([...e.dataTransfer.files]);
    };

    // Dummy ROC Data
    const rocData = [
        { fpr: 0, tpr: 0 },
        { fpr: 0.1, tpr: 0.8 },
        { fpr: 1, tpr: 1 }
    ];

    return (
        <div className={`min-h-screen ${theme === 'dark' ? 'bg-gray-900 text-white' : 'bg-white text-black'} p-8`}>
            <header className="flex justify-between items-center mb-8">
                <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600">AI Detector Pro</h1>
                <button onClick={toggleTheme} className="p-2 border rounded">Toggle Theme</button>
            </header>
            
            <div className="grid grid-cols-2 gap-8">
                <div 
                    className="border-2 border-dashed border-gray-500 rounded-lg p-12 text-center transition-all hover:border-blue-500"
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={handleDrop}
                >
                    <p>Drag & Drop files here for Batch Analysis</p>
                    <p className="text-sm text-gray-400">{files.length} files queued</p>
                </div>
                
                <div className="bg-gray-800 p-6 rounded-lg shadow-xl">
                    <h2 className="text-xl mb-4">Live Analysis Pipeline</h2>
                    <div className="flex items-center space-x-4">
                        <span className="animate-pulse text-blue-400">{progress}</span>
                    </div>
                </div>
            </div>

            <div className="mt-8 bg-gray-800 p-6 rounded-lg">
                <h2 className="text-xl mb-4">ROC-AUC Curve</h2>
                <LineChart width={400} height={300} data={rocData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="fpr" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="tpr" stroke="#8884d8" />
                </LineChart>
            </div>
        </div>
    );
}
