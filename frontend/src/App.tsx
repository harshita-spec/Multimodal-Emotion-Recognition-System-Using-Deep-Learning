import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { HomePage } from './pages/HomePage';
import { AnalysisPage } from './pages/AnalysisPage';
import { ResultsPage } from './pages/ResultsPage';
import { AnalyticsDashboard } from './pages/AnalyticsDashboard';
import { EvaluationPage } from './pages/EvaluationPage';
import { AcademicReportPage } from './pages/AcademicReportPage';
import { MultimodalAnalysisResponse } from './types/emotion';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('home');
  const [currentResult, setCurrentResult] = useState<MultimodalAnalysisResponse | null>(null);

  const handleAnalysisComplete = (result: MultimodalAnalysisResponse) => {
    setCurrentResult(result);
    setActiveTab('results');
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'home' && <HomePage setActiveTab={setActiveTab} />}

        {activeTab === 'analyze' && (
          <AnalysisPage onAnalysisComplete={handleAnalysisComplete} />
        )}

        {activeTab === 'results' && currentResult && (
          <ResultsPage result={currentResult} onBack={() => setActiveTab('analyze')} />
        )}

        {activeTab === 'dashboard' && <AnalyticsDashboard />}

        {activeTab === 'evaluation' && <EvaluationPage />}

        {activeTab === 'report' && <AcademicReportPage />}
      </main>

      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-500">
        Multimodal Emotion Recognition System Using Deep Learning &copy; 2026 Academic Submission
      </footer>
    </div>
  );
}

export default App;
