import React from 'react';
import { Activity, BrainCircuit, BarChart3, FileText, Home, Sparkles } from 'lucide-react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab }) => {
  const navItems = [
    { id: 'home', label: 'Home', icon: Home },
    { id: 'analyze', label: 'Emotion Analysis', icon: BrainCircuit },
    { id: 'dashboard', label: 'Analytics', icon: BarChart3 },
    { id: 'evaluation', label: 'Model Benchmarks', icon: Activity },
    { id: 'report', label: 'Academic Report', icon: FileText },
  ];

  return (
    <header className="sticky top-0 z-50 bg-slate-950/80 backdrop-blur-md border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Title */}
          <div 
            onClick={() => setActiveTab('home')}
            className="flex items-center gap-3 cursor-pointer group"
          >
            <div className="p-2 bg-indigo-600/20 border border-indigo-500/40 rounded-xl group-hover:bg-indigo-600/30 transition-all">
              <BrainCircuit className="w-6 h-6 text-indigo-400" />
            </div>
            <div>
              <span className="text-lg font-bold bg-gradient-to-r from-indigo-400 via-purple-300 to-pink-400 bg-clip-text text-transparent">
                AURA-Emotion AI
              </span>
              <span className="block text-[10px] uppercase tracking-wider font-semibold text-slate-400">
                Multimodal Deep Learning
              </span>
            </div>
          </div>

          {/* Navigation Items */}
          <nav className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center gap-2 px-3.5 py-2 rounded-xl text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/30 shadow-lg shadow-indigo-500/10'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-indigo-400' : 'text-slate-500'}`} />
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Quick CTA */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => setActiveTab('analyze')}
              className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-medium text-sm rounded-xl shadow-lg shadow-indigo-500/25 transition-all hover:scale-105"
            >
              <Sparkles className="w-4 h-4" />
              Start Analysis
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
