import React, { useState } from 'react';
import { Camera, Mic, FileText, Sparkles, Upload, Loader2, Info, AlertTriangle } from 'lucide-react';
import { WebcamCapture } from '../components/WebcamCapture';
import { AudioRecorder } from '../components/AudioRecorder';
import { analyzeMultimodal } from '../services/api';
import { MultimodalAnalysisResponse } from '../types/emotion';

interface AnalysisPageProps {
  onAnalysisComplete: (result: MultimodalAnalysisResponse) => void;
}

export const AnalysisPage: React.FC<AnalysisPageProps> = ({ onAnalysisComplete }) => {
  // Input states
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | File | null>(null);
  const [textInput, setTextInput] = useState<string>('');

  // Mode selection
  const [fusionMethod, setFusionMethod] = useState<string>('attention_fusion');
  const [autoStt, setAutoStt] = useState<boolean>(true);
  const [imageMode, setImageMode] = useState<'upload' | 'webcam'>('upload');
  const [audioMode, setAudioMode] = useState<'upload' | 'record'>('record');

  // Loading & error
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const activeCount = [
    imageFile !== null,
    audioBlob !== null,
    textInput.trim().length > 0,
  ].filter(Boolean).length;

  const handleImageFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setImageFile(e.target.files[0]);
    }
  };

  const handleAudioFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setAudioBlob(e.target.files[0]);
    }
  };

  const handleRunAnalysis = async () => {
    if (activeCount === 0) {
      setErrorMsg('Please provide at least one modality (Image, Audio, or Text).');
      return;
    }

    setErrorMsg(null);
    setIsLoading(true);

    try {
      const result = await analyzeMultimodal(imageFile, audioBlob, textInput, fusionMethod, autoStt);
      setIsLoading(false);
      onAnalysisComplete(result);
    } catch (err: any) {
      setIsLoading(false);
      setErrorMsg(err.message || 'An error occurred during multimodal emotion analysis.');
    }
  };

  return (
    <div className="flex flex-col gap-8 py-4 max-w-6xl mx-auto">
      {/* Top Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-card p-6 border border-slate-800">
        <div>
          <h1 className="text-2xl font-extrabold text-white">Multimodal Emotion Analysis</h1>
          <p className="text-xs text-slate-400 mt-1">
            Provide inputs for one, two, or all three modalities. Missing modalities will be dynamically masked.
          </p>
        </div>

        {/* Modality Status Counter */}
        <div className="flex items-center gap-2 bg-slate-950 border border-slate-800 px-4 py-2 rounded-xl">
          <span className="text-xs text-slate-400">Active Modalities:</span>
          <span className="text-sm font-bold text-indigo-400">{activeCount} / 3</span>
          {activeCount > 0 && (
            <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-2 py-0.5 rounded-full border border-emerald-500/30">
              Ready
            </span>
          )}
        </div>
      </div>

      {errorMsg && (
        <div className="bg-rose-950/60 border border-rose-500/40 text-rose-200 p-4 rounded-xl text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-rose-400 flex-shrink-0" />
          {errorMsg}
        </div>
      )}

      {/* 3 Input Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Card 1: Facial Analysis (Vision) */}
        <div className="glass-card-hover p-6 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-indigo-400 font-bold text-sm">
              <Camera className="w-5 h-5" />
              Facial Analysis
            </div>
            <div className="flex bg-slate-950 p-1 rounded-lg border border-slate-800">
              <button
                onClick={() => setImageMode('upload')}
                className={`text-[10px] px-2 py-1 rounded-md font-medium transition-all ${
                  imageMode === 'upload' ? 'bg-indigo-600 text-white' : 'text-slate-400'
                }`}
              >
                Upload
              </button>
              <button
                onClick={() => setImageMode('webcam')}
                className={`text-[10px] px-2 py-1 rounded-md font-medium transition-all ${
                  imageMode === 'webcam' ? 'bg-indigo-600 text-white' : 'text-slate-400'
                }`}
              >
                Webcam
              </button>
            </div>
          </div>

          {imageMode === 'upload' ? (
            <div className="flex flex-col gap-3">
              <label className="border-2 border-dashed border-slate-800 hover:border-indigo-500/50 bg-slate-900/40 rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer transition-all">
                <Upload className="w-8 h-8 text-slate-500 mb-2" />
                <span className="text-xs text-slate-300 font-medium">Click to upload JPG / PNG</span>
                <span className="text-[10px] text-slate-500">Max file size 10MB</span>
                <input type="file" accept="image/*" onChange={handleImageFileChange} className="hidden" />
              </label>

              {imageFile && (
                <div className="bg-slate-900 border border-indigo-500/30 p-2.5 rounded-xl flex items-center justify-between text-xs">
                  <span className="truncate max-w-[180px] text-indigo-300 font-medium">{imageFile.name}</span>
                  <button onClick={() => setImageFile(null)} className="text-slate-500 hover:text-slate-300">
                    ✕
                  </button>
                </div>
              )}
            </div>
          ) : (
            <WebcamCapture onCapture={(file) => setImageFile(file)} />
          )}
        </div>

        {/* Card 2: Audio Analysis */}
        <div className="glass-card-hover p-6 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-pink-400 font-bold text-sm">
              <Mic className="w-5 h-5" />
              Audio Analysis
            </div>
            <div className="flex bg-slate-950 p-1 rounded-lg border border-slate-800">
              <button
                onClick={() => setAudioMode('record')}
                className={`text-[10px] px-2 py-1 rounded-md font-medium transition-all ${
                  audioMode === 'record' ? 'bg-pink-600 text-white' : 'text-slate-400'
                }`}
              >
                Record
              </button>
              <button
                onClick={() => setAudioMode('upload')}
                className={`text-[10px] px-2 py-1 rounded-md font-medium transition-all ${
                  audioMode === 'upload' ? 'bg-pink-600 text-white' : 'text-slate-400'
                }`}
              >
                Upload
              </button>
            </div>
          </div>

          {audioMode === 'record' ? (
            <AudioRecorder onAudioRecorded={(blob) => setAudioBlob(blob)} />
          ) : (
            <div className="flex flex-col gap-3">
              <label className="border-2 border-dashed border-slate-800 hover:border-pink-500/50 bg-slate-900/40 rounded-xl p-6 flex flex-col items-center justify-center cursor-pointer transition-all">
                <Upload className="w-8 h-8 text-slate-500 mb-2" />
                <span className="text-xs text-slate-300 font-medium">Upload WAV / MP3 Audio</span>
                <span className="text-[10px] text-slate-500">Recorded speech clip</span>
                <input type="file" accept="audio/*" onChange={handleAudioFileChange} className="hidden" />
              </label>

              {audioBlob && (
                <div className="bg-slate-900 border border-pink-500/30 p-2.5 rounded-xl flex items-center justify-between text-xs">
                  <span className="truncate max-w-[180px] text-pink-300 font-medium">
                    {audioBlob instanceof File ? audioBlob.name : 'Recorded Audio.wav'}
                  </span>
                  <button onClick={() => setAudioBlob(null)} className="text-slate-500 hover:text-slate-300">
                    ✕
                  </button>
                </div>
              )}
            </div>
          )}

          <div className="flex items-center gap-2 pt-2 border-t border-slate-800/60">
            <input
              type="checkbox"
              id="autoStt"
              checked={autoStt}
              onChange={(e) => setAutoStt(e.target.checked)}
              className="rounded bg-slate-900 border-slate-700 text-pink-600 focus:ring-pink-500"
            />
            <label htmlFor="autoStt" className="text-xs text-slate-400 cursor-pointer">
              Auto Speech-to-Text (Whisper STT)
            </label>
          </div>
        </div>

        {/* Card 3: Text Analysis */}
        <div className="glass-card-hover p-6 flex flex-col gap-4">
          <div className="flex items-center gap-2 text-teal-400 font-bold text-sm">
            <FileText className="w-5 h-5" />
            Text Analysis
          </div>

          <textarea
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            placeholder="Type your sentence here (e.g. 'I am feeling extremely excited and satisfied today!')..."
            rows={5}
            className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-teal-500/50 resize-none font-sans"
          />

          <div className="flex items-center justify-between text-[11px] text-slate-500">
            <span>Character count: {textInput.length}</span>
            {textInput.length > 0 && (
              <button onClick={() => setTextInput('')} className="hover:text-slate-300">
                Clear Text
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Fusion Architecture Selector & Submit Bar */}
      <div className="glass-card p-6 border border-slate-800 flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="flex flex-col gap-1">
          <span className="text-xs font-semibold text-slate-300">Select Fusion Architecture:</span>
          <div className="flex gap-2">
            {[
              { id: 'attention_fusion', label: 'Attention Fusion (Proposed)' },
              { id: 'feature_fusion', label: 'Feature Fusion (Concat)' },
              { id: 'late_fusion', label: 'Late Fusion (Baseline)' },
            ].map((method) => (
              <button
                key={method.id}
                onClick={() => setFusionMethod(method.id)}
                className={`text-xs px-3.5 py-1.5 rounded-xl border transition-all ${
                  fusionMethod === method.id
                    ? 'bg-indigo-600 text-white border-indigo-500 font-semibold shadow-md'
                    : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
                }`}
              >
                {method.label}
              </button>
            ))}
          </div>
        </div>

        <button
          onClick={handleRunAnalysis}
          disabled={isLoading}
          className="w-full md:w-auto px-8 py-4 bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 hover:from-indigo-500 hover:to-pink-500 text-white font-bold text-base rounded-2xl shadow-xl shadow-indigo-500/25 flex items-center justify-center gap-3 transition-all hover:scale-105 disabled:opacity-50 disabled:hover:scale-100"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Running Neural Networks...
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5" />
              Analyze Emotion
            </>
          )}
        </button>
      </div>
    </div>
  );
};
