import React, { useState, useRef } from 'react';
import { Mic, Square, RefreshCw, CheckCircle2, Volume2 } from 'lucide-react';

interface AudioRecorderProps {
  onAudioRecorded: (blob: Blob | null) => void;
}

export const AudioRecorder: React.FC<AudioRecorderProps> = ({ onAudioRecorded }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<any>(null);

  const startRecording = async () => {
    setRecordedBlob(null);
    setAudioUrl(null);
    audioChunksRef.current = [];
    setRecordingTime(0);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);
        setRecordedBlob(blob);
        setAudioUrl(url);
        onAudioRecorded(blob);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);

      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      alert('Microphone access permission denied or unavailable.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) clearInterval(timerRef.current);
    }
  };

  const resetRecording = () => {
    setRecordedBlob(null);
    setAudioUrl(null);
    onAudioRecorded(null);
    setRecordingTime(0);
  };

  return (
    <div className="flex flex-col items-center gap-3 w-full">
      {audioUrl ? (
        <div className="w-full bg-slate-900 border border-slate-700 rounded-xl p-3 flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-emerald-400 flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              Audio Recorded ({recordingTime}s)
            </span>
            <button
              onClick={resetRecording}
              className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1"
            >
              <RefreshCw className="w-3 h-3" />
              Re-record
            </button>
          </div>
          <audio src={audioUrl} controls className="w-full h-8" />
        </div>
      ) : isRecording ? (
        <div className="w-full bg-rose-950/40 border border-rose-500/40 rounded-xl p-4 flex flex-col items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-rose-500 animate-ping" />
            <span className="text-xs font-semibold text-rose-300">Recording Voice ({recordingTime}s)...</span>
          </div>

          {/* Animated Waveform Bars */}
          <div className="flex items-center gap-1.5 h-8">
            {[40, 70, 30, 90, 60, 100, 50, 80, 30, 70].map((h, i) => (
              <div
                key={i}
                className="w-1.5 bg-rose-500 rounded-full animate-pulse"
                style={{ height: `${h}%`, animationDelay: `${i * 100}ms` }}
              />
            ))}
          </div>

          <button
            onClick={stopRecording}
            className="px-4 py-1.5 bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold rounded-lg flex items-center gap-1.5 shadow-lg"
          >
            <Square className="w-3.5 h-3.5" />
            Stop Recording
          </button>
        </div>
      ) : (
        <div className="w-full bg-slate-900/40 border border-dashed border-slate-800 rounded-xl p-4 flex flex-col items-center gap-2 text-center">
          <Mic className="w-7 h-7 text-slate-500" />
          <p className="text-xs text-slate-400">Record spoken audio using your microphone</p>
          <button
            onClick={startRecording}
            className="px-3.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-lg border border-slate-700 flex items-center gap-1.5 transition-all"
          >
            <Mic className="w-3.5 h-3.5 text-rose-400" />
            Record Audio
          </button>
        </div>
      )}
    </div>
  );
};
