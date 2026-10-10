import React, { useRef, useState, useEffect } from 'react';
import { Camera, RefreshCw, CheckCircle2, VideoOff } from 'lucide-react';

interface WebcamCaptureProps {
  onCapture: (file: File | null) => void;
}

export const WebcamCapture: React.FC<WebcamCaptureProps> = ({ onCapture }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const startCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setIsStreaming(true);
      }
    } catch (err: any) {
      setCameraError('Webcam access denied or unavailable. Please use file upload.');
      setIsStreaming(false);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const takeSnapshot = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;

      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], `webcam_${Date.now()}.jpg`, { type: 'image/jpeg' });
            const dataUrl = canvas.toDataURL('image/jpeg');
            setCapturedImage(dataUrl);
            onCapture(file);
            stopCamera();
          }
        }, 'image/jpeg');
      }
    }
  };

  const resetCapture = () => {
    setCapturedImage(null);
    onCapture(null);
    startCamera();
  };

  return (
    <div className="flex flex-col items-center gap-3 w-full">
      <canvas ref={canvasRef} className="hidden" />

      {capturedImage ? (
        <div className="relative w-full aspect-video rounded-xl overflow-hidden border border-slate-700 bg-slate-900 group">
          <img src={capturedImage} alt="Captured Face" className="w-full h-full object-cover" />
          <div className="absolute top-2 right-2 bg-emerald-500/90 text-white text-xs font-semibold px-2.5 py-1 rounded-full flex items-center gap-1 shadow-md">
            <CheckCircle2 className="w-3.5 h-3.5" />
            Snapshot Captured
          </div>
          <button
            onClick={resetCapture}
            className="absolute bottom-3 right-3 bg-slate-900/80 hover:bg-slate-900 text-slate-200 text-xs px-3 py-1.5 rounded-lg border border-slate-700 flex items-center gap-1.5 backdrop-blur-md transition-all"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Retake
          </button>
        </div>
      ) : isStreaming ? (
        <div className="relative w-full aspect-video rounded-xl overflow-hidden border border-indigo-500/50 bg-slate-950">
          <video ref={videoRef} className="w-full h-full object-cover transform -scale-x-100" />
          <button
            onClick={takeSnapshot}
            className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold px-4 py-2 rounded-xl shadow-lg flex items-center gap-2 transition-all hover:scale-105"
          >
            <Camera className="w-4 h-4" />
            Capture Snapshot
          </button>
        </div>
      ) : (
        <div className="w-full aspect-video rounded-xl border border-dashed border-slate-800 bg-slate-900/40 flex flex-col items-center justify-center gap-2 p-4 text-center">
          <Camera className="w-8 h-8 text-slate-600" />
          {cameraError ? (
            <p className="text-xs text-rose-400 font-medium max-w-xs">{cameraError}</p>
          ) : (
            <p className="text-xs text-slate-400">Capture a face image directly from your webcam</p>
          )}
          <button
            onClick={startCamera}
            className="mt-1 px-3.5 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-lg border border-slate-700 flex items-center gap-1.5 transition-all"
          >
            <Camera className="w-3.5 h-3.5 text-indigo-400" />
            Open Webcam
          </button>
        </div>
      )}
    </div>
  );
};
