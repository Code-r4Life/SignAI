import React, { useCallback, useEffect, useRef, useState } from "react";
import axios from "axios";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertCircle,
  Brain,
  Loader2,
  Play,
  RotateCcw,
  Send,
  Sparkles,
  Square,
  Video,
  Volume2,
} from "lucide-react";
import clsx from "clsx";

const API_BASE = "http://localhost:5000";
const CONFIDENCE_THRESHOLD = 0.55;
const DUPLICATE_WINDOW_MS = 3500;
const RECORDING_WINDOW_MS = 2200;
const LOOP_PAUSE_MS = 300;

const Detection = () => {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const timeoutRef = useRef(null);
  const abortRef = useRef(null);
  const recorderRef = useRef(null);
  const sessionRef = useRef(0);
  const detectingRef = useRef(false);
  const lastAcceptedRef = useRef({ word: "", at: 0, confidence: 0 });

  const [isDetecting, setIsDetecting] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [health, setHealth] = useState({ status: "checking", model_loaded: false });
  const [result, setResult] = useState(null);
  const [words, setWords] = useState([]);
  const [sentence, setSentence] = useState("");
  const [error, setError] = useState("");
  const [statusText, setStatusText] = useState("Ready");

  const clearTimersAndRequests = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }

    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }

    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    recorderRef.current = null;
  }, []);

  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const stopDetection = useCallback(() => {
    detectingRef.current = false;
    sessionRef.current += 1;
    setIsDetecting(false);
    setIsPredicting(false);
    setStatusText("Stopped. Sequence frozen.");
    clearTimersAndRequests();
    stopStream();
  }, [clearTimersAndRequests, stopStream]);

  useEffect(() => {
    let mounted = true;

    axios
      .get(`${API_BASE}/health`)
      .then((response) => {
        if (mounted) {
          setHealth(response.data);
        }
      })
      .catch(() => {
        if (mounted) {
          setHealth({ status: "offline", model_loaded: false });
        }
      });

    return () => {
      mounted = false;
      stopDetection();
    };
  }, [stopDetection]);

  const appendPrediction = useCallback((prediction, confidence) => {
    if (!prediction || ["No Motion", "Uncertain"].includes(prediction)) {
      return;
    }

    if (confidence < CONFIDENCE_THRESHOLD) {
      return;
    }

    const now = Date.now();
    const last = lastAcceptedRef.current;
    const confidenceDelta = Math.abs(confidence - last.confidence);
    const shouldAppend =
      prediction !== last.word ||
      now - last.at > DUPLICATE_WINDOW_MS ||
      confidenceDelta > 0.2;

    if (!shouldAppend) {
      return;
    }

    lastAcceptedRef.current = { word: prediction, at: now, confidence };
    setWords((prev) => [...prev, prediction]);
  }, []);

  const recordChunk = useCallback(() => {
    return new Promise((resolve, reject) => {
      const stream = streamRef.current;
      if (!stream) {
        reject(new Error("Camera stream is not available"));
        return;
      }

      const chunks = [];
      const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp8")
        ? "video/webm;codecs=vp8"
        : "video/webm";
      const recorder = new MediaRecorder(stream, { mimeType });
      recorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      recorder.onerror = () => reject(new Error("Video recording failed"));
      recorder.onstop = () => {
        recorderRef.current = null;
        resolve(new Blob(chunks, { type: "video/webm" }));
      };

      recorder.start();
      setTimeout(() => {
        if (recorder.state !== "inactive") {
          recorder.stop();
        }
      }, RECORDING_WINDOW_MS);
    });
  }, []);

  const sendChunk = useCallback(
    async (blob, sessionId) => {
      const formData = new FormData();
      formData.append("video", blob, "live.webm");

      abortRef.current = new AbortController();
      setIsPredicting(true);

      try {
        const response = await axios.post(`${API_BASE}/predict`, formData, {
          headers: { "Content-Type": "multipart/form-data" },
          signal: abortRef.current.signal,
        });

        if (!detectingRef.current || sessionRef.current !== sessionId) {
          return;
        }

        setResult(response.data);
        appendPrediction(response.data.prediction, response.data.confidence || 0);
        setStatusText("Listening for signs");
      } catch (err) {
        if (axios.isCancel(err) || err.name === "CanceledError") {
          return;
        }

        if (detectingRef.current) {
          const message = err.response?.data?.error || err.message || "Prediction request failed";
          setError(message);
          setStatusText("Prediction paused");
        }
      } finally {
        if (sessionRef.current === sessionId) {
          setIsPredicting(false);
        }
      }
    },
    [appendPrediction]
  );

  const predictionLoop = useCallback(async () => {
    const sessionId = sessionRef.current;

    if (!detectingRef.current) {
      return;
    }

    try {
      const blob = await recordChunk();
      if (!detectingRef.current || sessionRef.current !== sessionId) {
        return;
      }
      await sendChunk(blob, sessionId);
    } catch (err) {
      if (detectingRef.current) {
        setError(err.message || "Could not record webcam chunk");
      }
    }

    if (detectingRef.current && sessionRef.current === sessionId) {
      timeoutRef.current = setTimeout(predictionLoop, LOOP_PAUSE_MS);
    }
  }, [recordChunk, sendChunk]);

  const startDetection = async () => {
    if (detectingRef.current) {
      return;
    }

    clearTimersAndRequests();
    stopStream();
    setError("");
    setSentence("");
    setStatusText("Starting camera");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      sessionRef.current += 1;
      detectingRef.current = true;
      setIsDetecting(true);
      setStatusText("Listening for signs");
      predictionLoop();
    } catch (err) {
      setStatusText("Camera blocked");
      setError(err.message || "Camera permission is required to start detection");
      stopStream();
    }
  };

  const clearPrediction = () => {
    setWords([]);
    setSentence("");
    setResult(null);
    setError("");
    lastAcceptedRef.current = { word: "", at: 0, confidence: 0 };
    setStatusText(isDetecting ? "Listening for signs" : "Ready");
  };

  const generateSentence = async () => {
    if (words.length === 0) {
      setError("Detect at least one sign before generating a sentence.");
      return;
    }

    setIsGenerating(true);
    setError("");

    try {
      const response = await axios.post(`${API_BASE}/api/generate-sentence`, { words });
      setSentence(response.data.sentence || "");
      if (response.data.warning) {
        setError(response.data.warning);
      }
    } catch (err) {
      setError(err.response?.data?.error || err.message || "Sentence generation failed");
    } finally {
      setIsGenerating(false);
    }
  };

  const speakText = () => {
    const text = sentence || words.join(" ");
    if (!text) {
      return;
    }
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  };

  const confidence = result?.confidence ? Math.round(result.confidence * 100) : 0;

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,#12315f_0%,#07111e_34%,#080b10_72%)] px-4 pb-16 pt-24 text-white sm:px-6 lg:px-8">
      <section className="mx-auto max-w-7xl">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between"
        >
          <div>
            <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-300/10 px-4 py-2 text-sm text-cyan-100">
              <Brain size={16} />
              MediaPipe Holistic + GCN BiLSTM
            </div>
            <h1 className="max-w-4xl text-4xl font-semibold tracking-normal text-white md:text-6xl">
              Real-time Indian Sign Language transcription
            </h1>
            <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300">
              Capture signs, accumulate detected words, then reconstruct a natural sentence with Gemini.
            </p>
          </div>

          <div className="flex flex-wrap gap-3 text-sm">
            <span
              className={clsx(
                "inline-flex items-center gap-2 rounded-full border px-4 py-2",
                health.model_loaded
                  ? "border-emerald-300/30 bg-emerald-300/10 text-emerald-100"
                  : "border-amber-300/30 bg-amber-300/10 text-amber-100"
              )}
            >
              <span className="h-2 w-2 rounded-full bg-current" />
              Backend {health.model_loaded ? "ready" : health.status === "offline" ? "offline" : "loading"}
            </span>
            <span className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/10 px-4 py-2 text-slate-200">
              {statusText}
            </span>
          </div>
        </motion.div>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.45fr)_minmax(340px,0.75fr)]">
          <section className="overflow-hidden rounded-[2rem] border border-white/10 bg-white/[0.06] shadow-2xl shadow-black/30 backdrop-blur-xl">
            <div className="relative aspect-video bg-black">
              <video
                ref={videoRef}
                muted
                playsInline
                className="h-full w-full object-cover"
              />

              {!isDetecting && !streamRef.current && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-950/70 text-center">
                  <div className="mb-5 rounded-full border border-white/10 bg-white/10 p-5">
                    <Video size={40} className="text-cyan-200" />
                  </div>
                  <p className="text-lg font-medium text-white">Camera preview appears here</p>
                  <p className="mt-2 max-w-sm text-sm text-slate-400">
                    Start detection to begin a reusable live session.
                  </p>
                </div>
              )}

              <div className="absolute left-4 top-4 flex items-center gap-2 rounded-full border border-white/10 bg-black/50 px-3 py-2 text-sm backdrop-blur">
                <span
                  className={clsx(
                    "h-2.5 w-2.5 rounded-full",
                    isDetecting ? "bg-emerald-400 shadow-[0_0_14px_#34d399]" : "bg-slate-500"
                  )}
                />
                {isDetecting ? "Live" : "Idle"}
              </div>

              <AnimatePresence>
                {isPredicting && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute bottom-4 right-4 inline-flex items-center gap-2 rounded-full bg-black/60 px-4 py-2 text-sm text-cyan-100 backdrop-blur"
                  >
                    <Loader2 size={16} className="animate-spin" />
                    Predicting
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            <div className="flex flex-wrap items-center gap-3 border-t border-white/10 p-4">
              {!isDetecting ? (
                <button
                  onClick={startDetection}
                  className="inline-flex items-center gap-2 rounded-full bg-cyan-400 px-5 py-3 font-semibold text-slate-950 shadow-lg shadow-cyan-500/20 transition hover:bg-cyan-300"
                >
                  <Play size={18} />
                  Start Detection
                </button>
              ) : (
                <button
                  onClick={stopDetection}
                  className="inline-flex items-center gap-2 rounded-full bg-rose-500 px-5 py-3 font-semibold text-white shadow-lg shadow-rose-500/20 transition hover:bg-rose-400"
                >
                  <Square size={18} />
                  Stop Detection
                </button>
              )}

              <button
                onClick={clearPrediction}
                className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/10 px-5 py-3 font-semibold text-white transition hover:bg-white/15"
              >
                <RotateCcw size={18} />
                Clear Prediction
              </button>

              <button
                onClick={generateSentence}
                disabled={words.length === 0 || isGenerating}
                className="inline-flex items-center gap-2 rounded-full bg-violet-400 px-5 py-3 font-semibold text-slate-950 shadow-lg shadow-violet-500/20 transition hover:bg-violet-300 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {isGenerating ? <Loader2 size={18} className="animate-spin" /> : <Sparkles size={18} />}
                Generate Sentence
              </button>

              <button
                onClick={speakText}
                disabled={words.length === 0 && !sentence}
                className="ml-auto inline-flex h-12 w-12 items-center justify-center rounded-full border border-white/10 bg-white/10 text-white transition hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-40"
                title="Speak sentence"
              >
                <Volume2 size={18} />
              </button>
            </div>
          </section>

          <aside className="space-y-6">
            <section className="rounded-[2rem] border border-white/10 bg-white/[0.06] p-6 shadow-2xl shadow-black/20 backdrop-blur-xl">
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <p className="text-sm uppercase tracking-[0.22em] text-slate-400">Live Prediction</p>
                  <h2 className="mt-2 text-3xl font-semibold text-white">
                    {result?.prediction || "Waiting"}
                  </h2>
                </div>
                <div className="flex h-16 w-16 items-center justify-center rounded-full border border-cyan-300/20 bg-cyan-300/10 text-lg font-bold text-cyan-100">
                  {confidence}%
                </div>
              </div>

              <div className="h-2 overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-cyan-300 to-violet-300 transition-all duration-500"
                  style={{ width: `${confidence}%` }}
                />
              </div>

              <div className="mt-6 space-y-3">
                {(result?.all_predictions || []).map((item) => (
                  <div key={item.sign} className="flex items-center justify-between rounded-2xl bg-white/7 px-4 py-3 text-sm">
                    <span className="text-slate-200">{item.sign}</span>
                    <span className="font-medium text-slate-400">{Math.round(item.score * 100)}%</span>
                  </div>
                ))}
              </div>
            </section>

            <section className="rounded-[2rem] border border-white/10 bg-white/[0.06] p-6 shadow-2xl shadow-black/20 backdrop-blur-xl">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-xl font-semibold text-white">Detected Words</h2>
                <span className="rounded-full bg-white/10 px-3 py-1 text-sm text-slate-300">{words.length}</span>
              </div>

              <div className="flex min-h-24 flex-wrap content-start gap-2 rounded-3xl border border-white/10 bg-black/20 p-4">
                {words.length > 0 ? (
                  words.map((word, index) => (
                    <motion.span
                      key={`${word}-${index}`}
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-2 text-sm font-medium text-cyan-50"
                    >
                      {word}
                    </motion.span>
                  ))
                ) : (
                  <span className="text-sm text-slate-500">Words will accumulate here.</span>
                )}
              </div>
            </section>
          </aside>
        </div>

        <section className="mt-6 rounded-[2rem] border border-white/10 bg-white/[0.06] p-6 shadow-2xl shadow-black/20 backdrop-blur-xl">
          <div className="mb-4 flex items-center gap-2">
            <Send size={18} className="text-violet-200" />
            <h2 className="text-xl font-semibold text-white">Generated Sentence</h2>
          </div>
          <div className="min-h-24 rounded-3xl border border-white/10 bg-black/20 p-5 text-lg leading-8 text-slate-100">
            {sentence || "Generate a concise sentence after signs are detected."}
          </div>
        </section>

        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              className="mt-6 flex items-start gap-3 rounded-3xl border border-amber-300/20 bg-amber-300/10 p-4 text-sm text-amber-100"
            >
              <AlertCircle size={18} className="mt-0.5 shrink-0" />
              <span>{error}</span>
            </motion.div>
          )}
        </AnimatePresence>
      </section>
    </main>
  );
};

export default Detection;
