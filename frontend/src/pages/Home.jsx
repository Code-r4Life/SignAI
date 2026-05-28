import React from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Brain,
  Camera,
  Sparkles,
  Languages,
  ShieldCheck,
  Activity,
  Waves,
} from "lucide-react";

const features = [
  {
    icon: Camera,
    title: "Real-Time Detection",
    description:
      "Capture and translate Indian Sign Language gestures instantly using webcam-based inference.",
  },
  {
    icon: Brain,
    title: "GCN + BiLSTM AI",
    description:
      "Advanced Graph Neural Networks and temporal modeling for intelligent sign understanding.",
  },
  {
    icon: Languages,
    title: "Sentence Generation",
    description:
      "Transform predicted sign words into meaningful human-like sentences using LLMs.",
  },
  {
    icon: ShieldCheck,
    title: "Accessibility First",
    description:
      "Designed to bridge communication gaps for speech and hearing impaired individuals.",
  },
];

const stats = [
  {
    value: "263+",
    label: "Target Sign Classes",
  },
  {
    value: "Real-Time",
    label: "Live Detection",
  },
  {
    value: "AI Powered",
    label: "Sentence Reconstruction",
  },
  {
    value: "MediaPipe",
    label: "Skeletal Tracking",
  },
];

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="relative min-h-screen overflow-x-hidden bg-[#060816] text-white">
      {/* Background Effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10rem] left-[-10rem] h-[30rem] w-[30rem] rounded-full bg-cyan-500/10 blur-[120px]" />
        <div className="absolute bottom-[-10rem] right-[-10rem] h-[30rem] w-[30rem] rounded-full bg-violet-500/10 blur-[120px]" />
        <div className="absolute top-[40%] left-[50%] h-[25rem] w-[25rem] -translate-x-1/2 rounded-full bg-blue-500/5 blur-[100px]" />
      </div>

      {/* Hero Section */}
      <section className="relative z-10 flex min-h-screen items-center">
        <div className="mx-auto grid max-w-7xl grid-cols-1 gap-16 px-6 py-24 lg:grid-cols-2 lg:px-8">
          {/* Left Content */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="flex flex-col justify-center"
          >
            <div className="mb-6 inline-flex w-fit items-center gap-2 rounded-full border border-cyan-400/20 bg-white/5 px-4 py-2 text-sm text-cyan-300 backdrop-blur-xl">
              <Sparkles className="h-4 w-4" />
              AI Powered Accessibility Platform
            </div>

            <h1 className="text-5xl font-black leading-tight tracking-tight md:text-7xl">
              Real-Time
              <span className="bg-gradient-to-r from-cyan-400 via-blue-400 to-violet-500 bg-clip-text text-transparent">
                {" "}
                Indian Sign{" "}
              </span>
              Language Intelligence
            </h1>

            <p className="mt-8 max-w-2xl text-lg leading-relaxed text-slate-400 md:text-xl">
              Empowering communication through cutting-edge AI, Graph Neural
              Networks, and real-time gesture understanding built for Indian
              Sign Language translation.
            </p>

            <div className="mt-10 flex flex-wrap gap-4">
              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                onClick={() => navigate("/detection")}
                className="group flex items-center gap-2 rounded-2xl bg-gradient-to-r from-cyan-500 to-blue-600 px-7 py-4 font-semibold shadow-lg shadow-cyan-500/20 transition-all duration-300 hover:shadow-cyan-500/40"
              >
                Start Detection
                <ArrowRight className="h-5 w-5 transition-transform duration-300 group-hover:translate-x-1" />
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                onClick={() => navigate("/signs")}
                className="rounded-2xl border border-white/10 bg-white/5 px-7 py-4 font-semibold text-slate-200 backdrop-blur-xl transition-all duration-300 hover:border-cyan-400/30 hover:bg-white/10"
              >
                Explore Signs
              </motion.button>
            </div>

            {/* Stats — fixed: text won't overflow, label always below value */}
            <div className="mt-14 grid grid-cols-2 gap-5 md:grid-cols-4">
              {stats.map((stat, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 25 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 + index * 0.1 }}
                  className="flex flex-col justify-between rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur-2xl"
                >
                  <h3 className="text-xl font-bold text-cyan-300 leading-tight break-words">
                    {stat.value}
                  </h3>
                  <p className="mt-2 text-xs text-slate-400 leading-snug">
                    {stat.label}
                  </p>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Right Visual */}
          <motion.div
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.9 }}
            className="relative flex items-center justify-center"
          >
            <div className="relative h-[38rem] w-full max-w-[34rem] overflow-hidden rounded-[2.5rem] border border-white/10 bg-white/5 backdrop-blur-2xl">
              {/* Glow */}
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 via-transparent to-violet-500/10" />

              {/* Decorative */}
              <div className="absolute left-6 top-6 flex gap-2">
                <div className="h-3 w-3 rounded-full bg-red-400" />
                <div className="h-3 w-3 rounded-full bg-yellow-400" />
                <div className="h-3 w-3 rounded-full bg-green-400" />
              </div>

              {/* Main Content */}
              <div className="flex h-full flex-col items-center justify-center p-10 text-center">
                <div className="relative mb-10 flex h-36 w-36 items-center justify-center rounded-full border border-cyan-400/20 bg-cyan-500/10">
                  <div className="absolute h-44 w-44 animate-pulse rounded-full border border-cyan-400/10" />
                  <Activity className="h-16 w-16 text-cyan-300" />
                </div>

                <h2 className="text-3xl font-bold">
                  Gesture Intelligence Engine
                </h2>

                <p className="mt-5 max-w-md text-slate-400">
                  MediaPipe skeletal tracking combined with Graph Neural
                  Networks and BiLSTM temporal modeling for accurate sign
                  recognition.
                </p>

                {/* Floating Cards */}
                <motion.div
                  animate={{ y: [0, -10, 0] }}
                  transition={{
                    duration: 4,
                    repeat: Infinity,
                  }}
                  className="absolute right-6 top-28 rounded-2xl border border-white/10 bg-black/30 p-4 backdrop-blur-xl"
                >
                  <Waves className="mb-2 h-6 w-6 text-cyan-300" />
                  <p className="text-sm text-slate-300">Live Detection</p>
                </motion.div>

                <motion.div
                  animate={{ y: [0, 10, 0] }}
                  transition={{
                    duration: 5,
                    repeat: Infinity,
                  }}
                  className="absolute bottom-24 left-6 rounded-2xl border border-white/10 bg-black/30 p-4 backdrop-blur-xl"
                >
                  <Brain className="mb-2 h-6 w-6 text-violet-300" />
                  <p className="text-sm text-slate-300">AI Reconstruction</p>
                </motion.div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="relative z-10 py-28">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 25 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            viewport={{ once: true }}
            className="mx-auto mb-20 max-w-3xl text-center"
          >
            <h2 className="text-4xl font-bold md:text-5xl">
              Built for Intelligent Accessibility
            </h2>

            <p className="mt-6 text-lg leading-relaxed text-slate-400">
              Combining modern AI architectures with human-centered design to
              create a seamless sign language communication experience.
            </p>
          </motion.div>

          {/* Feature cards — fixed: equal height, description aligned at same vertical level */}
          <div className="grid grid-cols-1 gap-8 md:grid-cols-2 xl:grid-cols-4">
            {features.map((feature, index) => {
              const Icon = feature.icon;

              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  viewport={{ once: true }}
                  whileHover={{ y: -6 }}
                  className="group relative flex flex-col overflow-hidden rounded-3xl border border-white/10 bg-white/5 p-8 backdrop-blur-2xl transition-all duration-300 hover:border-cyan-400/20 hover:bg-white/[0.07]"
                >
                  <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/0 via-transparent to-violet-500/0 opacity-0 transition-opacity duration-500 group-hover:opacity-100" />

                  <div className="relative z-10 flex flex-col h-full">
                    {/* Icon — fixed height so titles start at same level */}
                    <div className="mb-6 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-500/20 to-violet-500/20 shrink-0">
                      <Icon className="h-7 w-7 text-cyan-300" />
                    </div>

                    {/* Title — fixed height (2 lines) so descriptions align */}
                    <h3 className="mb-4 text-2xl font-semibold min-h-[3.5rem] flex items-start">
                      {feature.title}
                    </h3>

                    {/* Description — always at same vertical position */}
                    <p className="leading-relaxed text-slate-400 flex-1">
                      {feature.description}
                    </p>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative z-10 pb-28">
        <div className="mx-auto max-w-5xl px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 25 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7 }}
            viewport={{ once: true }}
            className="relative overflow-hidden rounded-[2.5rem] border border-white/10 bg-white/5 px-10 py-20 text-center backdrop-blur-2xl"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 via-transparent to-violet-500/10" />

            <div className="relative z-10">
              <h2 className="text-4xl font-bold md:text-5xl">
                Experience AI-Powered Sign Translation
              </h2>

              <p className="mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-slate-400">
                Translate Indian Sign Language gestures into meaningful natural
                language sentences in real-time.
              </p>

              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                onClick={() => navigate("/detection")}
                className="mt-10 inline-flex items-center gap-2 rounded-2xl bg-gradient-to-r from-cyan-500 to-blue-600 px-8 py-4 font-semibold shadow-lg shadow-cyan-500/20 transition-all duration-300 hover:shadow-cyan-500/40"
              >
                Launch Detection
                <ArrowRight className="h-5 w-5" />
              </motion.button>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default Home;