import React from "react";
import { motion } from "framer-motion";
import {
  Code,
  Database,
  Globe,
  Brain,
  Zap,
  Shield,
  Activity,
  Github,
  Linkedin,
  Cpu,
} from "lucide-react";

const About = () => {
  return (
    <div className="relative min-h-screen overflow-x-hidden bg-[#060816] text-white">
      {/* Background Effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-8rem] left-[-8rem] h-[28rem] w-[28rem] rounded-full bg-cyan-500/10 blur-[120px]" />
        <div className="absolute bottom-[-8rem] right-[-8rem] h-[28rem] w-[28rem] rounded-full bg-violet-500/10 blur-[120px]" />
        <div className="absolute top-[50%] left-[50%] h-[20rem] w-[20rem] -translate-x-1/2 rounded-full bg-blue-500/5 blur-[100px]" />
      </div>

      <div className="relative z-10 mx-auto max-w-7xl px-6 py-28 lg:px-8 space-y-28">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-cyan-400/20 bg-white/5 px-4 py-2 text-sm text-cyan-300 backdrop-blur-xl">
            <Cpu className="h-4 w-4" />
            The Project
          </div>
          <h1 className="text-5xl md:text-7xl font-black leading-tight tracking-tight">
            About{" "}
            <span className="bg-gradient-to-r from-cyan-400 via-blue-400 to-violet-500 bg-clip-text text-transparent">
              SignAI
            </span>
          </h1>
          <p className="mt-5 text-xl text-slate-400 max-w-xl">
            Advanced Real-time Indian Sign Language Recognition
          </p>
        </motion.div>

        {/* Mission Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-14 items-center">
          {/* Left: Text */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7 }}
            viewport={{ once: true }}
            className="space-y-6"
          >
            <h2 className="text-3xl md:text-4xl font-bold">Mission Overview</h2>
            <p className="text-slate-400 leading-relaxed text-lg">
              SignAI represents a leap forward in accessible communication
              technology. Our mission is to bridge the gap between sign language
              users and the digital world through intelligent, real-time
              transcription.
            </p>
            <p className="text-slate-400 leading-relaxed text-lg">
              Using state-of-the-art{" "}
              <span className="text-cyan-300 font-semibold">Deep Learning</span>{" "}
              and Computer Vision, our system instantly identifies and translates
              Indian Sign Language gestures with high precision, empowering users
              to communicate naturally and effectively.
            </p>
            <p className="text-slate-400 leading-relaxed text-lg">
              The project combines{" "}
              <span className="text-violet-300 font-semibold">
                advanced AI algorithms
              </span>{" "}
              with a modern, intuitive web interface, making cutting-edge
              accessibility tools available to everyone.
            </p>
          </motion.div>

          {/* Right: Feature Cards */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7 }}
            viewport={{ once: true }}
            className="space-y-5"
          >
            {[
              {
                icon: Zap,
                color: "text-cyan-300",
                bg: "from-cyan-500/20 to-blue-500/20",
                title: "Real-time Processing",
                desc: "Instant gesture detection and classification at 60 FPS",
              },
              {
                icon: Activity,
                color: "text-violet-300",
                bg: "from-violet-500/20 to-purple-500/20",
                title: "High Accuracy",
                desc: "Trained on diverse ISL datasets for robust prediction",
              },
              {
                icon: Shield,
                color: "text-emerald-300",
                bg: "from-emerald-500/20 to-teal-500/20",
                title: "Secure & Private",
                desc: "All processing handled locally — your data stays yours",
              },
            ].map((item, idx) => {
              const Icon = item.icon;
              return (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  viewport={{ once: true }}
                  whileHover={{ y: -4 }}
                  className="group flex items-start gap-5 rounded-2xl border border-white/10 bg-white/5 p-6 backdrop-blur-2xl transition-all duration-300 hover:border-cyan-400/20 hover:bg-white/[0.07]"
                >
                  <div
                    className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br ${item.bg}`}
                  >
                    <Icon className={`h-6 w-6 ${item.color}`} />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-1">
                      {item.title}
                    </h3>
                    <p className="text-sm text-slate-400 leading-relaxed">
                      {item.desc}
                    </p>
                  </div>
                </motion.div>
              );
            })}
          </motion.div>
        </div>

        {/* Tech Stack */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          viewport={{ once: true }}
        >
          <div className="mb-14 text-center">
            <h2 className="text-4xl md:text-5xl font-bold">Technology Stack</h2>
            <p className="mt-4 text-slate-400 text-lg">
              Built on a foundation of modern, battle-tested tools
            </p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {[
              {
                name: "React",
                icon: Globe,
                color: "text-cyan-400",
                sub: "Frontend",
              },
              {
                name: "Tailwind CSS",
                icon: Code,
                color: "text-sky-400",
                sub: "Styling",
              },
              {
                name: "Flask",
                icon: Database,
                color: "text-slate-200",
                sub: "Backend",
              },
              {
                name: "TensorFlow",
                icon: Brain,
                color: "text-orange-400",
                sub: "Deep Learning",
              },
            ].map((tech, idx) => {
              const Icon = tech.icon;
              return (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, y: 25 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  viewport={{ once: true }}
                  whileHover={{ y: -6 }}
                  className="group flex flex-col items-center justify-center rounded-3xl border border-white/10 bg-white/5 p-8 backdrop-blur-2xl transition-all duration-300 hover:border-cyan-400/20 hover:bg-white/[0.07]"
                >
                  <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-500/20 to-violet-500/20 group-hover:from-cyan-500/30 group-hover:to-violet-500/30 transition-all duration-300">
                    <Icon className={`h-8 w-8 ${tech.color}`} />
                  </div>
                  <h3 className="font-bold text-lg mb-1">{tech.name}</h3>
                  <p className="text-xs text-slate-500 uppercase tracking-widest">
                    {tech.sub}
                  </p>
                </motion.div>
              );
            })}
          </div>
        </motion.div>

        {/* Team Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          viewport={{ once: true }}
        >
          <div className="mb-14 text-center">
            <h2 className="text-4xl md:text-5xl font-bold">Meet The Team</h2>
            <p className="mt-4 text-slate-400 text-lg">
              The people behind SignAI
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                name: "Shinjan Saha",
                role: "Team Lead",
                linkedin: "https://www.linkedin.com/in/shinjan-saha-1bb744319/",
                github: "https://github.com/Code-r4Life/",
              },
              {
                name: "Satyabrata Das Adhikari",
                role: "Developer",
                linkedin:
                  "https://www.linkedin.com/in/satyabrata-das-adhikari-1813a7324/",
                github: "https://github.com/satya-py/",
              },
              {
                name: "Sayan Sk",
                role: "Developer",
                linkedin: "https://www.linkedin.com/in/sayan-sk-092203318/",
                github: "https://github.com/Sayan474/",
              },
            ].map((member, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.1 }}
                viewport={{ once: true }}
                whileHover={{ y: -8 }}
                className="group relative overflow-hidden rounded-3xl border border-white/10 bg-white/5 p-10 text-center backdrop-blur-2xl transition-all duration-300 hover:border-cyan-400/20 hover:bg-white/[0.07]"
              >
                {/* subtle glow on hover */}
                <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/0 via-transparent to-violet-500/0 opacity-0 transition-opacity duration-500 group-hover:opacity-100 group-hover:from-cyan-500/5 group-hover:to-violet-500/5" />

                <div className="relative z-10">
                  {/* Avatar */}
                  <div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-full border border-cyan-400/20 bg-gradient-to-br from-cyan-500/20 to-violet-500/20 text-3xl font-black text-cyan-300">
                    {member.name[0]}
                  </div>

                  <h3 className="text-xl font-bold text-white mb-1">
                    {member.name}
                  </h3>
                  <p className="text-cyan-400 font-medium mb-6">{member.role}</p>

                  <div className="flex justify-center gap-3">
                    <a
                      href={member.linkedin}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex h-10 w-10 items-center justify-center rounded-xl border border-white/10 bg-white/5 text-slate-400 transition-all hover:border-cyan-400/30 hover:bg-white/10 hover:text-cyan-300"
                    >
                      <Linkedin size={18} />
                    </a>
                    <a
                      href={member.github}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex h-10 w-10 items-center justify-center rounded-xl border border-white/10 bg-white/5 text-slate-400 transition-all hover:border-violet-400/30 hover:bg-white/10 hover:text-violet-300"
                    >
                      <Github size={18} />
                    </a>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

      </div>
    </div>
  );
};

export default About;