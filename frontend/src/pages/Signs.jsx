import React, { useEffect, useMemo, useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { AlertCircle, Filter, Loader2, Search } from "lucide-react";

const Signs = () => {
  const [signs, setSigns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("All");
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchSigns = async () => {
      try {
        const response = await axios.get("http://localhost:5000/api/signs");
        setSigns(response.data.signs || response.data || []);
      } catch (err) {
        setError(err.response?.data?.error || err.message || "Unable to load sign previews");
      } finally {
        setLoading(false);
      }
    };

    fetchSigns();
  }, []);

  const categories = useMemo(() => {
    const values = new Set(signs.map((sign) => sign.category || sign.name?.[0]?.toUpperCase()).filter(Boolean));
    return ["All", ...Array.from(values).sort()];
  }, [signs]);

  const filteredSigns = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();

    return signs.filter((sign) => {
      const matchesSearch = sign.name.toLowerCase().includes(normalizedQuery);
      const signCategory = sign.category || sign.name?.[0]?.toUpperCase();
      const matchesCategory = category === "All" || signCategory === category;
      return matchesSearch && matchesCategory;
    });
  }, [category, query, signs]);

  return (
    <main className="min-h-screen bg-[linear-gradient(135deg,#07111e_0%,#0b1218_48%,#101314_100%)] px-4 pb-16 pt-24 text-white sm:px-6 lg:px-8">
      <section className="mx-auto max-w-7xl">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between"
        >
          <div>
            <p className="mb-3 text-sm uppercase tracking-[0.24em] text-cyan-200/80">Sign Library</p>
            <h1 className="text-4xl font-semibold tracking-normal md:text-6xl">
              Explore trained ISL gestures
            </h1>
            <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300">
              Search the available classes and preview representative videos from the dataset.
            </p>
          </div>

          <div className="flex w-full flex-col gap-3 sm:flex-row lg:w-auto">
            <label className="relative min-w-0 flex-1 lg:w-80">
              <Search className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search signs"
                className="h-12 w-full rounded-full border border-white/10 bg-white/10 pl-11 pr-4 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-cyan-300/50"
              />
            </label>

            <label className="relative">
              <Filter className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
              <select
                value={category}
                onChange={(event) => setCategory(event.target.value)}
                className="h-12 rounded-full border border-white/10 bg-white/10 pl-11 pr-10 text-sm text-white outline-none transition focus:border-cyan-300/50"
              >
                {categories.map((item) => (
                  <option key={item} value={item} className="bg-slate-950">
                    {item}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </motion.div>

        {error && (
          <div className="mb-6 flex items-start gap-3 rounded-3xl border border-amber-300/20 bg-amber-300/10 p-4 text-sm text-amber-100">
            <AlertCircle size={18} className="mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {loading ? (
          <div className="flex h-72 items-center justify-center">
            <Loader2 className="h-10 w-10 animate-spin text-cyan-200" />
          </div>
        ) : filteredSigns.length === 0 ? (
          <div className="rounded-[2rem] border border-white/10 bg-white/[0.06] p-12 text-center text-slate-300">
            No signs match your current search.
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-5 pb-10 sm:grid-cols-2 xl:grid-cols-3">
            {filteredSigns.map((sign, index) => (
              <motion.article
                key={`${sign.name}-${sign.video_url}`}
                initial={{ opacity: 0, y: 18 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: Math.min(index * 0.035, 0.4) }}
                className="overflow-hidden rounded-[1.5rem] border border-white/10 bg-white/[0.06] shadow-xl shadow-black/20 backdrop-blur-xl transition hover:-translate-y-1 hover:border-cyan-300/30"
              >
                <div className="aspect-video bg-black">
                  <video
                    src={sign.video_url}
                    className="h-full w-full object-cover"
                    controls
                    muted
                    preload="metadata"
                  />
                </div>
                <div className="flex items-center justify-between gap-4 p-5">
                  <h2 className="text-lg font-semibold text-white">{sign.name}</h2>
                  <span className="shrink-0 rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-sm text-cyan-100">
                    {sign.category || sign.name?.[0]?.toUpperCase()}
                  </span>
                </div>
              </motion.article>
            ))}
          </div>
        )}
      </section>
    </main>
  );
};

export default Signs;
