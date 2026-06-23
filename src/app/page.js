"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { DotsRing } from "@/components/loading-ui/dots-ring";
import { DIST_STYLES, SAMPLE_TEXT } from "@/app/constants/distStyles";
import HighlightedText from "@/components/HighlightedText";
import Tooltip from "@/components/Tooltip";
import StepperCard from "@/components/StepperCard";
import { FaMoon } from "react-icons/fa";
import { IoSunnyOutline } from "react-icons/io5";

export default function Home() {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [text, setText] = useState("");
  const [mode, setMode] = useState("write");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [hoveredDistortion, setHoveredDistortion] = useState(null);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState({});
  const [error, setError] = useState("");
  const [activeStep, setActiveStep] = useState(0);
  const [stepDirection, setStepDirection] = useState("next");
  const [rewriteLoading, setRewriteLoading] = useState(false);
  const [rewritten, setRewritten] = useState(null);
  const [rewriteError, setRewriteError] = useState(null);
  const [backendReady, setBackendReady] = useState(false);

  const analyzeText = async () => {
    setResult(null);
    setHoveredDistortion(null);
    setFeedbackSubmitted({});
    setError("");
    if (text.trim() === "") {
      setError("Please enter some text first.");
      return;
    }
    setLoading(true);
    try {
      const res = await fetch("api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (res.ok) {
        const data = await res.json();
        const normalized = data.results.map((r, i) => ({
          id: i + 1,
          originalIdx: i,
          text: r.input,
          dist: r.prediction,
          conf: r.confidence,
        }));
        setResult(normalized);
        setMode("review");
        setActiveStep(0);
      } else {
        setError(
          `Status: ${res.status}. Something went wrong. Please try again.`,
        );
      }
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  };

  const sendFeedback = async (idx, isAccepted, correction = null) => {
    const r = result[idx];
    try {
      await fetch("api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: r.text,
          predicted_distortion: r.dist,
          user_correction: correction,
          is_accepted: isAccepted,
          confidence: r.conf,
        }),
      });
      setFeedbackSubmitted((prev) => ({ ...prev, [idx]: true }));
    } catch (e) {
      console.error(`Feedback error: ${e.message}`);
    }
  };

  const rewrite = async () => {
    setRewriteError(null);
    setRewriteLoading(true);
    try {
      const res = await fetch("api/rewrite", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          distortions: result.map((r) => ({
            input: r.text,
            prediction: r.dist,
            confidence: r.conf,
          })),
        }),
      });
      const data = await res.json();
      setRewritten(data["rewritten"]);
    } catch (e) {
      setRewriteError("Failed to create rewrite. Please try again.");
    }
    setRewriteLoading(false);
  };

  const backToEdit = () => {
    setMode("write");
    setResult(null);
    setRewritten(null);
  };

  const distortedResults = result
    ? result.filter((r) => r.dist !== "No Distortion" && DIST_STYLES[r.dist])
    : [];

  useEffect(() => {
    fetch("api/health").then(() => setBackendReady(true));
  }, []);

  return (
    <>
      <nav className="sticky top-0 z-50 flex items-center justify-between px-6 py-3.5 border-b border-border bg-background/80 backdrop-blur-lg">
        <a
          href="/"
          className="text-[22px] text-foreground no-underline"
          style={{ fontFamily: "var(--font-instrument-serif)" }}
        >
          Reframe
          <span className="inline-block text-primary pl-2 mt-1 text-xl">
            {"\u2737"}
          </span>
        </a>
        <Button
          variant="outline"
          size="lg"
          className="px-4"
          onClick={() => {
            document.documentElement.classList.toggle("dark");
            setIsDarkMode(!isDarkMode);
          }}
        >
          {isDarkMode ? (
            <>
              {" "}
              <IoSunnyOutline className="size-4" /> Light
            </>
          ) : (
            <>
              <FaMoon className="size-3.5" /> Dark
            </>
          )}
        </Button>
      </nav>

      <main className="px-6 py-8 max-w-2xl mx-auto">
        <h1
          className="text-4xl font-normal text-foreground mb-2"
          style={{
            fontFamily: "var(--font-instrument-serif)",
            animation: "revealUp 0.5s cubic-bezier(0.23,1,0.32,1) 0.1s both",
          }}
        >
          What's on your mind?
        </h1>
        <p
          className="text-[15px] text-muted-foreground mb-6"
          style={{
            animation: "revealUp 0.5s cubic-bezier(0.23,1,0.32,1) 0.2s both",
          }}
        >
          Write freely. The tool will help you spot patterns in your thinking.
        </p>

        {mode === "write" && (
          <textarea
            aria-label="Journal entry text input"
            value={text}
            placeholder="Write your journal entry here ..."
            onChange={(e) => setText(e.target.value)}
            rows={4}
            className="w-full p-4 rounded-xl border border-border bg-card text-foreground text-[15px] leading-relaxed resize-y outline-none focus:ring-2 focus:ring-ring/50 box-border"
            style={{ fontFamily: "var(--font-figtree)" }}
          />
        )}

        <div className="flex items-center justify-between mt-3">
          <span
            className={`text-[13px] tabular-nums ${text.length > 4500 ? "text-red-500" : "text-muted-foreground"}`}
          >
            {mode === "write" && `${text.length.toLocaleString()} / 5,000`}
          </span>
          <div className="flex gap-2">
            {mode === "write" && text && (
              <Button
                variant="outline"
                className="py-5 px-4"
                onClick={() => setText("")}
              >
                Clear
              </Button>
            )}
            {mode === "write" && (
              <>
                {backendReady && (
                  <Button
                    className="py-5 px-4"
                    onClick={analyzeText}
                    disabled={!text || loading}
                  >
                    {loading ? <DotsRing className="size-4" /> : "Analyze →"}
                  </Button>
                )}
                {!backendReady && (
                  <Button
                    className="py-5 px-4"
                    disabled
                    aria-label="Backend is getting ready"
                  >
                    <DotsRing className="size-4" />
                    <span className="sr-only">Backend is getting ready</span>
                  </Button>
                )}
              </>
            )}
          </div>
        </div>

        {mode === "write" && !text && !result && (
          <p className="mt-2.5 text-[13px] text-muted-foreground">
            or{" "}
            <span
              onClick={() => setText(SAMPLE_TEXT)}
              className="cursor-pointer underline"
              style={{ color: "var(--accent-light)" }}
            >
              try with sample text
            </span>
          </p>
        )}

        {error && (
          <div className="mt-4 p-3 rounded-xl bg-red-50 border border-red-200 text-red-800 text-sm dark:bg-red-900 dark:border-red-700 dark:text-red-200">
            {error}
          </div>
        )}

        {mode === "review" && result && (
          <div className="relative mt-4 p-4 rounded-xl border border-border bg-card text-[15px] leading-relaxed text-foreground">
            <HighlightedText
              text={text}
              results={result}
              activeId={distortedResults[activeStep]?.id ?? null}
              onHover={(id) => setHoveredDistortion(id)}
              onClick={(id) =>
                setActiveStep(distortedResults.findIndex((r) => r.id === id))
              }
            />
            {hoveredDistortion &&
              DIST_STYLES[
                distortedResults.find((r) => r.id === hoveredDistortion)?.dist
              ] && (
                <Tooltip
                  result={distortedResults.find(
                    (r) => r.id === hoveredDistortion,
                  )}
                />
              )}
          </div>
        )}

        {mode === "review" && (
          <>
            <div className="flex justify-end gap-2 mt-2">
              <Button
                variant="outline"
                className="py-5 px-4"
                onClick={backToEdit}
              >
                ← Edit text
              </Button>
              <Button
                variant="outline"
                className="py-5 px-4"
                onClick={rewrite}
                disabled={rewriteLoading}
              >
                {rewriteLoading ? "Rewriting..." : "Rewrite with AI ✦"}
              </Button>
            </div>
            <span className="text-[13px] text-muted-foreground">{`${distortedResults.length} distortion${distortedResults.length === 1 ? "" : "s"} found`}</span>
          </>
        )}

        {mode === "review" && distortedResults.length > 0 && (
          <div className="mt-2">
            <StepperCard
              key={distortedResults[activeStep].id}
              result={distortedResults[activeStep]}
              dark={isDarkMode}
              total={distortedResults.length}
              current={activeStep}
              direction={stepDirection}
              onPrev={() => {
                setStepDirection("prev");
                setActiveStep(activeStep - 1);
              }}
              onNext={() => {
                setStepDirection("next");
                setActiveStep(activeStep + 1);
              }}
              sendFeedback={sendFeedback}
              feedbackSubmitted={feedbackSubmitted}
            />
          </div>
        )}

        {mode === "review" && result && distortedResults.length === 0 && (
          <div
            className="mt-6 p-5 rounded-xl border"
            style={{
              background: "var(--accent-faint)",
              borderColor: "var(--accent-muted)",
            }}
          >
            <p
              className="text-[17px] text-foreground m-0"
              style={{ fontFamily: "var(--font-instrument-serif)" }}
            >
              No distortions detected. Your thinking looks balanced here.
            </p>
          </div>
        )}
        {rewritten && (
          <div
            className="mt-5 p-5 rounded-xl border border-border bg-card"
            style={{
              animation: "revealUp 0.4s cubic-bezier(0.23,1,0.32,1) both",
            }}
          >
            <p
              className="text-[10px] font-semibold uppercase tracking-widest mb-2.5"
              style={{ color: "var(--accent-light)" }}
            >
              AI Rewritten Entry
            </p>
            <p
              className="text-[15px] leading-relaxed m-0 text-foreground"
              style={{ fontFamily: "var(--font-instrument-serif)" }}
            >
              {rewritten}
            </p>
          </div>
        )}

        {rewriteError && (
          <div className="mt-4 p-3 rounded-xl bg-red-50 border border-red-200 text-red-800 text-sm dark:bg-red-900 dark:border-red-700 dark:text-red-200">
            {rewriteError}
          </div>
        )}
      </main>

      <footer className="text-center px-6 py-8 text-[13px] italic text-muted-foreground">
        Reframe is a journaling aid, not a clinical tool (~50% accuracy). Your
        feedback helps improve predictions over time.
      </footer>
    </>
  );
}
