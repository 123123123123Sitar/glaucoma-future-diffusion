"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type CaseImage = {
  caseId: string;
  horizonYears: number;
  laterality: string;
  baseline: string;
  candidateA: string;
  candidateB: string;
};

type Manifest = {
  schemaVersion: string;
  warning: string;
  cases: CaseImage[];
};

type ModelStatus = {
  progressionOutputReleased: boolean;
  diffusionOutputReleased: boolean;
  status: string;
  message: string;
};

type Progression = "none" | "possible" | "definite";
type Authenticity = "A" | "B" | "unsure";
type Diagnosis = "normal" | "glaucoma" | "uncertain";
type SaveState = "idle" | "saving" | "saved" | "error";
type ComparisonMode = "side-by-side" | "blink" | "highlight";

type Draft = {
  authenticity: Authenticity;
  diagnosisBaseline: Diagnosis | "";
  diagnosisA: Diagnosis | "";
  diagnosisB: Diagnosis | "";
  progressionA: Progression | "";
  progressionB: Progression | "";
  features: string[];
  confidence: number;
  notes: string;
};

const features = [
  "Rim thinning / increased cupping",
  "Vessel displacement or bayoneting",
  "Disc hemorrhage",
  "Peripapillary change",
  "Acquisition or registration artifact",
  "No meaningful structural change",
];

const blankDraft = (): Draft => ({
  authenticity: "unsure",
  diagnosisBaseline: "",
  diagnosisA: "",
  diagnosisB: "",
  progressionA: "",
  progressionB: "",
  features: [],
  confidence: 50,
  notes: "",
});

function ChangeMap({
  baseline,
  candidate,
  label,
}: {
  baseline: string;
  candidate: string;
  label: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let cancelled = false;
    const baselineImage = new Image();
    const candidateImage = new Image();

    Promise.all([
      new Promise<void>((resolve, reject) => {
        baselineImage.onload = () => resolve();
        baselineImage.onerror = () => reject(new Error("baseline image failed"));
        baselineImage.src = baseline;
      }),
      new Promise<void>((resolve, reject) => {
        candidateImage.onload = () => resolve();
        candidateImage.onerror = () => reject(new Error("candidate image failed"));
        candidateImage.src = candidate;
      }),
    ]).then(() => {
      if (cancelled || !canvasRef.current) return;
      const canvas = canvasRef.current;
      const size = 512;
      canvas.width = size;
      canvas.height = size;
      const output = canvas.getContext("2d", { willReadFrequently: true });
      if (!output) return;

      const scratch = document.createElement("canvas");
      scratch.width = size;
      scratch.height = size;
      const context = scratch.getContext("2d", { willReadFrequently: true });
      if (!context) return;
      context.drawImage(baselineImage, 0, 0, size, size);
      const before = context.getImageData(0, 0, size, size);
      context.clearRect(0, 0, size, size);
      context.drawImage(candidateImage, 0, 0, size, size);
      const after = context.getImageData(0, 0, size, size);
      const mapped = output.createImageData(size, size);

      for (let pixel = 0; pixel < before.data.length; pixel += 4) {
        const red = before.data[pixel];
        const green = before.data[pixel + 1];
        const blue = before.data[pixel + 2];
        const gray = Math.round((red * 0.21 + green * 0.72 + blue * 0.07) * 0.42);
        const difference =
          (Math.abs(red - after.data[pixel]) +
            Math.abs(green - after.data[pixel + 1]) +
            Math.abs(blue - after.data[pixel + 2])) /
          3;
        // Fixed high-sensitivity scale: reveal subtle change without normalizing
        // each candidate independently or changing the source photographs.
        const strength = Math.max(0, Math.min(1, (difference - 0.75) / 12));
        mapped.data[pixel] = Math.round(gray * (1 - strength) + 255 * strength);
        mapped.data[pixel + 1] = Math.round(gray * (1 - strength) + 205 * strength * (1 - strength));
        mapped.data[pixel + 2] = Math.round(gray * (1 - strength));
        mapped.data[pixel + 3] = 255;
      }
      output.putImageData(mapped, 0, 0);
    }).catch(() => {
      // The original photographs remain available if a browser cannot draw the aid.
    });

    return () => {
      cancelled = true;
    };
  }, [baseline, candidate]);

  return <canvas ref={canvasRef} role="img" aria-label={`${label} highlighted pixel-change map`} />;
}

function ComparisonCard({
  label,
  baseline,
  candidate,
  mode,
  blinkShowsCandidate,
}: {
  label: string;
  baseline: string;
  candidate: string;
  mode: ComparisonMode;
  blinkShowsCandidate: boolean;
}) {
  return (
    <article className="comparison-card">
      <div className="comparison-card-title">
        <strong>Baseline ↔ {label}</strong>
        {mode === "blink" && <span>{blinkShowsCandidate ? label : "Baseline"}</span>}
      </div>
      {mode === "side-by-side" && (
        <div className="matched-pair">
          <figure><img src={baseline} alt="Baseline fundus" /><figcaption>Baseline</figcaption></figure>
          <figure><img src={candidate} alt={`${label} fundus`} /><figcaption>{label}</figcaption></figure>
        </div>
      )}
      {mode === "blink" && (
        <img
          className="blink-image"
          src={blinkShowsCandidate ? candidate : baseline}
          alt={blinkShowsCandidate ? `${label} fundus` : "Baseline fundus"}
        />
      )}
      {mode === "highlight" && <ChangeMap baseline={baseline} candidate={candidate} label={label} />}
    </article>
  );
}

function shuffle<T>(items: T[], seedText: string): T[] {
  const copy = [...items];
  let seed = [...seedText].reduce((value, char) => (value * 31 + char.charCodeAt(0)) >>> 0, 2166136261);
  for (let index = copy.length - 1; index > 0; index -= 1) {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    const target = seed % (index + 1);
    [copy[index], copy[target]] = [copy[target], copy[index]];
  }
  return copy;
}

export default function Home() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [sessionId] = useState(() => crypto.randomUUID());
  const [cases, setCases] = useState<CaseImage[]>([]);
  const [index, setIndex] = useState(0);
  const [draft, setDraft] = useState<Draft>(blankDraft);
  const [startedAt, setStartedAt] = useState(Date.now());
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [complete, setComplete] = useState(false);
  const [feedback, setFeedback] = useState("");
  const [reviewerCode, setReviewerCode] = useState("");
  const [specialty, setSpecialty] = useState("Optometry");
  const [yearsExperience, setYearsExperience] = useState("");
  const [profileOpen, setProfileOpen] = useState(false);
  const [expanded, setExpanded] = useState<{ src: string; label: string; baseline?: string } | null>(null);
  const [showBaseline, setShowBaseline] = useState(false);
  const [comparisonMode, setComparisonMode] = useState<ComparisonMode>("blink");
  const [blinkShowsCandidate, setBlinkShowsCandidate] = useState(false);

  useEffect(() => {
    fetch("/cases/manifest.json")
      .then((response) => response.json())
      .then((data: Manifest) => {
        setManifest(data);
        setCases(shuffle(data.cases, sessionId));
        setStartedAt(Date.now());
      });
  }, [sessionId]);

  useEffect(() => {
    fetch("/model-status.json")
      .then((response) => response.json())
      .then((data: ModelStatus) => setModelStatus(data));
  }, []);

  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      if (!expanded?.baseline || event.key.toLowerCase() !== "b") return;
      setShowBaseline(event.type === "keydown");
    };
    window.addEventListener("keydown", key);
    window.addEventListener("keyup", key);
    return () => {
      window.removeEventListener("keydown", key);
      window.removeEventListener("keyup", key);
    };
  }, [expanded]);

  useEffect(() => {
    if (comparisonMode !== "blink") return;
    setBlinkShowsCandidate(false);
    const timer = window.setInterval(
      () => setBlinkShowsCandidate((value) => !value),
      700,
    );
    return () => window.clearInterval(timer);
  }, [comparisonMode, cases[index]?.caseId]);

  const current = cases[index];
  const canSubmit = Boolean(
    draft.progressionA &&
      draft.progressionB &&
      draft.diagnosisBaseline &&
      draft.diagnosisA &&
      draft.diagnosisB,
  );
  const apiBase = process.env.NEXT_PUBLIC_STUDY_API_URL?.replace(/\/$/, "") ?? "";
  const progress = cases.length ? ((index + 1) / cases.length) * 100 : 0;
  const statusLabel = useMemo(
    () => ({ idle: "", saving: "Saving…", saved: "Saved to the study database", error: "Save failed — try again" })[saveState],
    [saveState],
  );

  async function saveCase() {
    if (!current || !canSubmit || saveState === "saving") return;
    setSaveState("saving");
    try {
      const response = await fetch(`${apiBase}/api/study`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId,
          profile: { reviewerCode, specialty, yearsExperience },
          response: {
            ...draft,
            caseId: current.caseId,
            caseOrder: index + 1,
            elapsedSeconds: Math.round((Date.now() - startedAt) / 1000),
          },
        }),
      });
      if (!response.ok) throw new Error("save failed");
      setSaveState("saved");
      if (index === cases.length - 1) {
        setComplete(true);
      } else {
        setIndex((value) => value + 1);
        setDraft(blankDraft());
        setStartedAt(Date.now());
        window.scrollTo({ top: 0, behavior: "smooth" });
        setTimeout(() => setSaveState("idle"), 900);
      }
    } catch {
      setSaveState("error");
    }
  }

  async function finish() {
    setSaveState("saving");
    try {
      const response = await fetch(`${apiBase}/api/study`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ type: "complete", sessionId, feedback }),
      });
      if (!response.ok) throw new Error("save failed");
      setSaveState("saved");
    } catch {
      setSaveState("error");
    }
  }

  if (!manifest || !current) {
    return <main className="loading"><span /><p>Preparing blinded cases…</p></main>;
  }

  if (complete) {
    return (
      <main className="complete-shell">
        <section className="complete-card">
          <p className="eyebrow">REVIEW COMPLETE</p>
          <h1>Thank you—your twenty ratings were saved.</h1>
          <p>
            Your answers were saved to the study database. We will compare
            clinicians&apos; answers with the hidden answer key to see whether
            the computer-made images look believable and keep the correct
            disease appearance.
          </p>
          <label>
            One final comment <span>optional</span>
            <textarea
              value={feedback}
              onChange={(event) => setFeedback(event.target.value)}
              placeholder="What made the generated images look credible or artificial?"
            />
          </label>
          <button className="primary" onClick={finish} disabled={saveState === "saving"}>
            Save comment
          </button>
          <small className={saveState === "error" ? "error" : ""}>{statusLabel}</small>
        </section>
      </main>
    );
  }

  return (
    <main>
      <header className="topbar">
        <div className="brand"><b>RP</b><span><strong>RetinaProgress</strong><small>Clinician image study</small></span></div>
        <div className="top-actions">
          <span className="research-pill">
            {modelStatus?.progressionOutputReleased
              ? "Research candidate"
              : "20 image comparisons"}
          </span>
          <button className="profile-button" onClick={() => setProfileOpen(!profileOpen)}>Reviewer details</button>
        </div>
      </header>

      {profileOpen && (
        <aside className="profile-panel">
          <label>Reviewer code <input value={reviewerCode} onChange={(event) => setReviewerCode(event.target.value)} placeholder="Optional" /></label>
          <label>Clinical background
            <select value={specialty} onChange={(event) => setSpecialty(event.target.value)}>
              <option>Optometry</option><option>Ophthalmology</option>
              <option>Glaucoma specialist</option><option>Retina specialist</option>
              <option>Other eye-care professional</option>
            </select>
          </label>
          <label>Years reviewing fundus images <input type="number" min="0" max="80" value={yearsExperience} onChange={(event) => setYearsExperience(event.target.value)} placeholder="Optional" /></label>
          <button onClick={() => setProfileOpen(false)}>Done</button>
        </aside>
      )}

      <section className="study-shell">
        <div className="case-header">
          <div><p>CASE {String(index + 1).padStart(2, "0")}</p><h1>{current.caseId}</h1></div>
          <div className="progress"><span>{index + 1} of {cases.length}</span><i><b style={{ width: `${progress}%` }} /></i></div>
          <div className="case-meta"><span>{current.laterality === "OD" ? "Right eye" : "Left eye"}</span><strong>{current.horizonYears.toFixed(1)}-year interval</strong></div>
        </div>

        {index === 0 && (
          <>
            <div className="protocol-strip">
              <strong>What is this test?</strong> For each eye, you will see
              the first photo and two possible later photos. One later photo
              is the comparison image. The other was made by our computer
              model. Some eyes are healthy and some have glaucoma. Tell us
              which image looks more believable and whether you see glaucoma
              or worsening. We hide the answers until the study is analyzed.
            </div>
            {modelStatus && !modelStatus.progressionOutputReleased && (
              <div className="gate-strip" role="status">
                <strong>Research use only.</strong> These images are being
                tested and must not be used to diagnose or treat a patient.
              </div>
            )}
          </>
        )}

        <div className="image-grid">
          {[
            { label: "Baseline", kind: "REFERENCE", src: current.baseline },
            { label: "Follow-up A", kind: "CANDIDATE", src: current.candidateA },
            { label: "Follow-up B", kind: "CANDIDATE", src: current.candidateB },
          ].map((image) => (
            <figure key={image.label}>
              <figcaption><span>{image.kind}</span><strong>{image.label}</strong></figcaption>
              <button onClick={() => setExpanded({ src: image.src, label: image.label, baseline: image.kind === "CANDIDATE" ? current.baseline : undefined })}>
                <img src={image.src} alt={`${image.label} fundus`} />
                <em>＋ Enlarge</em>
              </button>
            </figure>
          ))}
        </div>

        <section className="comparison-lab" aria-label="Image comparison tools">
          <div className="comparison-intro">
            <div>
              <p className="eyebrow">COMPARISON TOOLS</p>
              <h2>Having trouble seeing a difference?</h2>
              <p>
                Use the same view for A and B. Blink swaps each later image
                with the baseline. Highlight changes makes small pixel
                differences easier to see by magnifying them, without changing
                the study images.
              </p>
            </div>
            <div className="mode-switch" role="group" aria-label="Comparison view">
              {([
                ["side-by-side", "Side by side"],
                ["blink", "Blink"],
                ["highlight", "Highlight changes"],
              ] as const).map(([mode, label]) => (
                <button
                  key={mode}
                  className={comparisonMode === mode ? "selected" : ""}
                  onClick={() => setComparisonMode(mode)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          <div className="comparison-grid">
            <ComparisonCard
              label="Follow-up A"
              baseline={current.baseline}
              candidate={current.candidateA}
              mode={comparisonMode}
              blinkShowsCandidate={blinkShowsCandidate}
            />
            <ComparisonCard
              label="Follow-up B"
              baseline={current.baseline}
              candidate={current.candidateB}
              mode={comparisonMode}
              blinkShowsCandidate={blinkShowsCandidate}
            />
          </div>
          {comparisonMode === "highlight" && (
            <p className="map-note">
              This is a high-sensitivity map: dark areas changed little, while
              yellow or red areas changed more. Color intensity is amplified.
              Camera, compression, and alignment differences can also light
              up, so confirm your answer using the original photographs above.
            </p>
          )}
        </section>

        <section className="assessment">
          <div className="question">
            <span>1</span><div><h2>Which later image is the comparison image?</h2><p>The other image was made by the computer model. Choose “Cannot tell” if you are unsure.</p>
              <div className="choices three">
                {(["A", "B", "unsure"] as const).map((choice) => <button key={choice} className={draft.authenticity === choice ? "selected" : ""} onClick={() => setDraft({ ...draft, authenticity: choice })}>{choice === "unsure" ? "Cannot tell" : `Follow-up ${choice}`}</button>)}
              </div>
            </div>
          </div>
          <div className="question">
            <span>2</span><div className="wide"><h2>How would you classify each image?</h2>
              <div className="progression-grid diagnosis-grid">
                {([
                  ["Baseline", "diagnosisBaseline"],
                  ["Candidate A", "diagnosisA"],
                  ["Candidate B", "diagnosisB"],
                ] as const).map(([label, key]) => (
                  <div key={key}><p>{label}</p><div className="choices">
                    {(["normal", "glaucoma", "uncertain"] as const).map((choice) => (
                      <button key={choice} className={draft[key] === choice ? "selected" : ""} onClick={() => setDraft({ ...draft, [key]: choice })}>
                        {choice[0].toUpperCase() + choice.slice(1)}
                      </button>
                    ))}
                  </div></div>
                ))}
              </div>
            </div>
          </div>
          <div className="question">
            <span>3</span><div className="wide"><h2>Does the eye appear to be getting worse?</h2>
              <div className="progression-grid">
                {(["A", "B"] as const).map((side) => {
                  const key = side === "A" ? "progressionA" : "progressionB";
                  const labels = { none: "No change", possible: "Maybe worse", definite: "Clearly worse" };
                  return <div key={side}><p>Follow-up {side}</p><div className="choices">{(["none", "possible", "definite"] as const).map((choice) => <button key={choice} className={draft[key] === choice ? "selected" : ""} onClick={() => setDraft({ ...draft, [key]: choice })}>{labels[choice]}</button>)}</div></div>;
                })}
              </div>
            </div>
          </div>
          <div className="question">
            <span>4</span><div className="wide"><h2>What did you notice?</h2><div className="feature-grid">
              {features.map((feature) => <label key={feature}><input type="checkbox" checked={draft.features.includes(feature)} onChange={() => setDraft({ ...draft, features: draft.features.includes(feature) ? draft.features.filter((item) => item !== feature) : [...draft.features, feature] })} /><i>{feature}</i></label>)}
            </div></div>
          </div>
          <div className="question confidence">
            <span>5</span><div><h2>Confidence</h2><p>Overall confidence in this case.</p></div>
            <input type="range" min="0" max="100" value={draft.confidence} onChange={(event) => setDraft({ ...draft, confidence: Number(event.target.value) })} /><strong>{draft.confidence}%</strong>
          </div>
          <label className="notes">Optional note<textarea value={draft.notes} onChange={(event) => setDraft({ ...draft, notes: event.target.value })} placeholder="Subtle findings, artifacts, or uncertainty…" /></label>
          <div className="submit-row"><small className={saveState === "error" ? "error" : ""}>{statusLabel}</small><button className="primary" onClick={saveCase} disabled={!canSubmit || saveState === "saving"}>{index === cases.length - 1 ? "Save & finish" : "Save & next case"} <span>→</span></button></div>
        </section>
      </section>

      {expanded && (
        <div className="lightbox" role="dialog" aria-modal="true" onClick={() => setExpanded(null)}>
          <button className="close" aria-label="Close">×</button><p>{showBaseline ? "Baseline reference" : expanded.label}</p>
          <img src={showBaseline && expanded.baseline ? expanded.baseline : expanded.src} alt={showBaseline ? "Baseline reference" : expanded.label} />
          {expanded.baseline && <button className="blink" onClick={(event) => event.stopPropagation()} onMouseDown={() => setShowBaseline(true)} onMouseUp={() => setShowBaseline(false)} onMouseLeave={() => setShowBaseline(false)} onTouchStart={() => setShowBaseline(true)} onTouchEnd={() => setShowBaseline(false)}>Hold to show baseline · keyboard B</button>}
        </div>
      )}
      <footer>{manifest.warning}</footer>
    </main>
  );
}
