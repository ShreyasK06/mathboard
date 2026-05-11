import katex from "katex";

import { AlertCircleIcon, CheckIcon, CopyIcon } from "./Icons.jsx";

function renderLatex(latex, displayMode = false) {
  return {
    __html: katex.renderToString(latex || "\\,", {
      throwOnError: false,
      displayMode,
    }),
  };
}

export default function ResultPanel({
  isLoading,
  error,
  result,
  copied,
  onCopyLatex,
}) {
  if (!isLoading && !error && !result) return null;

  return (
    <div className="results">
      {isLoading && (
        <div className="result-section" aria-live="polite">
          <div className="loading-row">
            <span className="shimmer-text">Recognizing handwriting&hellip;</span>
          </div>
          <div className="loading-bar" aria-hidden="true" />
        </div>
      )}

      {!isLoading && error && (
        <div className="result-error" role="alert">
          <AlertCircleIcon />
          <div>
            <div className="result-error-title">Could not convert</div>
            <div className="result-error-body">{error}</div>
          </div>
        </div>
      )}

      {!isLoading && result && (
        <div className="result-success">
          {result.operationLabel && (
            <div className="badge-row">
              <div className="operation-badge">{result.operationLabel}</div>
            </div>
          )}

          <div className="result-card result-card-input">
            <div className="result-label-row">
              <h3 className="result-label">Recognized expression</h3>
              {result.latex && (
                <button
                  type="button"
                  className={copied ? "copy-btn copied" : "copy-btn"}
                  onClick={onCopyLatex}
                  aria-label="Copy LaTeX to clipboard"
                >
                  {copied ? <CheckIcon /> : <CopyIcon />}
                  <span>{copied ? "Copied" : "Copy LaTeX"}</span>
                </button>
              )}
            </div>
            <div
              className="latex-hero"
              dangerouslySetInnerHTML={renderLatex(result.latex, true)}
            />
            <div className="latex-raw">
              <code>{result.latex}</code>
            </div>
          </div>

          {result.steps && result.steps.length > 0 && (
            <details className="steps-details">
              <summary className="steps-summary">Steps ({result.steps.length})</summary>
              <ol className="steps-list">
                {result.steps.map((step, i) => (
                  <li key={i}>{step}</li>
                ))}
              </ol>
            </details>
          )}

          <div className="result-equals-divider" aria-hidden="true">
            <span className="result-equals-line" />
            <span className="result-equals-sym">=</span>
            <span className="result-equals-line" />
          </div>

          <div className="result-card result-card-solution">
            <h3 className="result-label">Solution</h3>
            {result.isSolutionError ? (
              <div className="result-error">
                <AlertCircleIcon />
                <div className="result-error-body">{result.solution}</div>
              </div>
            ) : result.latexResult ? (
              <div
                className="latex-hero latex-solution-hero"
                dangerouslySetInnerHTML={renderLatex(result.latexResult, true)}
              />
            ) : (
              <div className="solution-text">{result.solution}</div>
            )}
          </div>

          {result.agreement === "match" && (
            <div className="agreement agreement-match">
              ✓ Cross-checked with Gemini (results agree)
            </div>
          )}
          {result.agreement === "differ" && result.crosscheckLatex && (
            <div className="agreement agreement-differ">
              ⚠ Gemini got a different answer:&nbsp;
              <span dangerouslySetInnerHTML={renderLatex(result.crosscheckLatex)} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
