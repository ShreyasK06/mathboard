import { EraserIcon, PenIcon, RedoIcon, TrashIcon, UndoIcon } from "./Icons.jsx";

export default function Toolbar({
  tool,
  setTool,
  penSize,
  setPenSize,
  eraserSize,
  setEraserSize,
  penColor,
  setPenColor,
  onUndo,
  onRedo,
  onClear,
  canUndo,
  canRedo,
}) {
  const isEraser = tool === "eraser";
  const size = isEraser ? eraserSize : penSize;

  return (
    <div className="toolbar">
      <div className="tool-group" role="group" aria-label="Drawing tool">
        <button
          type="button"
          className={tool === "pen" ? "icon-btn active" : "icon-btn"}
          onClick={() => setTool("pen")}
          data-tooltip="Pen"
          aria-label="Pen"
          aria-pressed={tool === "pen"}
        >
          <PenIcon />
        </button>
        <button
          type="button"
          className={tool === "eraser" ? "icon-btn active" : "icon-btn"}
          onClick={() => setTool("eraser")}
          data-tooltip="Eraser"
          aria-label="Eraser"
          aria-pressed={tool === "eraser"}
        >
          <EraserIcon />
        </button>
      </div>

      <label className="control" title="Stroke thickness">
        <span>Size</span>
        <input
          type="range"
          min="1"
          max="40"
          value={size}
          onChange={(e) => {
            const v = Number(e.target.value);
            if (isEraser) setEraserSize(v);
            else setPenSize(v);
          }}
        />
        <span className="value">{size}</span>
      </label>

      <label
        className="control"
        title={isEraser ? "Switch to Pen to pick a color" : "Pen color"}
      >
        <span>Color</span>
        <input
          type="color"
          value={penColor}
          onChange={(e) => {
            setPenColor(e.target.value);
            if (isEraser) setTool("pen");
          }}
          disabled={isEraser}
          aria-label="Pen color"
        />
      </label>

      <div className="toolbar-spacer" />

      <div className="tool-group" role="group" aria-label="History">
        <button
          type="button"
          className="icon-btn"
          onClick={onUndo}
          disabled={!canUndo}
          data-tooltip="Undo"
          aria-label="Undo"
        >
          <UndoIcon />
        </button>
        <button
          type="button"
          className="icon-btn"
          onClick={onRedo}
          disabled={!canRedo}
          data-tooltip="Redo"
          aria-label="Redo"
        >
          <RedoIcon />
        </button>
        <button
          type="button"
          className="icon-btn danger"
          onClick={onClear}
          data-tooltip="Clear page"
          aria-label="Clear page"
        >
          <TrashIcon />
        </button>
      </div>
    </div>
  );
}
