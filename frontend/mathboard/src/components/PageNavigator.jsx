import {
  ChevronLeftIcon,
  ChevronRightIcon,
  PlusIcon,
  TrashIcon,
} from "./Icons.jsx";

export default function PageNavigator({
  currentIndex,
  pageCount,
  onPrev,
  onNext,
  onAdd,
  onDelete,
  saving,
}) {
  const canPrev = currentIndex > 0;
  const canNext = currentIndex < pageCount - 1;
  const canDelete = pageCount > 1;
  return (
    <div className="page-nav" role="group" aria-label="Page navigation">
      <button
        type="button"
        className="icon-btn"
        onClick={onPrev}
        disabled={!canPrev}
        aria-label="Previous page"
        title="Previous page"
      >
        <ChevronLeftIcon />
      </button>
      <span className="page-nav-label" aria-live="polite">
        Page <strong>{currentIndex + 1}</strong> of {pageCount}
      </span>
      <button
        type="button"
        className="icon-btn"
        onClick={onNext}
        disabled={!canNext}
        aria-label="Next page"
        title="Next page"
      >
        <ChevronRightIcon />
      </button>
      <div className="page-nav-spacer" />
      <span
        className={`page-nav-status ${saving ? "saving" : ""}`}
        aria-live="polite"
      >
        {saving ? "Saving…" : "Saved"}
      </span>
      <button
        type="button"
        className="icon-btn"
        onClick={onAdd}
        aria-label="Add page"
        title="Add page"
      >
        <PlusIcon />
      </button>
      <button
        type="button"
        className="icon-btn danger"
        onClick={onDelete}
        disabled={!canDelete}
        aria-label="Delete page"
        title="Delete page"
      >
        <TrashIcon />
      </button>
    </div>
  );
}
