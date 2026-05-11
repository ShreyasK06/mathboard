/* Inline icon set. Each icon is a plain SVG component so we can drop the
   icon-library dependency and keep bundle size down. */

const baseProps = {
  fill: "none",
  stroke: "currentColor",
  strokeLinecap: "round",
  strokeLinejoin: "round",
  strokeWidth: 2,
};

const make = (paths, { width = 18, height = 18, viewBox = "0 0 24 24", strokeWidth } = {}) =>
  function Icon(props) {
    return (
      <svg
        width={width}
        height={height}
        viewBox={viewBox}
        {...baseProps}
        {...(strokeWidth ? { strokeWidth } : null)}
        {...props}
      >
        {paths}
      </svg>
    );
  };

export const PenIcon = make(
  <>
    <path d="M12 19l7-7 3 3-7 7-3-3z" />
    <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z" />
    <path d="M2 2l7.586 7.586" />
    <circle cx="11" cy="11" r="2" />
  </>,
);

export const EraserIcon = make(
  <>
    <path d="M20 20H7L3 16a2 2 0 0 1 0-2.8L13.2 3a2 2 0 0 1 2.8 0L21 8a2 2 0 0 1 0 2.8L11 21" />
    <path d="M14 4l6 6" />
  </>,
);

export const UndoIcon = make(
  <>
    <path d="M3 7v6h6" />
    <path d="M3 13a9 9 0 1 0 3-7L3 9" />
  </>,
);

export const RedoIcon = make(
  <>
    <path d="M21 7v6h-6" />
    <path d="M21 13a9 9 0 1 1-3-7l3 3" />
  </>,
);

export const TrashIcon = make(
  <>
    <path d="M3 6h18" />
    <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
    <path d="M10 11v6M14 11v6" />
  </>,
);

export const SparklesIcon = make(
  <path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M5.6 18.4l2.8-2.8M15.6 8.4l2.8-2.8" />,
  { width: 16, height: 16 },
);

export const SunIcon = make(
  <>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
  </>,
);

export const MoonIcon = make(
  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />,
);

export const CopyIcon = make(
  <>
    <rect x="9" y="9" width="13" height="13" rx="2" />
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </>,
  { width: 14, height: 14 },
);

export const CheckIcon = make(<path d="M20 6L9 17l-5-5" />, {
  width: 14,
  height: 14,
  strokeWidth: 2.5,
});

export const AlertTriangleIcon = make(
  <>
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </>,
);

export const AlertCircleIcon = make(
  <>
    <circle cx="12" cy="12" r="10" />
    <line x1="12" y1="8" x2="12" y2="12" />
    <line x1="12" y1="16" x2="12.01" y2="16" />
  </>,
);

export const LogoIcon = make(
  <>
    <path d="M4 19l4-12 3 8 2-5 3 9" />
    <path d="M3 21h18" />
  </>,
  { width: 22, height: 22, strokeWidth: 2.2 },
);

export const DrawCanvasIcon = make(
  <>
    <rect x="2" y="3" width="20" height="14" rx="2" />
    <path d="M8 21h8M12 17v4" />
    <path d="M7 9l2 2 4-4" />
  </>,
  { width: 24, height: 24, strokeWidth: 1.8 },
);

export const RecognizeIcon = make(
  <>
    <circle cx="11" cy="11" r="8" />
    <path d="M21 21l-4.35-4.35" />
    <path d="M11 8v6M8 11h6" />
  </>,
  { width: 24, height: 24, strokeWidth: 1.8 },
);

export const CalculatorIcon = make(
  <>
    <rect x="4" y="2" width="16" height="20" rx="2" />
    <path d="M8 6h8" />
    <path d="M8 10h2M14 10h2M8 14h2M14 14h2M8 18h2M14 18h2" />
  </>,
  { width: 24, height: 24, strokeWidth: 1.8 },
);

export const GitHubIcon = make(
  <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22" />,
  { width: 16, height: 16 },
);

export const PencilDrawIcon = make(
  <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z" />,
  { width: 32, height: 32, strokeWidth: 1.5 },
);

export const ChevronLeftIcon = make(<path d="M15 18l-6-6 6-6" />, {
  strokeWidth: 2.5,
});

export const ChevronRightIcon = make(<path d="M9 18l6-6-6-6" />, {
  strokeWidth: 2.5,
});

export const PlusIcon = make(<path d="M12 5v14M5 12h14" />);

export const NotebookIcon = make(
  <>
    <path d="M2 3h20v18H2z" />
    <path d="M8 3v18" />
    <path d="M2 8h6M2 14h6" />
  </>,
);

export const SignOutIcon = make(
  <>
    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
    <polyline points="16 17 21 12 16 7" />
    <line x1="21" y1="12" x2="9" y2="12" />
  </>,
);
