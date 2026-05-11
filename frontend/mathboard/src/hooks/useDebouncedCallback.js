import { useCallback, useEffect, useRef } from "react";

/**
 * Returns a stable callback that delays invoking `fn` until `delay` ms have
 * elapsed since the last call. Cancels on unmount.
 */
export function useDebouncedCallback(fn, delay) {
  const fnRef = useRef(fn);
  const timerRef = useRef(null);

  useEffect(() => {
    fnRef.current = fn;
  }, [fn]);

  useEffect(() => () => {
    if (timerRef.current) clearTimeout(timerRef.current);
  }, []);

  return useCallback((...args) => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      timerRef.current = null;
      fnRef.current(...args);
    }, delay);
  }, [delay]);
}
