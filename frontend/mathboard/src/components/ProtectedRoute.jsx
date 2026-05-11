import { Navigate, useLocation } from "react-router-dom";

import { useAuth } from "../contexts/AuthContext.jsx";

export default function ProtectedRoute({ children }) {
  const { user, loading, configured } = useAuth();
  const location = useLocation();

  if (!configured) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }
  if (loading) {
    return (
      <div className="page-loading" role="status" aria-live="polite">
        <span className="shimmer-text">Loading&hellip;</span>
      </div>
    );
  }
  if (!user) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }
  return children;
}
