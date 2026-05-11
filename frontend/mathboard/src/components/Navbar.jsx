import { Link, NavLink, useNavigate } from "react-router-dom";

import { useAuth } from "../contexts/AuthContext.jsx";
import { SignOutIcon } from "./Icons.jsx";

export default function Navbar() {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();

  const handleSignOut = async () => {
    await signOut();
    navigate("/", { replace: true });
  };

  return (
    <header className="app-navbar">
      <div className="navbar-inner">
        {/* Left: brand */}
        <Link to={user ? "/notebooks" : "/"} className="brand">
          <div className="brand-glyph" aria-hidden="true">M</div>
          <span>Mathboard</span>
        </Link>

        {/* Center: pill tabs */}
        <nav className="navbar-links">
          <NavLink to="/" end className={({ isActive }) => `navbar-link${isActive ? " active" : ""}`}>
            Overview
          </NavLink>
          <NavLink to="/notebooks" className={({ isActive }) => `navbar-link${isActive ? " active" : ""}`}>
            Boards
          </NavLink>
          <NavLink to="/dashboard" className={({ isActive }) => `navbar-link${isActive ? " active" : ""}`}>
            Activity
          </NavLink>
        </nav>

        {/* Right: user / auth */}
        <div className="navbar-actions">
          {user ? (
            <div className="navbar-user">
              <span className="user-name" title={user.email || ""}>
                · {(user.email || "").toLowerCase()}
              </span>
              {user.photoURL ? (
                <img src={user.photoURL} alt="" className="user-avatar" referrerPolicy="no-referrer" />
              ) : (
                <div className="user-avatar user-avatar-fallback" aria-hidden="true">
                  {(user.displayName || user.email || "?").charAt(0).toUpperCase()}
                </div>
              )}
              <button
                type="button"
                className="icon-btn"
                onClick={handleSignOut}
                aria-label="Sign out"
                title="Sign out"
              >
                <SignOutIcon />
              </button>
            </div>
          ) : (
            <>
              <Link to="/login" className="navbar-link">Sign in</Link>
              <Link to="/login" className="navbar-cta">Get started</Link>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
