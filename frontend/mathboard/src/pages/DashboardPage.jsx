import { SHINY_URL } from "../services/backend.js";

export default function DashboardPage() {
  return (
    <main className="explorer">
      <iframe
        className="explorer-frame"
        src={SHINY_URL}
        title="MathBoard Model & Activity Dashboard"
      />
    </main>
  );
}
