import { Head } from "@inertiajs/react";
import CalculatorTable from "./components/test-statistics/CalculatorTable";

export default function Home({ catalog }) {
  return (
    <>
      <Head title="Test statistics calculators" />
      <main className="home">
        <section className="home__section">
          <div className="home__eyebrow">Bid statistics</div>
          <h1 className="home__title">Test statistics calculators</h1>
          <p className="home__intro">
            Explore the complete 26-test catalog with one shared, metadata-driven
            workflow for parametric, nonparametric, ANOVA, survival, and ROC
            comparison methods.
          </p>
          <div className="home__summary">
            <div className="home__summary-card">
              <div className="home__summary-label">Catalog size</div>
              <div className="home__summary-value">{catalog.length}</div>
            </div>
            <div className="home__summary-card">
              <div className="home__summary-label">Delivery shape</div>
              <div className="home__summary-value">One shared UI</div>
            </div>
          </div>
          <CalculatorTable calculators={catalog} />
        </section>
      </main>
    </>
  );
}
