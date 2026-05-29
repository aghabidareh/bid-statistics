import { Head } from "@inertiajs/react";
import SiteHeader from "../components/SiteHeader";
import RegressionTable from "../components/regression/RegressionTable";

export default function Index({ catalog }) {
  return (
    <>
      <Head title="Regression" />
      <SiteHeader currentPath="/regression" />
      <main className="home">
        <section className="home__section">
          <div className="home__eyebrow">Bid statistics</div>
          <h1 className="home__title">Regression</h1>
          <p className="home__intro">
            Explore the regression catalog with spreadsheet-style data entry,
            prediction rows, import support, and dedicated workflows for linear
            models, logistic models, and propensity score matching.
          </p>
          <div className="home__summary">
            <div className="home__summary-card">
              <div className="home__summary-label">Catalog size</div>
              <div className="home__summary-value">{catalog.length}</div>
            </div>
            <div className="home__summary-card">
              <div className="home__summary-label">Workflow</div>
              <div className="home__summary-value">Spreadsheet dataset</div>
            </div>
          </div>
          <RegressionTable calculators={catalog} />
        </section>
      </main>
    </>
  );
}
