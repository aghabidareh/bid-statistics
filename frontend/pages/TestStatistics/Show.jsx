import { Head, Link } from "@inertiajs/react";
import CalculatorForm from "../components/test-statistics/CalculatorForm";
import ResultPanel from "../components/test-statistics/ResultPanel";

export default function Show({ calculator, form, result }) {
  return (
    <>
      <Head title={calculator.name} />
      <main className="calculator-page">
        <div className="calculator-page__content">
          <Link href="/" className="calculator-page__backlink">
            Back to calculators
          </Link>

          <header className="calculator-page__hero">
            <div>
              <div className="calculator-page__family">{calculator.family}</div>
              <h1 className="calculator-page__title">{calculator.name}</h1>
              <p className="calculator-page__description">
                {calculator.description}
              </p>
            </div>
            <div className="calculator-page__summary-card">
              <h2 className="calculator-page__summary-title">What it checks</h2>
              <p className="calculator-page__summary-copy">{calculator.check}</p>
              <div className="calculator-page__formula-label">
                Statistic / formula
              </div>
              <code className="calculator-page__formula">
                {calculator.statisticFormula}
              </code>
            </div>
          </header>

          <section className="calculator-page__details">
            <article className="calculator-page__detail-card">
              <h2 className="calculator-page__detail-title">Assumptions</h2>
              <ul className="calculator-page__list">
                {calculator.assumptions.map((assumption) => (
                  <li key={assumption}>{assumption}</li>
                ))}
              </ul>
            </article>

            <article className="calculator-page__detail-card">
              <h2 className="calculator-page__detail-title">
                Required sample data
              </h2>
              <ul className="calculator-page__list">
                {calculator.requiredSampleData.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </article>
          </section>

          <section className="calculator-page__workspace">
            <CalculatorForm calculator={calculator} form={form} />
            <ResultPanel result={result} />
          </section>
        </div>
      </main>
    </>
  );
}
