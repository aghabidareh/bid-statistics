import { Head, Link } from "@inertiajs/react";
import SiteHeader from "./components/SiteHeader";

export default function Home({ sections }) {
  return (
    <>
      <Head title="Bid Statistics" />
      <SiteHeader currentPath="/" />
      <main className="home">
        <section className="home__section">
          <div className="home__eyebrow">Bid statistics</div>
          <h1 className="home__title">Statistics online</h1>
          <p className="home__intro">
            Browse the available sections of the statistics toolkit. Test
            statistics keeps the existing 26-calculator workflow, regression
            adds spreadsheet-style modeling, and statistical tables provide Z
            and T critical-value lookups.
          </p>

          <div className="section-grid">
            {sections.map((section) => (
              <article key={section.slug} className="section-card">
                <div className="section-card__eyebrow">Section</div>
                <h2 className="section-card__title">{section.name}</h2>
                <p className="section-card__copy">{section.description}</p>
                <div className="section-card__meta">
                  <div className="section-card__meta-label">Items</div>
                  <div className="section-card__meta-value">{section.itemCount}</div>
                </div>
                <Link href={section.href} className="section-card__link">
                  Open section
                </Link>
              </article>
            ))}
          </div>
        </section>
      </main>
    </>
  );
}
