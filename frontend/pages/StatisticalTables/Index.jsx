import { Head, Link } from "@inertiajs/react";
import SiteHeader from "../components/SiteHeader";

export default function Index({ tables }) {
  return (
    <>
      <Head title="Statistical Tables" />
      <SiteHeader currentPath="/statistical-tables" />
      <main className="home">
        <section className="home__section">
          <div className="home__eyebrow">Bid statistics</div>
          <h1 className="home__title">Statistical Tables</h1>
          <p className="home__intro">
            Explore interactive reference tables for common distribution lookups.
            Start with the standard normal Z table or Student&apos;s t critical values.
          </p>

          <div className="section-grid statistical-table-grid">
            {tables.map((table) => (
              <article key={table.slug} className="section-card">
                <div className="section-card__eyebrow">Table</div>
                <h2 className="section-card__title">{table.name}</h2>
                <p className="section-card__copy">{table.description}</p>
                <Link href={table.href} className="section-card__link">
                  Open table
                </Link>
              </article>
            ))}
          </div>
        </section>
      </main>
    </>
  );
}
