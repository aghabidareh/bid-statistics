function renderHeroMetrics(result) {
  const heroMetrics = [];

  if (result.statisticName && result.statistic) {
    heroMetrics.push({
      label: result.statisticName,
      value: result.statistic.display,
    });
  }

  if (result.pValue) {
    heroMetrics.push({
      label: "p-value",
      value: result.pValue.display,
    });
  }

  if (heroMetrics.length === 0) {
    return null;
  }

  return (
    <div className="result-panel__hero-metrics">
      {heroMetrics.map((metric) => (
        <div key={metric.label} className="result-panel__hero-metric">
          <div className="result-panel__hero-label">{metric.label}</div>
          <div className="result-panel__hero-value">{metric.value}</div>
        </div>
      ))}
    </div>
  );
}

function renderMetricCards(metrics) {
  if (metrics.length === 0) {
    return null;
  }

  return (
    <div className="result-panel__metrics">
      {metrics.map((metric) => (
        <div
          key={`${metric.label}-${metric.value}`}
          className={`result-panel__metric${metric.emphasis ? " result-panel__metric--emphasis" : ""}`}
        >
          <div className="result-panel__metric-label">{metric.label}</div>
          <div className="result-panel__metric-value">{metric.value}</div>
        </div>
      ))}
    </div>
  );
}

function renderSections(sections) {
  if (sections.length === 0) {
    return null;
  }

  return sections.map((section) => (
    <section key={section.title} className="result-panel__section">
      <h3 className="result-panel__section-title">{section.title}</h3>
      {renderMetricCards(section.metrics)}
    </section>
  ));
}

function renderTables(tables) {
  if (tables.length === 0) {
    return null;
  }

  return tables.map((table) => (
    <section key={table.title} className="result-panel__section">
      <h3 className="result-panel__section-title">{table.title}</h3>
      <div className="result-panel__table-scroll">
        <table className="result-panel__table">
          <thead>
            <tr>
              {table.columns.map((column) => (
                <th key={column} className="result-panel__table-head">
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, rowIndex) => (
              <tr key={`${table.title}-${rowIndex}`}>
                {row.map((cell, cellIndex) => (
                  <td
                    key={`${table.title}-${rowIndex}-${cellIndex}`}
                    className="result-panel__table-cell"
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {table.caption ? (
        <p className="result-panel__table-caption">{table.caption}</p>
      ) : null}
    </section>
  ));
}

function renderStringList(title, values, modifier) {
  if (values.length === 0) {
    return null;
  }

  return (
    <div className={`result-panel__message-block result-panel__message-block--${modifier}`}>
      <h3 className="result-panel__message-title">{title}</h3>
      <ul className="result-panel__warnings-list">
        {values.map((value) => (
          <li key={value}>{value}</li>
        ))}
      </ul>
    </div>
  );
}

export default function ResultPanel({ result }) {
  if (!result) {
    return (
      <aside className="result-panel">
        <div className="result-panel__empty">
          <h2 className="result-panel__title">Results</h2>
          <p className="result-panel__copy">
            Submit the form to see the primary statistic, p-value, supporting
            metrics, tables, and interpretation.
          </p>
        </div>
      </aside>
    );
  }

  return (
    <aside className="result-panel">
      <div className="result-panel__header">
        <h2 className="result-panel__title">Results</h2>
        <p className="result-panel__copy">{result.interpretation}</p>
      </div>

      {renderHeroMetrics(result)}

      {result.decision ? (
        <div className="result-panel__decision">
          <div className="result-panel__decision-label">
            Decision at α = {result.decision.alpha}
          </div>
          <p className="result-panel__decision-copy">
            {result.decision.conclusion}
          </p>
        </div>
      ) : null}

      {renderMetricCards(result.metrics)}
      {renderSections(result.sections)}
      {renderTables(result.tables)}
      {renderStringList("Notes", result.notes, "notes")}
      {renderStringList("Warnings", result.warnings, "warnings")}
    </aside>
  );
}
