import { Head, Link } from "@inertiajs/react";
import { useId, useMemo, useState } from "react";
import SiteHeader from "../components/SiteHeader";

const PREVIEW_LEFT = 20;
const PREVIEW_RIGHT = 340;
const PREVIEW_WIDTH = PREVIEW_RIGHT - PREVIEW_LEFT;

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function markerXForValue(value, range) {
  const [min, max] = range;
  const normalized = (value - min) / (max - min || 1);
  return PREVIEW_LEFT + clamp(normalized, 0, 1) * PREVIEW_WIDTH;
}

function pointerValueFromEvent(event, range) {
  const bounds = event.currentTarget.getBoundingClientRect();
  const viewBoxX = ((event.clientX - bounds.left) / bounds.width) * 360;
  const normalized = clamp((viewBoxX - PREVIEW_LEFT) / PREVIEW_WIDTH, 0, 1);
  const [min, max] = range;
  return min + normalized * (max - min);
}

function BellCurvePreview({ label, value, markerValue, range, onPointerValue }) {
  const markerX = markerXForValue(markerValue, range);
  const leftClipId = useId();
  const rightClipId = useId();
  const handlePointerMove = (event) => {
    if (!onPointerValue) {
      return;
    }
    onPointerValue(pointerValueFromEvent(event, range));
  };

  return (
    <div className="stat-table-preview" aria-label={label}>
      <svg
        viewBox="0 0 360 130"
        role="img"
        onPointerMove={handlePointerMove}
        onClick={handlePointerMove}
        className={onPointerValue ? "stat-table-preview__svg stat-table-preview__svg--interactive" : "stat-table-preview__svg"}
      >
        <defs>
          <clipPath id={leftClipId}>
            <rect x="20" y="0" width={Math.max(markerX - 20, 0)} height="110" />
          </clipPath>
          <clipPath id={rightClipId}>
            <rect x={markerX} y="0" width={Math.max(340 - markerX, 0)} height="110" />
          </clipPath>
        </defs>
        <path className="stat-table-preview__grid" d="M20 90H340M60 20V92M120 20V92M180 20V92M240 20V92M300 20V92" />
        <path
          className="stat-table-preview__shade stat-table-preview__shade--left"
          clipPath={`url(#${leftClipId})`}
          d="M20 90 C64 90 78 84 96 66 C116 42 137 17 180 17 C223 17 244 42 264 66 C282 84 296 90 340 90 L340 90 L20 90Z"
        />
        <path
          className="stat-table-preview__shade stat-table-preview__shade--right"
          clipPath={`url(#${rightClipId})`}
          d="M20 90 C64 90 78 84 96 66 C116 42 137 17 180 17 C223 17 244 42 264 66 C282 84 296 90 340 90 L340 90 L20 90Z"
        />
        <path
          className="stat-table-preview__curve"
          d="M20 90 C64 90 78 84 96 66 C116 42 137 17 180 17 C223 17 244 42 264 66 C282 84 296 90 340 90"
        />
        <line className="stat-table-preview__marker" x1={markerX} y1="18" x2={markerX} y2="90" />
        <line className="stat-table-preview__axis" x1="20" y1="90" x2="340" y2="90" />
        <g className="stat-table-preview__legend">
          <rect x="72" y="108" width="26" height="6" className="stat-table-preview__legend-left" />
          <text x="104" y="114">p(x≤z)</text>
          <rect x="152" y="108" width="26" height="6" className="stat-table-preview__legend-right" />
          <text x="184" y="114">p(x&gt;z)</text>
          <line x1="246" y1="111" x2="276" y2="111" className="stat-table-preview__legend-marker" />
          <text x="282" y="114">z</text>
        </g>
      </svg>
      <div className="stat-table-preview__value">{value}</div>
      <div className="stat-table-preview__hint">Hover or click the curve to move the marker.</div>
    </div>
  );
}

function ZTable({ table }) {
  const zCells = useMemo(
    () => table.probability.rows.flatMap((row) => row.cells.map((cell) => ({ ...cell, numericZ: Number(cell.z) }))),
    [table.probability.rows],
  );
  const zRange = useMemo(
    () => [Math.min(...zCells.map((cell) => cell.numericZ)), Math.max(...zCells.map((cell) => cell.numericZ))],
    [zCells],
  );
  const [selected, setSelected] = useState(table.probability.defaultCell);
  const selectedLabel = `P(X ≤ ${selected.z}) = ${selected.value}`;

  const selectNearestZ = (zValue) => {
    const nearestCell = zCells.reduce((nearest, cell) => (
      Math.abs(cell.numericZ - zValue) < Math.abs(nearest.numericZ - zValue) ? cell : nearest
    ));
    setSelected({ z: nearestCell.z, value: nearestCell.value });
  };

  return (
    <>
      <section className="stat-table-hero">
        <div className="stat-table-hero__formula">{selectedLabel}</div>
        <BellCurvePreview
          label="Standard normal cumulative probability"
          value={`z = ${selected.z}`}
          markerValue={Number(selected.z)}
          range={zRange}
          onPointerValue={selectNearestZ}
        />
      </section>

      <section className="stat-table-card">
        <h2 className="stat-table-card__title">Interactive Z table</h2>
        <p className="stat-table-card__copy">
          Hover over a cell or the curve to see its cumulative probability. The left column and top row combine to form the Z score.
        </p>
        <div className="stat-table-scroll stat-table-scroll--large">
          <table className="stat-table stat-table--z">
            <thead>
              <tr>
                <th>Z</th>
                {table.probability.columns.map((column) => (
                  <th key={column}>{column}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.probability.rows.map((row) => (
                <tr key={row.z}>
                  <th>{row.z}</th>
                  {row.cells.map((cell) => (
                    <td
                      key={`${row.z}-${cell.z}`}
                      onMouseEnter={() => setSelected(cell)}
                      className={cell.z === selected.z ? "stat-table__cell--selected" : ""}
                    >
                      {cell.value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="stat-table-card">
        <h2 className="stat-table-card__title">Inverse Z table</h2>
        <p className="stat-table-card__copy">Common one-tail and two-tail critical values for the standard normal distribution.</p>
        <div className="stat-table-scroll stat-table-scroll--compact">
          <table className="stat-table stat-table--inverse">
            <thead>
              <tr>
                {table.inverse.columns.map((column) => (
                  <th key={column}>{column}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.inverse.rows.map((row) => (
                <tr key={row.alpha}>
                  <th>{row.alpha}</th>
                  <td>{row.zAlpha}</td>
                  <td>{row.zOneMinusAlpha}</td>
                  <td>{row.zAlphaOverTwo}</td>
                  <td>{row.zOneMinusAlphaOverTwo}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}

function TTable({ table }) {
  const [selected, setSelected] = useState(table.criticalValues.defaultCell);
  const selectedRow = table.criticalValues.rows.find((row) => row.df === selected.df) || table.criticalValues.rows[0];
  const tRange = useMemo(() => {
    const rowValues = selectedRow.cells.map((cell) => Number(cell.value));
    return [0, Math.max(...rowValues)];
  }, [selectedRow]);
  const selectedLabel = `t(${selected.df}, α=${selected.alpha}) = ${selected.value}`;

  const selectNearestCriticalValue = (criticalValue) => {
    const nearestCell = selectedRow.cells.reduce((nearest, cell) => (
      Math.abs(Number(cell.value) - criticalValue) < Math.abs(Number(nearest.value) - criticalValue) ? cell : nearest
    ));
    setSelected({ ...nearestCell, df: selectedRow.df });
  };

  return (
    <>
      <section className="stat-table-hero">
        <div className="stat-table-hero__formula">{selectedLabel}</div>
        <BellCurvePreview
          label="Student t critical value"
          value={`critical value = ${selected.value}`}
          markerValue={Number(selected.value)}
          range={tRange}
          onPointerValue={selectNearestCriticalValue}
        />
      </section>

      <section className="stat-table-card">
        <h2 className="stat-table-card__title">Interactive T table</h2>
        <p className="stat-table-card__copy">
          Hover over a critical value or the curve to inspect the degrees of freedom and one-tail alpha level. Two-tail alpha values are shown in the second header row.
        </p>
        <div className="stat-table-scroll stat-table-scroll--large">
          <table className="stat-table stat-table--t">
            <thead>
              <tr>
                <th rowSpan="2">df</th>
                <th colSpan={table.criticalValues.oneTailColumns.length}>One-tail α</th>
              </tr>
              <tr>
                {table.criticalValues.oneTailColumns.map((column, index) => (
                  <th key={column}>
                    <div>{column}</div>
                    <div className="stat-table__subhead">two-tail {table.criticalValues.twoTailColumns[index]}</div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.criticalValues.rows.map((row) => (
                <tr key={row.df}>
                  <th>{row.df}</th>
                  {row.cells.map((cell) => {
                    const isSelected = row.df === selected.df && cell.alpha === selected.alpha;
                    return (
                      <td
                        key={`${row.df}-${cell.alpha}`}
                        onMouseEnter={() => setSelected({ ...cell, df: row.df })}
                        className={isSelected ? "stat-table__cell--selected" : ""}
                      >
                        {cell.value}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}

function EducationPanel({ items }) {
  return (
    <section className="stat-table-info">
      {items.map((item) => (
        <article key={item.title} className="stat-table-info__item">
          <h2>{item.title}</h2>
          <p>{item.body}</p>
        </article>
      ))}
    </section>
  );
}

export default function Show({ table, metadata }) {
  const content = useMemo(() => {
    if (table.kind === "z") {
      return <ZTable table={table} />;
    }
    return <TTable table={table} />;
  }, [table]);

  return (
    <>
      <Head title={metadata.name} />
      <SiteHeader currentPath="/statistical-tables" />
      <main className="home">
        <section className="home__section stat-table-page">
          <Link href="/statistical-tables/" className="calculator-page__backlink">
            Back to statistical tables
          </Link>
          <div className="home__eyebrow">Statistical tables</div>
          <h1 className="home__title">{table.title}</h1>
          <p className="home__intro">{table.intro}</p>
          {content}
          <EducationPanel items={table.education} />
        </section>
      </main>
    </>
  );
}
