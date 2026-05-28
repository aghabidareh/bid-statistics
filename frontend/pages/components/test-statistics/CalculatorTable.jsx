import { Link } from "@inertiajs/react";

export default function CalculatorTable({ calculators }) {
  return (
    <div className="calculator-table">
      <div className="calculator-table__scroll">
        <table className="calculator-table__table">
          <thead>
            <tr className="calculator-table__row calculator-table__row--head">
              <th className="calculator-table__cell calculator-table__cell--head">
                #
              </th>
              <th className="calculator-table__cell calculator-table__cell--head">
                Test name
              </th>
              <th className="calculator-table__cell calculator-table__cell--head">
                Family
              </th>
              <th className="calculator-table__cell calculator-table__cell--head">
                Check
              </th>
              <th className="calculator-table__cell calculator-table__cell--head">
                Statistic / formula
              </th>
              <th className="calculator-table__cell calculator-table__cell--head">
                Required sample data
              </th>
            </tr>
          </thead>
          <tbody>
            {calculators.map((calculator) => (
              <tr key={calculator.slug} className="calculator-table__row">
                <td className="calculator-table__cell calculator-table__cell--index">
                  {calculator.catalogPosition}
                </td>
                <td className="calculator-table__cell calculator-table__cell--name">
                  <Link
                    href={`/test-statistics/${calculator.slug}/`}
                    className="calculator-table__link"
                  >
                    {calculator.name}
                  </Link>
                </td>
                <td className="calculator-table__cell">{calculator.family}</td>
                <td className="calculator-table__cell">{calculator.check}</td>
                <td className="calculator-table__cell calculator-table__cell--formula">
                  <code>{calculator.statisticFormula}</code>
                </td>
                <td className="calculator-table__cell">
                  <ul className="calculator-table__list">
                    {calculator.requiredSampleData.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
