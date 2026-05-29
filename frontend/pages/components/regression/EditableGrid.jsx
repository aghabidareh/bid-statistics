function cloneDataset(dataset) {
  return {
    ...dataset,
    columns: dataset.columns.map((column) => ({ ...column })),
    rows: dataset.rows.map((row) => ({ ...row, cells: [...row.cells] })),
  };
}

function getFieldErrors(errors, path) {
  return errors[path] || [];
}

function buildEmptyRow(columnCount) {
  return { cells: Array.from({ length: columnCount }, () => "") };
}

function buildDefaultRole(roleOptions) {
  return roleOptions.find((option) => option.value === "predictor")?.value || roleOptions[0]?.value || "predictor";
}

function parseDelimitedText(text) {
  const normalized = text.replace(/\r\n/g, "\n").trim();
  if (!normalized) {
    return [];
  }
  return normalized.split("\n").map((line) => {
    const delimiter = line.includes("\t") ? "\t" : ",";
    return line.split(delimiter).map((cell) => cell.trim());
  });
}

function applyMatrix(dataset, matrix, startRow, startColumn, allowAddColumns, roleOptions) {
  if (matrix.length === 0) {
    return dataset;
  }

  const nextDataset = cloneDataset(dataset);
  const maxColumnsNeeded = startColumn + Math.max(...matrix.map((row) => row.length));
  if (allowAddColumns) {
    while (nextDataset.columns.length < maxColumnsNeeded) {
      const columnNumber = nextDataset.columns.length + 1;
      nextDataset.columns.push({
        key: `column_${Date.now()}_${columnNumber}`,
        label: `Column ${columnNumber}`,
        role: buildDefaultRole(roleOptions),
      });
      nextDataset.rows = nextDataset.rows.map((row) => ({
        ...row,
        cells: [...row.cells, ""],
      }));
    }
  }

  const columnLimit = nextDataset.columns.length;
  while (nextDataset.rows.length < startRow + matrix.length) {
    nextDataset.rows.push(buildEmptyRow(columnLimit));
  }

  matrix.forEach((matrixRow, rowOffset) => {
    matrixRow.forEach((value, columnOffset) => {
      const targetColumn = startColumn + columnOffset;
      if (targetColumn < columnLimit) {
        nextDataset.rows[startRow + rowOffset].cells[targetColumn] = value;
      }
    });
  });

  return nextDataset;
}

export default function EditableGrid({
  dataset,
  onChange,
  errors,
  roleOptions,
  allowAddColumns,
}) {
  const updateColumnLabel = (columnIndex, value) => {
    const nextDataset = cloneDataset(dataset);
    nextDataset.columns[columnIndex].label = value;
    onChange(nextDataset);
  };

  const updateColumnRole = (columnIndex, value) => {
    const nextDataset = cloneDataset(dataset);
    nextDataset.columns[columnIndex].role = value;
    onChange(nextDataset);
  };

  const updateCell = (rowIndex, columnIndex, value) => {
    const nextDataset = cloneDataset(dataset);
    nextDataset.rows[rowIndex].cells[columnIndex] = value;
    onChange(nextDataset);
  };

  const addRow = () => {
    const nextDataset = cloneDataset(dataset);
    nextDataset.rows.push(buildEmptyRow(nextDataset.columns.length));
    onChange(nextDataset);
  };

  const removeRow = () => {
    if (dataset.rows.length <= 1) {
      return;
    }
    const nextDataset = cloneDataset(dataset);
    nextDataset.rows.pop();
    onChange(nextDataset);
  };

  const addColumn = () => {
    const nextDataset = cloneDataset(dataset);
    const columnNumber = nextDataset.columns.length + 1;
    nextDataset.columns.push({
      key: `column_${Date.now()}_${columnNumber}`,
      label: `Column ${columnNumber}`,
      role: buildDefaultRole(roleOptions),
    });
    nextDataset.rows = nextDataset.rows.map((row) => ({
      ...row,
      cells: [...row.cells, ""],
    }));
    onChange(nextDataset);
  };

  const removeColumn = () => {
    if (!allowAddColumns || dataset.columns.length <= 2) {
      return;
    }
    const nextDataset = cloneDataset(dataset);
    nextDataset.columns.pop();
    nextDataset.rows = nextDataset.rows.map((row) => ({
      ...row,
      cells: row.cells.slice(0, -1),
    }));
    onChange(nextDataset);
  };

  const handlePaste = (rowIndex, columnIndex, event) => {
    const matrix = parseDelimitedText(event.clipboardData.getData("text"));
    if (matrix.length <= 1 && matrix[0]?.length <= 1) {
      return;
    }
    event.preventDefault();
    onChange(applyMatrix(dataset, matrix, rowIndex, columnIndex, allowAddColumns, roleOptions));
  };

  return (
    <div className="editable-grid">
      <div className="editable-grid__toolbar">
        <div className="editable-grid__toolbar-copy">
          Paste tabular data straight into the grid, or import a CSV/TSV file.
        </div>
        <div className="editable-grid__toolbar-actions">
          <button type="button" className="editable-grid__button" onClick={addRow}>
            Add row
          </button>
          <button type="button" className="editable-grid__button" onClick={removeRow}>
            Remove row
          </button>
          {allowAddColumns ? (
            <>
              <button type="button" className="editable-grid__button" onClick={addColumn}>
                Add column
              </button>
              <button type="button" className="editable-grid__button" onClick={removeColumn}>
                Remove column
              </button>
            </>
          ) : null}
        </div>
      </div>

      {getFieldErrors(errors, "dataset.columns").length > 0 ? (
        <ul className="calculator-form__errors editable-grid__errors">
          {getFieldErrors(errors, "dataset.columns").map((message) => (
            <li key={message}>{message}</li>
          ))}
        </ul>
      ) : null}

      {getFieldErrors(errors, "dataset.rows").length > 0 ? (
        <ul className="calculator-form__errors editable-grid__errors">
          {getFieldErrors(errors, "dataset.rows").map((message) => (
            <li key={message}>{message}</li>
          ))}
        </ul>
      ) : null}

      <div className="editable-grid__scroll">
        <table className="editable-grid__table">
          <thead>
            <tr>
              <th className="editable-grid__head editable-grid__head--index">#</th>
              {dataset.columns.map((column, columnIndex) => {
                const labelErrors = getFieldErrors(errors, `dataset.columns.${columnIndex}.label`);
                const roleErrors = getFieldErrors(errors, `dataset.columns.${columnIndex}.role`);
                return (
                  <th key={column.key || columnIndex} className="editable-grid__head">
                    <div className="editable-grid__column-header">
                      <input
                        type="text"
                        value={column.label}
                        onChange={(event) => updateColumnLabel(columnIndex, event.target.value)}
                        className={`editable-grid__column-input${labelErrors.length > 0 ? " editable-grid__column-input--error" : ""}`}
                        placeholder={`Column ${columnIndex + 1}`}
                      />
                      <select
                        value={column.role}
                        onChange={(event) => updateColumnRole(columnIndex, event.target.value)}
                        className={`editable-grid__role-select${roleErrors.length > 0 ? " editable-grid__role-select--error" : ""}`}
                      >
                        {roleOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                      {labelErrors.concat(roleErrors).length > 0 ? (
                        <ul className="calculator-form__errors editable-grid__column-errors">
                          {labelErrors.concat(roleErrors).map((message) => (
                            <li key={message}>{message}</li>
                          ))}
                        </ul>
                      ) : null}
                    </div>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {dataset.rows.map((row, rowIndex) => (
              <tr key={`row-${rowIndex}`} className="editable-grid__row">
                <td className="editable-grid__index">{rowIndex + 1}</td>
                {row.cells.map((cell, columnIndex) => {
                  const cellErrors = getFieldErrors(errors, `dataset.rows.${rowIndex}.cells.${columnIndex}`);
                  return (
                    <td key={`cell-${rowIndex}-${columnIndex}`} className="editable-grid__cell">
                      <input
                        type="text"
                        value={cell}
                        onChange={(event) => updateCell(rowIndex, columnIndex, event.target.value)}
                        onPaste={(event) => handlePaste(rowIndex, columnIndex, event)}
                        className={`editable-grid__cell-input${cellErrors.length > 0 ? " editable-grid__cell-input--error" : ""}`}
                      />
                      {cellErrors.length > 0 ? (
                        <ul className="calculator-form__errors editable-grid__cell-errors">
                          {cellErrors.map((message) => (
                            <li key={message}>{message}</li>
                          ))}
                        </ul>
                      ) : null}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export { applyMatrix, buildDefaultRole, cloneDataset, parseDelimitedText };
