import { useEffect, useMemo, useRef, useState } from "react";
import EditableGrid, {
  buildDefaultRole,
  cloneDataset,
  parseDelimitedText,
} from "./EditableGrid";

function hasAlphabeticHeader(row) {
  return row.some((cell) => /[A-Za-z]/.test(cell));
}

function replaceWithImportedMatrix(dataset, matrix, roleOptions) {
  if (matrix.length === 0) {
    return dataset;
  }

  const nextDataset = cloneDataset(dataset);
  const useFirstRowAsHeader = hasAlphabeticHeader(matrix[0]);
  const headerRow = useFirstRowAsHeader ? matrix[0] : null;
  const dataRows = useFirstRowAsHeader ? matrix.slice(1) : matrix;
  const columnCount = Math.max(headerRow?.length || 0, ...dataRows.map((row) => row.length), nextDataset.columns.length);
  const defaultRole = buildDefaultRole(roleOptions);

  nextDataset.columns = Array.from({ length: columnCount }, (_, index) => ({
    key: nextDataset.columns[index]?.key || `column_${Date.now()}_${index + 1}`,
    label:
      headerRow?.[index] || nextDataset.columns[index]?.label || `Column ${index + 1}`,
    role: nextDataset.columns[index]?.role || defaultRole,
  }));

  nextDataset.rows = dataRows.map((row) => ({
    cells: Array.from({ length: columnCount }, (_, index) => row[index] || ""),
  }));

  if (nextDataset.rows.length === 0) {
    nextDataset.rows = [
      { cells: Array.from({ length: columnCount }, () => "") },
      { cells: Array.from({ length: columnCount }, () => "") },
    ];
  }

  nextDataset.sourceMode = "import";
  return nextDataset;
}

export default function RegressionDatasetForm({ calculator, form }) {
  const datasetSchema = calculator.datasetSchema;
  const [dataset, setDataset] = useState(form.values.dataset || datasetSchema.defaultDataset);
  const fileInputRef = useRef(null);

  useEffect(() => {
    setDataset(form.values.dataset || datasetSchema.defaultDataset);
  }, [form.values.dataset, datasetSchema.defaultDataset]);

  const serializedDataset = useMemo(() => JSON.stringify(dataset), [dataset]);

  const applyExample = () => {
    setDataset(datasetSchema.exampleDataset);
  };

  const clearDataset = () => {
    setDataset(datasetSchema.blankDataset);
  };

  const triggerImport = () => {
    fileInputRef.current?.click();
  };

  const handleImport = async (event) => {
    const [file] = event.target.files || [];
    if (!file) {
      return;
    }
    const text = await file.text();
    const matrix = parseDelimitedText(text);
    const importedDataset = replaceWithImportedMatrix(dataset, matrix, datasetSchema.roleOptions);
    importedDataset.filename = file.name;
    setDataset(importedDataset);
    event.target.value = "";
  };

  return (
    <section className="calculator-page__form">
      <div className="calculator-page__panel-header">
        <h2 className="calculator-page__panel-title">Dataset workspace</h2>
        <p className="calculator-page__panel-copy">
          Build or paste a spreadsheet-like dataset for {calculator.name.toLowerCase()}. {datasetSchema.importHint}
        </p>
      </div>

      <form method="post" action={form.action} className="calculator-form regression-form">
        <input type="hidden" name="csrfmiddlewaretoken" value={form.csrfToken} />
        <input type="hidden" name="dataset" value={serializedDataset} />

        <div className="regression-form__actions">
          <button type="submit" className="calculator-form__submit">
            Calculate
          </button>
          <button type="button" className="regression-form__secondary-button" onClick={clearDataset}>
            Clear
          </button>
          <button type="button" className="regression-form__secondary-button" onClick={applyExample}>
            Example
          </button>
          <button type="button" className="regression-form__secondary-button" onClick={triggerImport}>
            Import
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.tsv,.txt"
            className="regression-form__file-input"
            onChange={handleImport}
          />
        </div>

        {dataset.filename ? (
          <div className="regression-form__source">Imported file: {dataset.filename}</div>
        ) : null}

        <EditableGrid
          dataset={dataset}
          onChange={setDataset}
          errors={form.errors}
          roleOptions={datasetSchema.roleOptions}
          allowAddColumns={datasetSchema.allowAddColumns}
        />
      </form>
    </section>
  );
}
