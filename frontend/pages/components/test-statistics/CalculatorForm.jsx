function renderField(field, fieldValue, errorMessages) {
  const commonProps = {
    id: field.name,
    name: field.name,
    defaultValue: fieldValue,
    placeholder: field.placeholder,
    className: "calculator-form__input",
    required: field.required,
  };

  if (field.kind === "textarea") {
    return (
      <textarea
        {...commonProps}
        rows={field.rows || 5}
        className="calculator-form__input calculator-form__input--textarea"
      />
    );
  }

  if (field.kind === "number") {
    return (
      <input
        {...commonProps}
        type="number"
        step={field.step || "any"}
        min={field.min || undefined}
        max={field.max || undefined}
      />
    );
  }

  if (field.kind === "text") {
    return <input {...commonProps} type="text" />;
  }

  if (field.kind === "select") {
    return (
      <select
        id={field.name}
        name={field.name}
        defaultValue={fieldValue}
        className="calculator-form__input"
      >
        {field.options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    );
  }

  if (field.kind === "radio") {
    return (
      <div className="calculator-form__radio-group">
        {field.options.map((option) => (
          <label key={option.value} className="calculator-form__radio-option">
            <input
              type="radio"
              name={field.name}
              value={option.value}
              defaultChecked={fieldValue === option.value}
            />
            <span>{option.label}</span>
          </label>
        ))}
      </div>
    );
  }

  return null;
}

export default function CalculatorForm({ calculator, form }) {
  return (
    <section className="calculator-page__form">
      <div className="calculator-page__panel-header">
        <h2 className="calculator-page__panel-title">Sample data</h2>
        <p className="calculator-page__panel-copy">
          Enter the input values for {calculator.name.toLowerCase()} and submit
          the form to calculate the result.
        </p>
      </div>

      <form method="post" action={form.action} className="calculator-form">
        <input
          type="hidden"
          name="csrfmiddlewaretoken"
          value={form.csrfToken}
        />

        {calculator.inputFields.map((field) => {
          const errorMessages = form.errors[field.name] || [];
          const fieldValue = form.values[field.name] ?? field.defaultValue ?? "";

          return (
            <div key={field.name} className="calculator-form__field">
              <label htmlFor={field.name} className="calculator-form__label">
                {field.label}
              </label>

              {renderField(field, fieldValue, errorMessages)}

              <p className="calculator-form__help">{field.helpText}</p>

              {errorMessages.length > 0 ? (
                <ul className="calculator-form__errors">
                  {errorMessages.map((message) => (
                    <li key={message}>{message}</li>
                  ))}
                </ul>
              ) : null}
            </div>
          );
        })}

        <button type="submit" className="calculator-form__submit">
          Calculate
        </button>
      </form>
    </section>
  );
}
