import "@vitejs/plugin-react/preamble";
import "../css/main.css";
import { createInertiaApp } from "@inertiajs/react";
import { createRoot } from "react-dom/client";

const el = document.getElementById("app");
const page = el?.dataset.page ? JSON.parse(el.dataset.page) : null;

createInertiaApp({
  page,
  resolve: (name) => {
    const pages = import.meta.glob("../pages/**/*.jsx");
    return pages[`../pages/${name}.jsx`]();
  },
  setup({ el: appElement, App, props }) {
    createRoot(appElement).render(<App {...props} />);
  },
});
