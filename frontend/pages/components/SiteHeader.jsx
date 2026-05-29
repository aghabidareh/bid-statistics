import { Link } from "@inertiajs/react";

const NAV_ITEMS = [
  { label: "Home", href: "/", key: "/" },
  { label: "Test Statistics", href: "/test-statistics/", key: "/test-statistics" },
  { label: "Regression", href: "/regression/", key: "/regression" },
];

export default function SiteHeader({ currentPath = "/" }) {
  return (
    <header className="site-header">
      <div className="site-header__brand-bar">
        <div className="site-header__brand">Bid Statistics</div>
      </div>
      <nav className="site-header__nav" aria-label="Primary navigation">
        <div className="site-header__nav-inner">
          {NAV_ITEMS.map((item) => {
            const isActive = currentPath === item.key;
            return (
              <Link
                key={`${item.label}-${item.href}`}
                href={item.href}
                className={`site-header__nav-link${isActive ? " site-header__nav-link--active" : ""}`}
              >
                {item.label}
              </Link>
            );
          })}
        </div>
      </nav>
    </header>
  );
}
