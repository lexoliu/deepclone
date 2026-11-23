import { expect, test } from "bun:test";
import { validate } from "./audit_web";

const audit = {
  picomatch: [{ url: "https://github.com/advisories/GHSA-fc5q-39xc-vjpf" }],
};
const lock = {
  workspaces: { "": { dependencies: { react: "19.3.1" } } },
  packages: {
    react: ["react@19.3.1", "", {}] as const,
    picomatch: ["picomatch@4.0.7", "", {}] as const,
  },
};

test("only the approved development advisory is accepted", () => {
  validate(lock, audit);
  expect(() =>
    validate(lock, { picomatch: [{ url: "new-advisory" }] }),
  ).toThrow();
  expect(() => validate(lock, { other: audit.picomatch })).toThrow();
});
test("the exception cannot enter the production dependency graph", () => {
  const production = {
    ...lock,
    packages: {
      ...lock.packages,
      react: [
        "react@19.3.1",
        "",
        { dependencies: { picomatch: "4.0.7" } },
      ] as const,
    },
  };
  expect(() => validate(production, audit)).toThrow();
});
test("version changes require their own review", () => {
  expect(() =>
    validate(
      {
        ...lock,
        packages: {
          ...lock.packages,
          picomatch: ["picomatch@4.0.8", "", {}] as const,
        },
      },
      audit,
    ),
  ).toThrow();
});
