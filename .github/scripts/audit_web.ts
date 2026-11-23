import { fileURLToPath } from "node:url";

type Package = readonly [
  string,
  string,
  {
    dependencies?: Record<string, string>;
    optionalDependencies?: Record<string, string>;
    peerDependencies?: Record<string, string>;
  },
  ...unknown[],
];
type Lock = {
  workspaces: Record<string, { dependencies?: Record<string, string> }>;
  packages: Record<string, Package>;
};
type Audit = Record<string, { url: string }[]>;

export function validate(lock: Lock, audit: Audit) {
  const active = new Set<string>();
  const visit = (name: string, parent = "") => {
    let key = parent ? `${parent}/${name}` : name;
    while (!(key in lock.packages)) {
      const slash = parent.lastIndexOf("/");
      if (!parent) throw new Error(`Missing locked dependency: ${name}`);
      parent = slash < 0 ? "" : parent.slice(0, slash);
      key = parent ? `${parent}/${name}` : name;
    }
    if (active.has(key)) return;
    active.add(key);
    const metadata = lock.packages[key]![2];
    for (const dependency of Object.keys({
      ...metadata.dependencies,
      ...metadata.optionalDependencies,
      ...metadata.peerDependencies,
    }))
      visit(dependency, key);
  };
  for (const workspace of Object.values(lock.workspaces)) {
    for (const name of Object.keys(workspace.dependencies ?? {})) visit(name);
  }
  const allowed = "https://github.com/advisories/GHSA-fc5q-39xc-vjpf";
  for (const [name, findings] of Object.entries(audit)) {
    const locked = Object.entries(lock.packages).filter(
      ([key]) => key === name || key.endsWith(`/${name}`),
    );
    if (
      name !== "picomatch" ||
      locked.length === 0 ||
      locked.some(
        ([key, entry]) => entry[0] !== "picomatch@4.0.7" || active.has(key),
      )
    ) {
      throw new Error(`Unresolved advisory for ${name}`);
    }
    for (const finding of findings) {
      if (finding.url !== allowed)
        throw new Error(`Unresolved advisory: ${finding.url}`);
    }
  }
}

if (import.meta.main) {
  const cwd = fileURLToPath(new URL("../../web/", import.meta.url));
  const child = Bun.spawn(["bun", "audit", "--json"], {
    cwd,
    stdout: "pipe",
    stderr: "inherit",
  });
  const [status, output] = await Promise.all([
    child.exited,
    new Response(child.stdout).text(),
  ]);
  if (status !== 0 && status !== 1)
    throw new Error(`bun audit exited ${status}`);
  validate(
    Bun.JSONC.parse(await Bun.file(`${cwd}/bun.lock`).text()),
    JSON.parse(output),
  );
}
