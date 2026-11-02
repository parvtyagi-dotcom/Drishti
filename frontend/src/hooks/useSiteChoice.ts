import { useState } from "react";

const KEY = "drishti.site";

function read(): string | null {
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

/** Which site this viewer is looking at, shared by Overview and Site Map and
 * remembered across reloads. Null means the backend's default site. */
export function useSiteChoice(): [string | null, (id: string | null) => void] {
  const [site, setSite] = useState<string | null>(read);
  function choose(id: string | null) {
    setSite(id);
    try {
      if (id) localStorage.setItem(KEY, id);
      else localStorage.removeItem(KEY);
    } catch {
      /* preference only */
    }
  }
  return [site, choose];
}
