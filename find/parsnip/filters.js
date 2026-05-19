window.addEventListener("load", () => {
  const list = window["quarto-listings"] && window["quarto-listings"]["listing-parsnip-list"];
  if (!list) return;

  const fields = ["model", "engine", "mode", "package"];
  const state = Object.fromEntries(fields.map(f => [f, new Set()]));
  const controls = {}; // field -> { summary, label, checkboxes: Map(value -> {input, label}) }

  function itemPasses(item, skipField) {
    for (const f of fields) {
      if (f === skipField) continue;
      const s = state[f];
      if (s.size === 0) continue;
      if (!s.has(item.values()["listing-" + f])) return false;
    }
    return true;
  }

  function applyFilter() {
    const any = fields.some(f => state[f].size > 0);
    if (!any) {
      list.filter();
    } else {
      list.filter(item => fields.every(f =>
        state[f].size === 0 || state[f].has(item.values()["listing-" + f])
      ));
    }
    refreshOptions();
  }

  function refreshOptions() {
    for (const f of fields) {
      const key = "listing-" + f;
      const allowed = new Set();
      for (const item of list.items) {
        if (itemPasses(item, f)) allowed.add(item.values()[key]);
      }
      const ctrl = controls[f];
      for (const [v, { input, label }] of ctrl.checkboxes) {
        const ok = allowed.has(v) || input.checked;
        label.style.display = ok ? "" : "none";
      }
    }
  }

  function buildMultiSelect(field, labelText) {
    const key = "listing-" + field;
    const values = Array.from(new Set(list.items.map(it => it.values()[key])))
      .filter(v => v && v !== ".na.character")
      .sort();

    const wrap = document.createElement("div");
    wrap.className = "listing-filter";

    const summary = document.createElement("button");
    summary.type = "button";
    summary.className = "listing-filter-summary";
    summary.textContent = labelText + ": All";

    const panel = document.createElement("div");
    panel.className = "listing-filter-panel";
    panel.hidden = true;

    const checkboxes = new Map();
    for (const v of values) {
      const lab = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = v;
      cb.addEventListener("change", () => {
        if (cb.checked) state[field].add(v);
        else state[field].delete(v);
        const picked = values.filter(x => state[field].has(x));
        summary.textContent =
          labelText + ": " + (picked.length === 0 ? "All" : picked.join(", "));
        applyFilter();
      });
      lab.appendChild(cb);
      lab.appendChild(document.createTextNode(" " + v));
      panel.appendChild(lab);
      checkboxes.set(v, { input: cb, label: lab });
    }

    summary.addEventListener("click", () => {
      panel.hidden = !panel.hidden;
    });
    document.addEventListener("click", e => {
      if (!wrap.contains(e.target)) panel.hidden = true;
    });

    wrap.appendChild(summary);
    wrap.appendChild(panel);
    controls[field] = { summary, checkboxes };
    return wrap;
  }

  const bar = document.createElement("div");
  bar.className = "listing-filter-bar";
  for (const [f, label] of [["model", "Model"], ["engine", "Engine"], ["mode", "Mode"], ["package", "Package"]]) {
    bar.appendChild(buildMultiSelect(f, label));
  }

  const listing = document.getElementById("listing-parsnip-list");
  if (listing && listing.parentNode) listing.parentNode.insertBefore(bar, listing);
});
