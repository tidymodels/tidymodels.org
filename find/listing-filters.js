// Generic faceted multi-select filters for Quarto custom listings.
//
// Each page that wants filters declares a placeholder element near its
// listing with the listing id and fields to filter on:
//
//   <div class="listing-filter-bar"
//        data-listing-id="parsnip-list"
//        data-fields="model:Model,engine:Engine,mode:Mode,package:Package">
//   </div>
//
// data-fields is a comma-separated list of `field:Label` pairs. The field
// name must match the metadata key in items.yml; "listing-" is prepended
// automatically when looking up values via List.js.

window.addEventListener("load", () => {
  document.querySelectorAll(".listing-filter-bar[data-listing-id]")
    .forEach(setupFilterBar);
});

function setupFilterBar(bar) {
  const listingId = bar.dataset.listingId;
  const list = window["quarto-listings"] &&
    window["quarto-listings"]["listing-" + listingId];
  if (!list) return;

  const fieldDefs = (bar.dataset.fields || "").split(",").map(s => {
    const [field, label] = s.split(":").map(x => x.trim());
    return { field, label: label || field };
  }).filter(d => d.field);

  const fields = fieldDefs.map(d => d.field);
  const state = Object.fromEntries(fields.map(f => [f, new Set()]));
  const controls = {};

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

  function buildMultiSelect({ field, label: labelText }) {
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

  for (const def of fieldDefs) {
    bar.appendChild(buildMultiSelect(def));
  }
}
