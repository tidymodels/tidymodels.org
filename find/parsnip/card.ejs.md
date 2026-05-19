```{=html}
<div class="parsnip-cards list">
<% for (const item of items) { %>
  <div class="parsnip-card">
    <span class="listing-package" hidden><%= item.package %></span>
    <span class="listing-mode" hidden><%= item.mode %></span>
    <span class="listing-model" hidden><%= item.model %></span>
    <span class="listing-engine" hidden><%= item.engine %></span>
    <span class="listing-title" hidden><%= item.title %></span>
    <div class="parsnip-card-title">
      <a href="<%- item.url %>" target="_blank"><%= item.title %></a>
    </div>
    <div class="parsnip-card-meta">
      <span><strong>model:</strong> <code><%= item.model %></code></span>
      <span><strong>engine:</strong> <code><%= item.engine %></code></span>
      <span><strong>mode:</strong> <%= item.mode %></span>
      <span><strong>package:</strong> <%= item.package %></span>
    </div>
  </div>
<% } %>
</div>
```
