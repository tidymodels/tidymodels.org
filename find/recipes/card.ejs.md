```{=html}
<div class="parsnip-cards list">
<% for (const item of items) { %>
  <div class="parsnip-card">
    <span class="listing-package" hidden><%= item.package %></span>
    <span class="listing-func" hidden><%= item.func %></span>
    <span class="listing-title" hidden><%= item.title %></span>
    <div class="parsnip-card-title">
      <a href="<%- item.url %>" target="_blank"><%= item.title %></a>
    </div>
    <div class="parsnip-card-meta parsnip-card-meta-2">
      <span><strong>function:</strong> <code><%= item.func %></code></span>
      <span><strong>package:</strong> <%= item.package %></span>
    </div>
  </div>
<% } %>
</div>
```
