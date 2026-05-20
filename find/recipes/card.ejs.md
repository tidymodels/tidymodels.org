```{=html}
<div class="parsnip-cards list">
<% for (const item of items) { %>
  <div class="parsnip-card">
    <span class="listing-package" hidden><%= item.package %></span>
    <span class="listing-func" hidden><%= item.func %></span>
    <span class="listing-title" hidden><%= item.title %></span>
    <div class="parsnip-card-title">
      <a href="<%- item.url %>" target="_blank"><%= item.title %><svg class="icon-inline card-link-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" aria-hidden="true"><!--! Font Awesome Free 6.7.2 --><path d="M320 0c-17.7 0-32 14.3-32 32s14.3 32 32 32h82.7L201.4 265.4c-12.5 12.5-12.5 32.8 0 45.3s32.8 12.5 45.3 0L448 109.3V192c0 17.7 14.3 32 32 32s32-14.3 32-32V32c0-17.7-14.3-32-32-32H320zM80 32C35.8 32 0 67.8 0 112V432c0 44.2 35.8 80 80 80H400c44.2 0 80-35.8 80-80V320c0-17.7-14.3-32-32-32s-32 14.3-32 32V432c0 8.8-7.2 16-16 16H80c-8.8 0-16-7.2-16-16V112c0-8.8 7.2-16 16-16H192c17.7 0 32-14.3 32-32s-14.3-32-32-32H80z"/></svg></a>
    </div>
    <div class="parsnip-card-meta parsnip-card-meta-2">
      <span><strong>function:</strong> <code><%= item.func %></code></span>
      <span><strong>package:</strong> <%= item.package %></span>
    </div>
  </div>
<% } %>
</div>
```
