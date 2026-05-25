const healthBadge = document.getElementById("healthBadge");
const apiKeyInput = document.getElementById("apiKeyInput");
const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");
const useAllDocsCheckbox = document.getElementById("useAllDocs");
const documentsList = document.getElementById("documentsList");
const refreshDocsBtn = document.getElementById("refreshDocsBtn");
const clearCacheBtn = document.getElementById("clearCacheBtn");
const cacheStatus = document.getElementById("cacheStatus");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const messages = document.getElementById("messages");
const conversationList = document.getElementById("conversationList");
const refreshHistoryBtn = document.getElementById("refreshHistoryBtn");
const toastHost = document.getElementById("toastHost");

const API_KEY_STORAGE = "basicrag_openai_key";

const appState = {
  documents: [],
};

const escapeHtml = (value) =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");

function notify(message, kind = "ok") {
  const toast = document.createElement("div");
  toast.className = `toast ${kind}`;
  toast.textContent = message;
  toastHost.appendChild(toast);

  setTimeout(() => {
    toast.remove();
  }, 3500);
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.detail || "Request failed.");
  }

  return data;
}

function currentApiKey() {
  const key = apiKeyInput.value.trim();
  if (key) {
    localStorage.setItem(API_KEY_STORAGE, key);
  }
  return key;
}

function setUploadStatus(message, kind = "") {
  uploadStatus.textContent = message;
  uploadStatus.classList.remove("ok", "error");
  if (kind) {
    uploadStatus.classList.add(kind);
  }
}

function selectedDocumentIds() {
  if (useAllDocsCheckbox.checked) {
    return [];
  }

  return Array.from(documentsList.querySelectorAll("input[type='checkbox']"))
    .filter((item) => item.checked)
    .map((item) => item.value);
}

function renderDocuments() {
  documentsList.innerHTML = "";

  if (!appState.documents.length) {
    documentsList.innerHTML = '<p class="hint">No indexed documents yet.</p>';
    return;
  }

  for (const doc of appState.documents) {
    const row = document.createElement("label");
    row.className = "doc-item";

    const disabled = useAllDocsCheckbox.checked ? "disabled" : "";

    row.innerHTML = `
      <input type="checkbox" value="${escapeHtml(doc.id)}" ${disabled} />
      <div>
        <div class="doc-name">${escapeHtml(doc.name)}</div>
        <div class="doc-meta">${escapeHtml(doc.file_type.toUpperCase())} · ${doc.chunk_count} chunks</div>
        <div class="doc-meta">Uploaded: ${new Date(doc.uploaded_at).toLocaleString()}</div>
      </div>
    `;

    documentsList.appendChild(row);
  }
}

function appendMessage({ role, text, references = [], cacheHit = false, topic = "" }) {
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;

  if (role === "assistant") {
    const refMarkup = references
      .map(
        (ref) => `
          <div class="reference">
            <div class="reference-title">${escapeHtml(ref.source)} · chunk ${ref.chunk_id}</div>
            <p>${escapeHtml(ref.text)}</p>
          </div>
        `,
      )
      .join("");

    const badgeClass = cacheHit ? "cache" : "fresh";
    const badgeText = cacheHit ? "Cache answer" : "Fresh RAG answer";

    wrapper.innerHTML = `
      <div class="message-header">
        <span class="badge ${badgeClass}">${badgeText}</span>
        ${topic ? `<small>Topic: ${escapeHtml(topic)}</small>` : ""}
      </div>
      <div>${escapeHtml(text).replaceAll("\n", "<br />")}</div>
      ${refMarkup ? `<div class="references">${refMarkup}</div>` : ""}
    `;
  } else {
    wrapper.innerHTML = `<div>${escapeHtml(text).replaceAll("\n", "<br />")}</div>`;
  }

  messages.appendChild(wrapper);
  messages.scrollTop = messages.scrollHeight;
}

function setPendingState(isPending) {
  const askBtn = document.getElementById("askBtn");
  askBtn.disabled = isPending;
  askBtn.textContent = isPending ? "Thinking..." : "Ask";
}

async function refreshHealth() {
  try {
    const data = await fetchJson("/api/health");
    const isReady = data.pinecone_ready;

    healthBadge.classList.remove("ok", "warn");
    healthBadge.classList.add(isReady ? "ok" : "warn");
    healthBadge.textContent = isReady
      ? "Pinecone connected"
      : `Pinecone: ${data.pinecone_status}`;

    cacheStatus.textContent = `Cache entries: ${data.cache_entries}`;
  } catch (error) {
    healthBadge.classList.remove("ok");
    healthBadge.classList.add("warn");
    healthBadge.textContent = "Health check failed";
  }
}

async function refreshDocuments() {
  const data = await fetchJson("/api/documents");
  appState.documents = data.documents;
  renderDocuments();
}

async function refreshConversations() {
  const data = await fetchJson("/api/conversations");
  conversationList.innerHTML = "";

  if (!data.conversations.length) {
    conversationList.innerHTML = '<p class="hint">No conversation history yet.</p>';
    return;
  }

  for (const item of data.conversations) {
    const card = document.createElement("article");
    card.className = "history-item";

    const cacheLabel = item.cache_hit ? " · cache" : "";
    card.innerHTML = `
      <div class="topic">${escapeHtml(item.topic || "general")}</div>
      <div class="question">${escapeHtml(item.question)}</div>
      <div class="meta">${new Date(item.asked_at).toLocaleString()} · ${escapeHtml(item.scope)}${cacheLabel}</div>
    `;

    conversationList.appendChild(card);
  }
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = fileInput.files?.[0];
  const key = currentApiKey();

  if (!key) {
    notify("Enter your OpenAI API key first.", "error");
    return;
  }

  if (!file) {
    notify("Choose a PDF, CSV, or TXT file.", "error");
    return;
  }

  const allowed = [".pdf", ".csv", ".txt"];
  const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (!allowed.includes(extension)) {
    notify("Only PDF, CSV, and TXT are supported.", "error");
    return;
  }

  const payload = new FormData();
  payload.append("file", file);
  payload.append("user_api_key", key);

  setUploadStatus("Uploading and indexing...");

  try {
    const data = await fetchJson("/api/upload", {
      method: "POST",
      body: payload,
    });

    setUploadStatus(
      `${data.name} indexed with ${data.chunk_count} chunks.`,
      "ok",
    );
    fileInput.value = "";

    await Promise.all([refreshDocuments(), refreshHealth()]);
    notify("Document indexed successfully.");
  } catch (error) {
    setUploadStatus(error.message, "error");
    notify(error.message, "error");
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  const key = currentApiKey();

  if (!question) {
    notify("Enter a question.", "error");
    return;
  }

  if (!key) {
    notify("Enter your OpenAI API key first.", "error");
    return;
  }

  appendMessage({ role: "user", text: question });
  questionInput.value = "";
  setPendingState(true);

  try {
    const data = await fetchJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        user_api_key: key,
        document_ids: selectedDocumentIds(),
      }),
    });

    appendMessage({
      role: "assistant",
      text: data.answer,
      references: data.references,
      cacheHit: data.cache_hit,
      topic: data.topic,
    });

    await Promise.all([refreshConversations(), refreshHealth()]);
  } catch (error) {
    appendMessage({
      role: "assistant",
      text: `Error: ${error.message}`,
      references: [],
      cacheHit: false,
    });
    notify(error.message, "error");
  } finally {
    setPendingState(false);
  }
});

useAllDocsCheckbox.addEventListener("change", () => {
  for (const input of documentsList.querySelectorAll("input[type='checkbox']")) {
    input.disabled = useAllDocsCheckbox.checked;
  }
});

clearCacheBtn.addEventListener("click", async () => {
  try {
    const data = await fetchJson("/api/cache/clear", { method: "POST" });
    notify(`${data.message} Removed ${data.cleared_count} entries.`);
    await refreshHealth();
  } catch (error) {
    notify(error.message, "error");
  }
});

refreshDocsBtn.addEventListener("click", async () => {
  try {
    await refreshDocuments();
    notify("Document history refreshed.");
  } catch (error) {
    notify(error.message, "error");
  }
});

refreshHistoryBtn.addEventListener("click", async () => {
  try {
    await refreshConversations();
    notify("Conversation history refreshed.");
  } catch (error) {
    notify(error.message, "error");
  }
});

window.addEventListener("DOMContentLoaded", async () => {
  const savedKey = localStorage.getItem(API_KEY_STORAGE);
  if (savedKey) {
    apiKeyInput.value = savedKey;
  }

  appendMessage({
    role: "assistant",
    text: "Upload a PDF, CSV, or TXT file, then ask questions. You can talk to one document or all documents, and each answer will show source references.",
  });

  try {
    await Promise.all([refreshHealth(), refreshDocuments(), refreshConversations()]);
  } catch (error) {
    notify(error.message, "error");
  }
});
