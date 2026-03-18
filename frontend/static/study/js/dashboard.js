// ── DASHBOARD STATE & LOGIC ──

let activeSubjectId = null;
let activeSubjectName = "";
let chatOpen = false;

// current exam session
let examSessionId = null;
let examQuestions = [];
let userAnswers = [];
let examTimerRef = null;
let selectedCategory = ""; // for upload
let currentWeaknessTopics = []; // store for practice

// ── Toggle Chat Widget ──
function toggleChat() {
  const widget = document.getElementById("chatWidget");
  const fabIcon = document.getElementById("chatFabIcon");
  const fabLabel = document.getElementById("chatFabLabel");
  const badge = document.getElementById("chatFabBadge");
  chatOpen = !chatOpen;
  if (chatOpen) {
    widget.classList.remove("chat-widget--hidden");
    widget.classList.add("chat-widget--visible");
    fabIcon.textContent = "💬";
    if (fabLabel) fabLabel.textContent = "Đóng chat";
    if (badge) badge.style.display = "none";
    const body = document.getElementById("chat-messages");
    if (body) body.scrollTop = body.scrollHeight;
  } else {
    widget.classList.remove("chat-widget--visible");
    widget.classList.add("chat-widget--hidden");
    fabIcon.textContent = "🤖";
    if (fabLabel) fabLabel.textContent = "Chat AI";
  }
}

// ── Section navigation ──
function showSection(name, el) {
  document.querySelectorAll("main > section").forEach((s) => (s.style.display = "none"));
  const section = document.getElementById("section-" + name);
  if (section) section.style.display = "block";

  document.querySelectorAll(".nav-item").forEach((n) => n.classList.remove("active"));
  if (el) el.classList.add("active");

  if (name === 'subjects') loadDashboardCharts();
}

// ── Select subject → activate chat ──
function selectSubject(id, name) {
  activeSubjectId = id;
  activeSubjectName = name;
  document.querySelectorAll(".subject-card").forEach((c) => c.classList.remove("selected"));

  // Find the card that was clicked (might be tricky if called via onclick with event)
  if (window.event && window.event.currentTarget) {
    window.event.currentTarget.classList.add("selected");
  }

  const input = document.getElementById("chatInput");
  const btn = document.getElementById("sendBtn");
  if (input) {
    input.disabled = false;
    input.placeholder = `Hỏi AI về môn ${name}...`;
    input.focus();
  }
  if (btn) btn.disabled = false;

  const badge = document.getElementById("chatSubjectBadge");
  if (badge) {
    badge.textContent = name;
    badge.style.display = "inline-block";
  }

  const chatBody = document.getElementById("chat-messages");
  if (chatBody) {
    chatBody.innerHTML = `<div class="chat-bubble ai">Đã chọn môn <strong>${name}</strong>. Tôi sẵn sàng hỗ trợ! 😊</div>`;
  }
  loadAndDisplayHistory(id);
}

// ── Chat history (widget) ──
async function loadAndDisplayHistory(subjectId) {
  try {
    const res = await fetch(`/api/subject/${subjectId}/history/`);
    const data = await res.json();
    const chatBody = document.getElementById("chat-messages");
    if (chatBody && data.history && data.history.length > 0) {
      chatBody.innerHTML = "";
      data.history.forEach((m) => {
        chatBody.innerHTML += `<div class="chat-bubble ${m.role}">
                    <span class="msg-time">${m.time}</span>${m.content}</div>`;
      });
      chatBody.scrollTop = chatBody.scrollHeight;
    }
  } catch (e) {
    console.error(e);
  }
}

// ── Chat history (section) ──
async function loadHistory(subjectId) {
  if (!subjectId) return;
  const container = document.getElementById("history-content");
  if (!container) return;
  container.innerHTML = '<p style="color:var(--text-muted)">Đang tải...</p>';
  try {
    const res = await fetch(`/api/subject/${subjectId}/history/`);
    const data = await res.json();
    if (!data.history.length) {
      container.innerHTML = '<p style="color:var(--text-muted)">Chưa có tin nhắn nào.</p>';
      return;
    }
    container.innerHTML = data.history.map((m) => `
            <div style="margin-bottom:1rem;padding:0.75rem;background:rgba(255,255,255,0.03);border-radius:8px;
              border-left:3px solid ${m.role === "user" ? "var(--primary)" : "var(--secondary)"}">
              <div style="font-size:0.75rem;color:var(--text-muted);margin-bottom:0.3rem">
                ${m.role === "user" ? "👤 Bạn" : "🤖 AI"} · ${m.time}</div>
              <div style="white-space:pre-wrap">${m.content}</div>
            </div>`).join("");
  } catch (e) {
    container.innerHTML = '<p style="color:#fca5a5">Lỗi tải lịch sử.</p>';
  }
}

// ── Chat Messaging ──
function showTyping() {
  const chatBody = document.getElementById("chat-messages");
  if (!chatBody) return;
  const el = document.createElement("div");
  el.id = "typing-indicator";
  el.className = "chat-bubble ai typing";
  el.innerHTML = "<span></span><span></span><span></span>";
  chatBody.appendChild(el);
  chatBody.scrollTop = chatBody.scrollHeight;
}

function hideTyping() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

async function sendMessage() {
  if (!activeSubjectId) return;
  const input = document.getElementById("chatInput");
  const msg = input.value.trim();
  if (!msg) return;

  const chatBody = document.getElementById("chat-messages");
  chatBody.innerHTML += `<div class="chat-bubble user">${msg}</div>`;
  input.value = "";
  chatBody.scrollTop = chatBody.scrollHeight;

  showTyping();
  try {
    const form = new FormData();
    form.append("query", msg);
    form.append("subject_id", activeSubjectId);
    form.append("csrfmiddlewaretoken", csrfToken);
    const res = await fetch("/api/chat/", { method: "POST", body: form });
    const data = await res.json();
    hideTyping();
    chatBody.innerHTML += `<div class="chat-bubble ai">${data.answer || "Có lỗi xảy ra."}</div>`;
    chatBody.scrollTop = chatBody.scrollHeight;
  } catch (e) {
    hideTyping();
    chatBody.innerHTML += `<div class="chat-bubble ai">❌ Lỗi kết nối máy chủ AI.</div>`;
    chatBody.scrollTop = chatBody.scrollHeight;
  }
}

// ── Upload Handlers ──
function setupChips() {
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      selectedCategory = chip.dataset.cat;
    });
  });
}

async function handleUpload(input) {
  if (!input.files[0]) return;
  const subjectId = document.getElementById("uploadSubjectSelect").value;
  if (!subjectId) {
    alert("Vui lòng chọn môn học!");
    return;
  }
  const status = document.getElementById("upload-status");
  status.innerHTML = '<p style="color:var(--text-muted)">⏳ Đang xử lý OCR & phân loại...</p>';

  const form = new FormData();
  form.append("file", input.files[0]);
  form.append("subject_id", subjectId);
  form.append("csrfmiddlewaretoken", csrfToken);
  if (selectedCategory) form.append("category", selectedCategory);

  try {
    const res = await fetch("/api/upload/", { method: "POST", body: form });
    const data = await res.json();
    const catMap = {
      Theory: "📖 Lý thuyết", Slide: "🖥️ Slide", Textbook: "📗 Giáo trình",
      Examples: "💡 Ví dụ", Exercises: "✏️ Bài tập", PastExam: "📜 Đề thi", Other: "📎 Khác"
    };
    const catLabel = catMap[data.category] || data.category;

    if (data.status === "success") {
      status.innerHTML = `<p style="color:var(--secondary)">✅ ${data.message} → <strong>${catLabel}</strong></p>`;
      loadDocuments(subjectId);
    } else {
      status.innerHTML = `<p style="color:#fca5a5">❌ ${data.message}</p>`;
    }
  } catch (e) {
    status.innerHTML = '<p style="color:#fca5a5">❌ Lỗi kết nối.</p>';
  }
  input.value = "";
}

async function loadDocuments(subjectId) {
  if (!subjectId) return;
  const container = document.getElementById("docs-by-category");
  if (!container) return;
  container.innerHTML = '<p style="color:var(--text-muted)">Đang tải...</p>';
  try {
    const res = await fetch(`/api/subject/${subjectId}/docs/`);
    const data = await res.json();
    if (!data.documents.length) {
      container.innerHTML = '<p style="color:var(--text-muted)">Chưa có tài liệu nào.</p>';
      return;
    }
    const cats = {
      Theory: ["📖", "Lý thuyết"], Slide: ["🖥️", "Slide"], Textbook: ["📗", "Giáo trình"],
      Examples: ["💡", "Ví dụ"], Exercises: ["✏️", "Bài tập"], PastExam: ["📜", "Đề thi các năm"], Other: ["📎", "Khác"]
    };
    const grouped = {};
    data.documents.forEach((d) => {
      if (!grouped[d.category]) grouped[d.category] = [];
      grouped[d.category].push(d);
    });
    let html = "";
    for (const [cat, docs] of Object.entries(grouped)) {
      const [icon, label] = cats[cat] || ["📄", cat];
      html += `<div style="margin-bottom:1.2rem">
              <div style="font-weight:600; color:#a78bfa; margin-bottom:0.5rem; font-size:0.9rem">
                ${icon} ${label} <span style="color:var(--text-muted); font-weight:400">(${docs.length})</span></div>
              <div style="display:flex; flex-direction:column; gap:0.3rem">`;
      docs.forEach((d) => {
        html += `<div style="display:flex; justify-content:space-between; align-items:center;
                padding:0.5rem 0.75rem; background:rgba(255,255,255,0.03); border-radius:8px;
                font-size:0.85rem; color:#cbd5e1;">
                <span>📄 ${d.filename}</span>
                <span style="color:var(--text-muted); font-size:0.75rem">${d.uploaded_at}</span>
              </div>`;
      });
      html += "</div></div>";
    }
    container.innerHTML = html;
  } catch (e) {
    container.innerHTML = '<p style="color:#fca5a5">Lỗi tải danh sách.</p>';
  }
}

// ── Exam/Quiz Handlers ──
async function openExam(subjectId, subjectName) {
  const section = document.getElementById("exam-section");
  const content = document.getElementById("exam-content");
  if (!section || !content) return;
  section.style.display = "block";
  document.getElementById("exam-title").textContent = `📝 Đề thi – ${subjectName}`;
  content.innerText = "⏳ Đang sinh đề thi bằng AI...";
  section.scrollIntoView({ behavior: "smooth" });
  try {
    const res = await fetch(`/api/exam/${subjectId}/`);
    const data = await res.json();
    content.innerText = data.exam;
  } catch (e) {
    content.innerText = "❌ Lỗi kết nối máy chủ AI.";
  }
}

function switchQuizTab(tab) {
  const panelNew = document.getElementById("panelNewExam");
  const panelRetake = document.getElementById("panelRetake");
  const tabNew = document.getElementById("tabNewExam");
  const tabRet = document.getElementById("tabRetake");
  if (tab === "new") {
    panelNew.style.display = "";
    panelRetake.style.display = "none";
    tabNew.style.borderBottom = "2px solid #8b5cf6";
    tabNew.style.color = "#a78bfa";
    tabNew.style.fontWeight = "700";
    tabRet.style.borderBottom = "2px solid transparent";
    tabRet.style.color = "var(--text-muted)";
    tabRet.style.fontWeight = "normal";
  } else {
    panelNew.style.display = "none";
    panelRetake.style.display = "";
    tabRet.style.borderBottom = "2px solid #06b6d4";
    tabRet.style.color = "#67e8f9";
    tabRet.style.fontWeight = "700";
    tabNew.style.borderBottom = "2px solid transparent";
    tabNew.style.color = "var(--text-muted)";
    tabNew.style.fontWeight = "normal";
    const subjId = document.getElementById("quizSubjectSelect").value;
    if (subjId) loadUploadedExams(subjId);
  }
}

function onQuizSubjectChange(subjId) {
  loadExamHistory(subjId);
  const panelRetake = document.getElementById("panelRetake");
  if (panelRetake.style.display !== "none") {
    loadUploadedExams(subjId);
  }
}

async function loadUploadedExams(subjId) {
  const container = document.getElementById("uploaded-exams-list");
  if (!container) return;
  if (!subjId) {
    container.innerHTML = '<p style="color:var(--text-muted); font-size:0.9rem">Chọn môn học trước.</p>';
    return;
  }
  container.innerHTML = '<p style="color:var(--text-muted); font-size:0.9rem">⏳ Đang tải...</p>';
  try {
    const res = await fetch(`/api/uploaded-exams/${subjId}/`);
    const data = await res.json();
    const exams = data.exams || [];
    if (!exams.length) {
      container.innerHTML = `<div style="padding:1.2rem; background:rgba(255,255,255,0.03);
              border-radius:12px; border:1px dashed rgba(255,255,255,0.1); text-align:center">
              <div style="font-size:1.5rem; margin-bottom:0.5rem">📂</div>
              <p style="color:var(--text-muted); font-size:0.88rem">Chưa có đề thi nào được upload cho môn này.<br>
              Hãy upload file đề thi (phân loại <strong>📜 Đề thi các năm</strong>) để thi lại.</p>
            </div>`;
      return;
    }
    let html = '<div style="display:flex; flex-direction:column; gap:0.8rem">';
    exams.forEach((e) => {
      const statusColor = e.status === "done" ? "#34d399" : "#fbbf24";
      const statusText = e.status === "done" ? "✅ Sẵn sàng" : "⏳ Đang xử lý";
      html += `<div style="display:flex; align-items:center; justify-content:space-between;
              padding:0.9rem 1.2rem; background:rgba(255,255,255,0.03); border-radius:12px;
              border:1px solid rgba(255,255,255,0.08); gap:1rem">
              <div style="min-width:0; flex:1">
                <div style="font-weight:600; color:#e2e8f0; margin-bottom:0.2rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis">
                  📜 ${e.name}
                </div>
                <div style="font-size:0.78rem; color:var(--text-muted)">${e.num_questions} câu · <span style="color:${statusColor}">${statusText}</span></div>
              </div>
              ${e.status === "done" ?
          `<button onclick="startRetake(${e.id}, '${e.name.replace(/'/g, "\\'")}')"
                style="background:linear-gradient(135deg,#06b6d4,#0891b2); color:#fff; border:none;
                border-radius:10px; padding:0.5rem 1rem; font-size:0.85rem; cursor:pointer;
                font-family:inherit; white-space:nowrap; font-weight:600">🎯 Thi lại</button>` :
          `<span style="color:#fbbf24; font-size:0.8rem">Đang xử lý...</span>`}
            </div>`;
    });
    html += "</div>";
    container.innerHTML = html;
  } catch (e) {
    container.innerHTML = '<p style="color:#fca5a5; font-size:0.88rem">❌ Lỗi tải danh sách đề thi.</p>';
  }
}

async function startRetake(examId, examName) {
  const selEl = document.getElementById("quizSubjectSelect");
  const subjId = selEl.value;
  const subjName = selEl.options[selEl.selectedIndex].dataset.name;
  try {
    const res = await fetch(`/api/exam-session/start/${subjId}/?mode=retake&exam_id=${examId}`);
    const data = await res.json();
    if (data.status !== "success") { alert("❌ " + data.message); return; }
    examSessionId = data.session_id;
    examQuestions = data.questions;
    userAnswers = new Array(examQuestions.length).fill(null);
    renderExamModal(subjName, examQuestions.length, data.exam_source || `📜 Thi lại: ${examName}`);
  } catch (e) { alert("❌ Lỗi kết nối."); }
}

function openQuizSection(subjectId, subjectName) {
  showSection("quiz", null);
  document.querySelectorAll(".nav-item").forEach((n) => n.classList.remove("active"));
  const qs = document.getElementById("quizSubjectSelect");
  if (qs) {
    qs.value = subjectId;
    loadExamHistory(subjectId);
  }
}

async function startQuiz() {
  const selEl = document.getElementById("quizSubjectSelect");
  const subjId = selEl.value;
  const subjName = selEl.options[selEl.selectedIndex].dataset.name;
  const numQ = document.getElementById("quizNumSelect").value;
  const btn = document.getElementById("startQuizBtn");
  btn.disabled = true;
  btn.innerHTML = "⏳ AI đang sinh đề thi...";
  try {
    const res = await fetch(`/api/exam-session/start/${subjId}/?num=${numQ}`);
    const data = await res.json();
    btn.disabled = false;
    btn.innerHTML = "🚀 Bắt đầu thi thử";
    if (data.status !== "success") { alert("❌ " + data.message); return; }
    examSessionId = data.session_id;
    examQuestions = data.questions;
    userAnswers = new Array(examQuestions.length).fill(null);
    renderExamModal(subjName, numQ, data.exam_source || "🤖 AI sinh đề mới");
  } catch (e) {
    btn.disabled = false;
    btn.innerHTML = "🚀 Bắt đầu thi thử";
    alert("❌ Lỗi kết nối. Thử lại sau.");
  }
}

function renderExamModal(subjectName, numQ, examSource) {
  const totalSecs = numQ * 60;
  let remainSecs = totalSecs;
  const modalEl = document.getElementById("examSessionModal");
  modalEl.innerHTML = `
      <div class="exam-modal-overlay" id="examOverlay">
        <div class="exam-modal-card">
          <div class="exam-top-bar">
            <div>
              <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0">🎯 Thi thử – ${subjectName}</div>
              <div style="font-size:0.8rem; color:var(--text-muted)">${examQuestions.length} câu trắc nghiệm</div>
            </div>
            <div class="exam-timer" id="examTimer">⏱ ${formatTime(remainSecs)}</div>
          </div>
          <div style="padding:0.55rem 1rem; background:rgba(6,182,212,0.08); border-left:3px solid #06b6d4; border-radius:0 8px 8px 0; margin-bottom:1.2rem; font-size:0.82rem; color:#67e8f9">
            <strong>📌 Nguồn:</strong> ${examSource || "🤖 AI sinh đề mới"}
          </div>
          <div class="exam-progress-bar">
            <div class="exam-progress-fill" id="progressFill" style="width:0%"></div>
          </div>
          <div id="examQuestionsContainer"></div>
          <div style="display:flex; gap:1rem; margin-top:2rem; justify-content:center">
            <button class="btn-secondary" onclick="confirmCloseExam()">✕ Hủy bài thi</button>
            <button class="btn-primary" id="submitExamBtn" onclick="submitExam()" style="min-width:180px; font-size:1rem">📬 Nộp bài thi</button>
          </div>
        </div>
      </div>`;
  modalEl.style.display = "block";
  renderQuestions();
  if (examTimerRef) clearInterval(examTimerRef);
  examTimerRef = setInterval(() => {
    remainSecs--;
    const timerEl = document.getElementById("examTimer");
    if (timerEl) {
      timerEl.textContent = "⏱ " + formatTime(remainSecs);
      if (remainSecs <= 60) timerEl.classList.add("danger");
      if (remainSecs <= 0) { clearInterval(examTimerRef); submitExam(); }
    }
  }, 1000);
}

function formatTime(secs) {
  const m = String(Math.floor(secs / 60)).padStart(2, "0");
  const s = String(secs % 60).padStart(2, "0");
  return `${m}:${s}`;
}

function renderQuestions() {
  const container = document.getElementById("examQuestionsContainer");
  const labels = ["A", "B", "C", "D"];
  let html = "";
  examQuestions.forEach((q, qi) => {
    html += `<div class="question-block" id="qblock-${qi}">
        <div class="question-num">Câu ${qi + 1} / ${examQuestions.length}</div>
        <div class="question-text">${q.question}</div>
        <div class="options-grid">`;
    q.options.forEach((opt, oi) => {
      html += `<button class="option-btn" id="opt-${qi}-${oi}" onclick="selectAnswer(${qi},${oi})">
          <span class="option-label">${labels[oi]}</span>${opt}</button>`;
    });
    html += `</div></div>`;
  });
  container.innerHTML = html;
}

function selectAnswer(qi, oi) {
  userAnswers[qi] = oi;
  for (let k = 0; k < 4; k++) {
    const btn = document.getElementById(`opt-${qi}-${k}`);
    if (btn) btn.classList.toggle("selected", k === oi);
  }
  const answered = userAnswers.filter((a) => a !== null).length;
  const fill = document.getElementById("progressFill");
  if (fill) fill.style.width = (answered / examQuestions.length) * 100 + "%";
}

async function submitExam() {
  if (examTimerRef) clearInterval(examTimerRef);
  const unanswered = userAnswers.filter((a) => a === null).length;
  if (unanswered > 0) { if (!confirm(`Bạn còn ${unanswered} câu chưa trả lời. Nộp bài ngay?`)) return; }
  const submitBtn = document.getElementById("submitExamBtn");
  if (submitBtn) { submitBtn.disabled = true; submitBtn.innerHTML = "⏳ Đang chấm bài..."; }
  try {
    const res = await fetch(`/api/exam-session/submit/${examSessionId}/`, {
      method: "POST", headers: { "Content-Type": "application/json", "X-CSRFToken": csrfToken },
      body: JSON.stringify({ answers: userAnswers }),
    });
    const data = await res.json();
    if (data.status === "success") renderResults(data); else alert("❌ " + data.message);
  } catch (e) { alert("❌ Lỗi kết nối khi nộp bài."); }
}

function renderResults(data) {
  const labels = ["A", "B", "C", "D"];
  const gradeClass = data.score >= 8.5 ? "grade-A" : data.score >= 7 ? "grade-B" : data.score >= 5 ? "grade-C" : "grade-D";
  const gradeText = data.score >= 8.5 ? "🏆 Xuất sắc" : data.score >= 7 ? "⭐ Giỏi" : data.score >= 5 ? "📘 Đạt" : "❌ Chưa đạt";
  let html = `
      <div class="score-banner">
        <div class="score-big ${gradeClass}">${data.score}/10</div>
        <div class="score-label">${gradeText} · Đúng ${data.correct_count}/${data.total} câu</div>
      </div>`;
  data.result_detail.forEach((q, qi) => {
    const isOk = q.is_correct;
    html += `<div class="question-block ${isOk ? "correct" : "wrong"}">
        <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem">
          <div class="question-num">Câu ${qi + 1}</div>
          <span style="font-weight:700; color:${isOk ? "#34d399" : "#f87171"}">${isOk ? "✅ Đúng" : "❌ Sai"}</span>
        </div>
        <div class="question-text">${q.question}</div>
        <div class="options-grid">`;
    q.options.forEach((opt, oi) => {
      let cls = "";
      if (oi === q.correct_index) cls = "correct-ans"; else if (oi === q.user_answer && !isOk) cls = "wrong-ans";
      const mark = oi === q.correct_index ? " ✓" : oi === q.user_answer && !isOk ? " ✗" : "";
      html += `<div class="option-btn ${cls}" style="cursor:default">
          <span class="option-label">${labels[oi]}</span>${opt}${mark}</div>`;
    });
    html += `</div>`;
    if (q.explanation) {
      const aiTag = q.has_answer === false ? `<span style="font-size:0.72rem; background:rgba(139,92,246,0.15); color:#a78bfa; padding:0.1rem 0.5rem; border-radius:10px; margin-left:0.4rem; font-weight:600">🤖 AI suy luận</span>` : "";
      html += `<div class="explanation-box">💡 <strong>Giải thích:${aiTag}</strong> ${q.explanation}</div>`;
    }
    html += `</div>`;
  });
  const modalEl = document.getElementById("examSessionModal");
  modalEl.innerHTML = `
      <div class="exam-modal-overlay">
        <div class="exam-modal-card">
          <div class="exam-top-bar">
            <div style="font-size:1.2rem; font-weight:800; color:#e2e8f0">📊 Kết quả bài thi</div>
            <button class="btn-secondary" onclick="closeExamModal()" style="padding:0.4rem 1rem">✕ Đóng</button>
          </div>
          ${html}
          <div style="text-align:center; margin-top:2rem">
            <button class="btn-primary" onclick="closeExamModal()" style="min-width:160px">🔄 Thi lại</button>
          </div>
        </div>
      </div>`;
  const quizSel = document.getElementById("quizSubjectSelect");
  if (quizSel) loadExamHistory(quizSel.value);
}

function confirmCloseExam() { if (confirm("Bạn có chắc muốn hủy bài thi? Kết quả sẽ không được lưu.")) closeExamModal(); }
function closeExamModal() {
  if (examTimerRef) clearInterval(examTimerRef);
  const modalEl = document.getElementById("examSessionModal");
  modalEl.style.display = "none";
  modalEl.innerHTML = "";
  examSessionId = null; examQuestions = []; userAnswers = [];
}

async function loadExamHistory(subjectId) {
  if (!subjectId) return;
  const container = document.getElementById("quiz-history-content");
  try {
    const res = await fetch(`/api/exam-session/history/${subjectId}/`);
    const data = await res.json();
    if (!data.history.length) { container.innerHTML = '<p style="color:var(--text-muted); font-size:0.9rem">Chưa có lần thi nào.</p>'; return; }
    let html = `<table class="exam-hist-table">
        <thead><tr><th>#</th><th>Điểm</th><th>Đúng/Tổng</th><th>Ngày thi</th></tr></thead><tbody>`;
    data.history.forEach((h, i) => {
      const sc = h.score;
      const cls = sc >= 8.5 ? "#34d399" : sc >= 7 ? "#60a5fa" : sc >= 5 ? "#fbbf24" : "#f87171";
      html += `<tr>
          <td>${i + 1}</td>
          <td><span class="score-pill" style="background:${cls}22; color:${cls}">${sc}/10</span></td>
          <td>${h.correct_count}/${h.total_questions} câu</td>
          <td>${h.submitted_at}</td>
        </tr>`;
    });
    html += "</tbody></table>";
    container.innerHTML = html;
  } catch (e) { container.innerHTML = '<p style="color:#fca5a5">Lỗi tải lịch sử.</p>'; }
}

// ── Summarization & ML ──
async function loadDocsForSummary() {
  const subjId = document.getElementById("sumSubjectSelect").value;
  const docSelect = document.getElementById("sumDocSelect");
  docSelect.innerHTML = '<option value="">-- Đang tải... --</option>';
  if (!subjId) { docSelect.innerHTML = '<option value="">-- Chọn môn trước --</option>'; return; }
  try {
    const res = await fetch(`/api/subject/${subjId}/docs/`);
    const data = await res.json();
    const docs = data.documents || [];
    if (docs.length === 0) { docSelect.innerHTML = '<option value="">Chưa có tài liệu</option>'; return; }
    docSelect.innerHTML = '<option value="">-- Chọn tài liệu --</option>';
    docs.forEach((d) => {
      const opt = document.createElement("option"); opt.value = d.id;
      opt.textContent = `${d.filename} (${d.category})`;
      docSelect.appendChild(opt);
    });
  } catch (e) { docSelect.innerHTML = '<option value="">Lỗi tải danh sách</option>'; }
}

async function runSummarize() {
  const docId = document.getElementById("sumDocSelect").value;
  const btn = document.getElementById("summarizeBtn");
  const container = document.getElementById("summary-results");
  if (!docId) { alert("Vui lòng chọn tài liệu!"); return; }
  btn.disabled = true; btn.innerHTML = "⏳ Đang tóm tắt (Extractive + Abstractive)...";
  container.innerHTML = '<p style="color:var(--text-muted)">Đang chạy TF-IDF + TextRank và Gemini AI...</p>';
  try {
    const res = await fetch(`/api/summarize/${docId}/`);
    const data = await res.json();
    btn.disabled = false; btn.innerHTML = "🔬 Tóm tắt tài liệu";
    if (data.status === "error") { container.innerHTML = `<p style="color:#fca5a5">❌ ${data.message}</p>`; return; }
    renderSummaryResults(data);
  } catch (e) { btn.disabled = false; btn.innerHTML = "🔬 Tóm tắt tài liệu"; container.innerHTML = '<p style="color:#fca5a5">❌ Lỗi kết nối.</p>'; }
}

function renderSummaryResults(data) {
  const container = document.getElementById("summary-results");
  const ds = data.document_stats; const ext = data.extractive; const abs = data.abstractive;
  let html = `
      <div style="padding:0.8rem 1rem; background:rgba(6,182,212,0.06); border-left:3px solid #06b6d4; border-radius:0 8px 8px 0; margin-bottom:1.5rem; font-size:0.85rem; color:#67e8f9">
        <strong>📄 ${data.document_name}</strong> · ${ds.word_count.toLocaleString()} từ · ${ds.sentence_count} câu · ${ds.tfidf_features} TF-IDF features
      </div>
      <div style="background:rgba(139,92,246,0.05); border:1px solid rgba(139,92,246,0.15); border-radius:14px; padding:1.5rem; margin-bottom:1.5rem">
        <h3 style="color:#a78bfa; margin-bottom:0.3rem">🔬 Extractive Summary (ML)</h3>
        <div style="font-size:0.78rem; color:var(--text-muted); margin-bottom:1rem">Thuật toán: ${ext.algorithm} · Chọn ${ext.num_selected} câu quan trọng nhất</div>
        <div style="color:#e2e8f0; line-height:1.8; font-size:0.92rem; margin-bottom:1.2rem; padding:1rem; background:rgba(255,255,255,0.03); border-radius:8px">${ext.summary}</div>
        <div style="font-size:0.78rem; color:var(--text-muted); margin-bottom:0.5rem">📊 TextRank Scores:</div>`;
  ext.top_sentences.forEach((s) => {
    const barW = Math.min(s.score * 2000, 100);
    html += `
        <div style="margin-bottom:1rem; padding:0.8rem; background:rgba(255,255,255,0.02); border-radius:8px">
          <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#c4b5fd; margin-bottom:0.3rem">
            <span>Câu ${s.original_index + 1}</span><span>Score: ${s.score}</span>
          </div>
          <div style="height:6px; background:rgba(255,255,255,0.05); border-radius:3px; overflow:hidden; margin-bottom:0.5rem">
            <div style="height:100%; width:${barW}%; background:linear-gradient(90deg,#8b5cf6,#06b6d4); border-radius:3px"></div>
          </div>
          <div style="font-size:0.82rem; color:#94a3b8; line-height:1.5; font-style:italic">"${s.sentence}"</div>
        </div>`;
  });
  html += `</div>
      <div style="background:rgba(6,182,212,0.05); border:1px solid rgba(6,182,212,0.15); border-radius:14px; padding:1.5rem; margin-bottom:1.5rem">
        <h3 style="color:#67e8f9; margin-bottom:0.3rem">🤖 Abstractive Summary (Gemini AI)</h3>
        <div style="font-size:0.78rem; color:var(--text-muted); margin-bottom:1rem">Model: ${abs.algorithm}${abs.truncated_input ? " · ⚠️ Input đã được cắt ngắn (>8000 ký tự)" : ""}</div>
        <div style="color:#e2e8f0; line-height:1.8; font-size:0.92rem; padding:1rem; background:rgba(255,255,255,0.03); border-radius:8px; white-space:pre-line">${abs.summary}</div>
      </div>
      <div style="padding:1.5rem; background:linear-gradient(135deg,rgba(139,92,246,0.08),rgba(6,182,212,0.08)); border-radius:14px; border:1px solid rgba(139,92,246,0.15); text-align:center">
        <div style="font-size:1.5rem; margin-bottom:0.5rem">⚖️</div><div style="color:#e2e8f0; font-weight:600; margin-bottom:0.5rem">So sánh 2 phương pháp</div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; text-align:left; font-size:0.85rem; color:#94a3b8">
          <div><div style="color:#a78bfa; font-weight:600; margin-bottom:0.3rem">🔬 Extractive</div><ul style="margin:0; padding-left:1.2rem; line-height:1.6"><li>Trích xuất câu nguyên gốc</li><li>Dùng TF-IDF + Graph ranking</li><li>Không cần AI, chạy offline</li></ul></div>
          <div><div style="color:#67e8f9; font-weight:600; margin-bottom:0.3rem">🤖 Abstractive</div><ul style="margin:0; padding-left:1.2rem; line-height:1.6"><li>Viết lại bằng ngôn ngữ mới</li><li>Dùng Gemini LLM</li><li>Tự nhiên, có cấu trúc hơn</li></ul></div>
        </div>
      </div>`;
  container.innerHTML = html;
}

async function runAnalysis() {
  const subjId = document.getElementById("mlSubjectSelect").value;
  const btn = document.getElementById("analyzeBtn");
  const container = document.getElementById("ml-results");
  btn.disabled = true; btn.innerHTML = "⏳ Đang phân tích bằng ML...";
  container.innerHTML = '<p style="color:var(--text-muted)">Đang thu thập dữ liệu và chạy K-Means Clustering...</p>';
  try {
    const url = subjId ? `/api/weakness/${subjId}/` : "/api/weakness/";
    const res = await fetch(url);
    const data = await res.json();
    btn.disabled = false; btn.innerHTML = "🔬 Bắt đầu phân tích";
    if (data.status === "no_data" || data.status === "insufficient") {
      container.innerHTML = `<div style="text-align:center; padding:2rem; color:var(--text-muted)"><div style="font-size:3rem; margin-bottom:1rem">📭</div><p>${data.message}</p></div>`;
      return;
    }
    if (data.status === "error") { container.innerHTML = `<p style="color:#fca5a5">❌ ${data.message}</p>`; return; }
    renderMLResults(data);
  } catch (e) { btn.disabled = false; btn.innerHTML = "🔬 Bắt đầu phân tích"; container.innerHTML = '<p style="color:#fca5a5">❌ Lỗi kết nối.</p>'; }
}

function renderMLResults(data) {
  const container = document.getElementById("ml-results");
  const s = data.summary; const ml = data.ml_info;
  const sevColors = { high: "#f87171", medium: "#fbbf24", low: "#34d399" };
  const sevLabels = { high: "🔴 Yếu nghiêm trọng", medium: "🟡 Cần cải thiện", low: "🟢 Tương đối ổn" };
  let html = `
      <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:1rem; margin-bottom:2rem">
        <div style="text-align:center; padding:1rem; background:rgba(139,92,246,0.1); border-radius:12px; border:1px solid rgba(139,92,246,0.2)">
          <div style="font-size:2rem; font-weight:800; color:#a78bfa">${s.total_exams}</div><div style="font-size:0.78rem; color:var(--text-muted)">Lần thi</div>
        </div>
        <div style="text-align:center; padding:1rem; background:rgba(52,211,153,0.1); border-radius:12px; border:1px solid rgba(52,211,153,0.2)">
          <div style="font-size:2rem; font-weight:800; color:#34d399">${s.accuracy}%</div><div style="font-size:0.78rem; color:var(--text-muted)">Tỷ lệ đúng</div>
        </div>
        <div style="text-align:center; padding:1rem; background:rgba(248,113,113,0.1); border-radius:12px; border:1px solid rgba(248,113,113,0.2)">
          <div style="font-size:2rem; font-weight:800; color:#f87171">${s.total_wrong}</div><div style="font-size:0.78rem; color:var(--text-muted)">Câu sai</div>
        </div>
        <div style="text-align:center; padding:1rem; background:rgba(6,182,212,0.1); border-radius:12px; border:1px solid rgba(6,182,212,0.2)">
          <div style="font-size:2rem; font-weight:800; color:#06b6d4">${s.avg_score}</div><div style="font-size:0.78rem; color:var(--text-muted)">Điểm TB</div>
        </div>
      </div>
      <div style="padding:0.8rem 1rem; background:rgba(139,92,246,0.06); border-left:3px solid #8b5cf6; border-radius:0 8px 8px 0; margin-bottom:2rem; font-size:0.85rem; color:#c4b5fd">
        <strong>🤖 Thuật toán:</strong> ${ml.algorithm} · <strong>Clusters:</strong> ${ml.n_clusters} · <strong>Silhouette Score:</strong> ${ml.silhouette_score}
      </div>
      <h3 style="color:#e2e8f0; margin-bottom:1.2rem">📊 Các chủ đề yếu (xếp theo mức độ nghiêm trọng)</h3>`;
  currentWeaknessTopics = data.weakness_topics || [];
  data.weakness_topics.forEach((topic, i) => {
    const color = sevColors[topic.severity]; const label = sevLabels[topic.severity];
    const keywords = topic.topic_keywords.map((k) => `<span style="background:rgba(139,92,246,0.15); color:#c4b5fd; padding:0.2rem 0.6rem; border-radius:12px; font-size:0.78rem; display:inline-block; margin:0.15rem">${k}</span>`).join("");
    html += `
        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-left:4px solid ${color}; border-radius:0 14px 14px 0; padding:1.5rem; margin-bottom:1.2rem">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.8rem"><span style="font-weight:700; color:#e2e8f0; font-size:1.1rem"> Nhóm ${i + 1}: ${topic.topic_name}</span><span style="font-size:0.82rem; color:${color}; font-weight:600">${label}</span></div>
          <div style="height:8px; background:rgba(255,255,255,0.07); border-radius:4px; margin-bottom:1rem; overflow:hidden"><div style="height:100%; width:${topic.percentage}%; background:${color}; border-radius:4px; transition:width 0.5s ease"></div></div>
          <div style="font-size:0.78rem; color:var(--text-muted); margin-bottom:0.8rem">${topic.wrong_count} câu sai (${topic.percentage}% tổng)</div>
          <div style="margin-bottom:1rem"><div style="font-size:0.78rem; color:var(--text-muted); margin-bottom:0.4rem">📌 Từ khóa liên quan:</div>${keywords}</div>
          <div style="font-size:0.78rem; color:var(--text-muted); margin-bottom:0.4rem">📝 Mẫu câu sai:</div>`;
    topic.sample_questions.forEach((sq) => {
      html += `<div style="background:rgba(248,113,113,0.05); border:1px solid rgba(248,113,113,0.15); border-radius:8px; padding:0.8rem; margin-bottom:0.5rem; font-size:0.85rem">
            <div style="color:#e2e8f0; margin-bottom:0.4rem"><strong>Q:</strong> ${sq.question}</div>
            <div style="color:#f87171">❌ Bạn trả lời: ${sq.user_answer}</div><div style="color:#34d399">✅ Đáp án đúng: ${sq.correct_answer}</div>
            ${sq.explanation ? `<div style="color:#67e8f9; margin-top:0.3rem; font-style:italic">💡 ${sq.explanation}</div>` : ""}
          </div>`;
    });
    html += `
          <button onclick="practiceWeakTopic(${i}, event)" class="btn-primary" style="margin-top:1.5rem; width:100%; background:linear-gradient(135deg,#06b6d4,#0891b2); font-weight:700; padding:0.8rem">
            📝 Luyện tập khắc phục chủ đề này ngay
          </button>
        </div>`;
  });
  html += `<div style="margin-top:2rem; padding:1.5rem; background:linear-gradient(135deg,rgba(6,182,212,0.1),rgba(139,92,246,0.08)); border-radius:14px; border:1px solid rgba(6,182,212,0.2); text-align:center">
        <div style="font-size:1.5rem; margin-bottom:0.5rem">💡</div><div style="color:#e2e8f0; font-weight:600; margin-bottom:0.5rem">Gợi ý ôn tập</div>
        <div style="color:#94a3b8; font-size:0.9rem; line-height:1.6">Hãy tập trung ôn lại <strong style="color:#f87171">${data.weakness_topics[0]?.topic_name || "Chủ đề 1"}</strong> vì đây là chủ đề bạn yếu nhất. Click vào nút "Luyện tập" để AI sinh đề trắc nghiệm giúp bạn khắc phục lỗ hổng!</div>
      </div>`;
  container.innerHTML = html;
}

async function practiceWeakTopic(index, event) {
  const topic = currentWeaknessTopics[index];
  if (!topic) return;
  const subjId = document.getElementById("mlSubjectSelect").value;
  if (!subjId) {
    alert("Vui lòng CHỌN MỘT MÔN HỌC CỤ THỂ (không chọn 'Tất cả các môn') ở combobox bên trên để sinh bài luyện tập bộ 5 câu phần này.");
    return;
  }

  const btn = event.currentTarget;
  const originText = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = "⏳ AI đang phân tích và sinh đề luyện tập...";

  try {
    const res = await fetch(`/api/weakness/practice/${subjId}/`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRFToken": csrfToken },
      body: JSON.stringify({
        topic_name: topic.topic_name,
        questions: topic.cluster_questions
      })
    });
    const data = await res.json();
    btn.disabled = false;
    btn.innerHTML = originText;

    if (data.status !== "success") { alert("❌ " + data.message); return; }

    // Select the subject name from the dropdown for the title
    const selEl = document.getElementById("mlSubjectSelect");
    const subjName = selEl.options[selEl.selectedIndex].text.replace(/^[^\w\s]+\s*/, ''); // strip icon

    examSessionId = data.session_id;
    examQuestions = data.questions;
    userAnswers = new Array(examQuestions.length).fill(null);
    renderExamModal(subjName, examQuestions.length, data.exam_source);
  } catch (e) {
    btn.disabled = false;
    btn.innerHTML = originText;
    alert("❌ Lỗi kết nối. Thử lại sau.");
  }
}

// ── Subject management ──
function openAddSubjectModal() {
  const modal = document.getElementById("addSubjectModal");
  if (modal) modal.style.display = "flex";
}
function closeModal() {
  const modal = document.getElementById("addSubjectModal");
  if (modal) modal.style.display = "none";
}
async function submitAddSubject(e) {
  e.preventDefault();
  const form = document.getElementById("addSubjectForm");
  const data = new FormData(form);
  data.append("csrfmiddlewaretoken", csrfToken);
  try {
    const res = await fetch("/api/subject/add/", { method: "POST", body: data });
    const resp = await res.json();
    if (resp.status === "success") { closeModal(); location.reload(); } else alert("Lỗi: " + JSON.stringify(resp.message));
  } catch (e) { alert("Lỗi kết nối."); }
}
async function deleteSubject(id, btn) {
  if (!confirm("Bạn có chắc muốn xóa môn học này?")) return;
  try {
    const res = await fetch(`/api/subject/${id}/delete/`, { method: "POST", headers: { "X-CSRFToken": csrfToken } });
    const data = await res.json();
    if (data.status === "success" && btn) btn.closest(".subject-card").remove();
  } catch (e) { alert("Lỗi xóa môn học."); }
}

// ── Dashboard Charts ──
let _chartInstances = {};
const CHART_COLORS = ['#8b5cf6', '#06b6d4', '#34d399', '#f59e0b', '#f87171', '#a78bfa', '#67e8f9', '#fbbf24'];

function destroyChart(id) { if (_chartInstances[id]) { _chartInstances[id].destroy(); delete _chartInstances[id]; } }

async function loadDashboardCharts() {
  try {
    const res = await fetch('/api/dashboard/stats/');
    const d = await res.json();
    renderStatsCards(d.overall);
    renderActivityChart(d.activity_days);
    renderScoreChart(d.score_trend);
    renderDocsChart(d.docs_by_category);
    renderSubjectChart(d.subject_avg_scores);
  } catch (e) { console.error('Dashboard stats error:', e); }
}

function renderStatsCards(overall) {
  const ses = document.getElementById('stat-sessions');
  const avg = document.getElementById('stat-avg');
  const bst = document.getElementById('stat-best');
  const act = document.getElementById('stat-activedays');
  if (ses) ses.textContent = overall.total_sessions || 0;
  if (avg) avg.textContent = overall.avg_score ? overall.avg_score + '/10' : '–';
  if (bst) bst.textContent = overall.best_score ? overall.best_score + '/10' : '–';
  if (act) act.textContent = overall.active_days_14 + ' ngày';
}

function renderActivityChart(days) {
  const hasAny = days.some(d => d.count > 0);
  const emptyEl = document.getElementById('activityEmpty');
  const canvasEl = document.getElementById('activityChart');
  if (!hasAny) { if (emptyEl) emptyEl.style.display = ''; if (canvasEl) canvasEl.style.display = 'none'; return; }
  if (emptyEl) emptyEl.style.display = 'none'; if (canvasEl) canvasEl.style.display = '';
  destroyChart('activity');
  _chartInstances['activity'] = new Chart(canvasEl, {
    type: 'bar', data: {
      labels: days.map(d => d.label),
      datasets: [{
        label: 'Hoạt động', data: days.map(d => d.count),
        backgroundColor: days.map(d => d.count > 0 ? 'rgba(139,92,246,0.7)' : 'rgba(255,255,255,0.05)'),
        borderRadius: 6, borderSkipped: false,
      }],
    },
    options: {
      responsive: true, plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ctx.raw + ' hoạt động' } } },
      scales: { x: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { display: false } }, y: { ticks: { color: '#64748b', stepSize: 1 }, grid: { color: 'rgba(255,255,255,0.04)' }, beginAtZero: true } },
    },
  });
}

function renderScoreChart(scoreTrend) {
  const subjects = Object.keys(scoreTrend);
  const emptyEl = document.getElementById('scoreEmpty');
  const canvasEl = document.getElementById('scoreChart');
  if (!subjects.length) { if (emptyEl) emptyEl.style.display = ''; if (canvasEl) canvasEl.style.display = 'none'; return; }
  if (emptyEl) emptyEl.style.display = 'none'; if (canvasEl) canvasEl.style.display = '';
  const allDates = [...new Set(subjects.flatMap(s => scoreTrend[s].map(p => p.date)))];
  destroyChart('score');
  _chartInstances['score'] = new Chart(canvasEl, {
    type: 'line', data: {
      labels: allDates,
      datasets: subjects.map((subj, i) => ({
        label: subj, data: allDates.map(date => { const pt = scoreTrend[subj].find(p => p.date === date); return pt ? pt.score : null; }),
        borderColor: CHART_COLORS[i % CHART_COLORS.length], backgroundColor: CHART_COLORS[i % CHART_COLORS.length] + '22', tension: 0.35, pointRadius: 5, pointHoverRadius: 7, spanGaps: true,
      })),
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } }, tooltip: { callbacks: { label: ctx => ctx.dataset.label + ': ' + ctx.raw + '/10' } } },
      scales: { x: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } }, y: { ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.04)' }, min: 0, max: 10 } },
    },
  });
}

function renderDocsChart(docsCats) {
  const emptyEl = document.getElementById('docsEmpty');
  const canvasEl = document.getElementById('docsChart');
  if (!docsCats.length) { if (emptyEl) emptyEl.style.display = ''; if (canvasEl) canvasEl.style.display = 'none'; return; }
  if (emptyEl) emptyEl.style.display = 'none'; if (canvasEl) canvasEl.style.display = '';
  destroyChart('docs');
  _chartInstances['docs'] = new Chart(canvasEl, {
    type: 'doughnut', data: {
      labels: docsCats.map(d => d.category),
      datasets: [{ data: docsCats.map(d => d.count), backgroundColor: CHART_COLORS.slice(0, docsCats.length), borderColor: 'rgba(0,0,0,0.3)', borderWidth: 2, hoverOffset: 8 }],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8', padding: 10, font: { size: 11 } } }, tooltip: { callbacks: { label: ctx => ctx.label + ': ' + ctx.raw + ' tài liệu' } } },
      cutout: '62%',
    },
  });
}

function renderSubjectChart(subjectAvgs) {
  const emptyEl = document.getElementById('subjectEmpty');
  const canvasEl = document.getElementById('subjectChart');
  if (!subjectAvgs.length) { if (emptyEl) emptyEl.style.display = ''; if (canvasEl) canvasEl.style.display = 'none'; return; }
  if (emptyEl) emptyEl.style.display = 'none'; if (canvasEl) canvasEl.style.display = '';
  destroyChart('subject');
  _chartInstances['subject'] = new Chart(canvasEl, {
    type: 'bar', data: {
      labels: subjectAvgs.map(s => s.subject),
      datasets: [{ label: 'Điểm TB', data: subjectAvgs.map(s => s.avg), backgroundColor: subjectAvgs.map((s, i) => CHART_COLORS[i % CHART_COLORS.length] + 'cc'), borderRadius: 8, borderSkipped: false }],
    },
    options: {
      indexAxis: 'y', responsive: true,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => `Điểm TB: ${ctx.raw}/10 (${subjectAvgs[ctx.dataIndex].total} lần thi)` } } },
      scales: { x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.04)' }, min: 0, max: 10 }, y: { ticks: { color: '#94a3b8', font: { size: 12 } }, grid: { display: false } } },
    },
  });
}

// ── Initialization ──
window.addEventListener('load', () => {
  loadDashboardCharts();
  setupChips();
  const qs = document.getElementById("quizSubjectSelect");
  if (qs && qs.value) loadExamHistory(qs.value);
});
