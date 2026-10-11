// ===== Cấu hình =====
// Tự động nhận diện URL: nếu truy cập qua Flask (localhost:5000) thì dùng relative URL "",
// nếu chạy qua Live Server (port 5500) hoặc file:// thì dùng http://localhost:5000
const API_URL = (window.location.protocol.startsWith("http") && window.location.port === "5000")
  ? ""
  : "http://localhost:5000";

// Tên hiển thị cho 5 trường chăm sóc (thứ tự hiển thị)
const CARE_LABELS = {
  anh_sang:  "☀️ Ánh sáng",
  nuoc_tuoi: "💧 Nước tưới",
  dat_trong: "🌱 Đất trồng",
  nhiet_do:  "🌡️ Nhiệt độ",
  luu_y:     "📌 Lưu ý",
};

// ===== Lấy các phần tử trong trang =====
const imageInput        = document.getElementById("image-input");
const previewImg        = document.getElementById("preview");
const predictBtn        = document.getElementById("predict-btn");
const messageSection    = document.getElementById("message-section");
const modelPredSection  = document.getElementById("model-predictions-section");
const modelPredTable    = document.getElementById("model-predictions-table");
const comparisonDiv     = document.getElementById("comparison-table");

// ===== Hàm tiện ích =====
// Chống chèn HTML lạ vào trang khi hiển thị chuỗi từ server
function escapeHTML(text) {
  const div = document.createElement("div");
  div.textContent = String(text);
  return div.innerHTML;
}

function showMessage(text, type) {          // type: "error" | "info"
  messageSection.textContent = text;
  messageSection.className = type;
  messageSection.hidden = false;
}

function clearMessages() {
  messageSection.hidden = true;
  messageSection.textContent = "";
  modelPredSection.hidden = true;
  modelPredTable.innerHTML = "";
  
  // Xóa card kết quả Tab cũ nếu có
  const oldCard = document.querySelector('.result-tabs-card');
  if (oldCard) oldCard.remove();
  
  // Xóa class trạng thái có kết quả
  const gridContainer = document.querySelector('.grid.grid-cols-1');
  if (gridContainer) gridContainer.classList.remove('has-results');
}

// ===== Xem trước ảnh (FR-01) =====
imageInput.addEventListener("change", () => {
  clearMessages();
  const file = imageInput.files[0];
  if (file) {
    previewImg.src = URL.createObjectURL(file);
    previewImg.hidden = false;
  } else {
    previewImg.hidden = true;
  }
});

// ===== Vẽ kết quả nhận diện dạng TAB =====
function renderResults(flowers) {
  const gridContainer = document.querySelector('.grid.grid-cols-1');
  
  // Đánh dấu là đã có kết quả để CSS sắp xếp lại Layout
  gridContainer.classList.add('has-results');

  // Tạo thẻ Card chính bọc toàn bộ kết quả dạng Tab
  const card = document.createElement("div");
  card.className = "glass panel dynamic-flower result-tabs-card";
  
  // 1. Tạo phần Nút bấm (Tabs) cho các loài hoa
  let tabsHTML = `<div class="flower-tabs">`;
  flowers.forEach((f, idx) => {
    tabsHTML += `<button class="tab-btn ${idx === 0 ? 'active' : ''}" data-index="${idx}">${escapeHTML(f.name)}</button>`;
  });
  tabsHTML += `</div>`;

  // 2. Tạo phần Nội dung chi tiết của từng hoa
  let contentHTML = `<div class="tab-content-container">`;
  flowers.forEach((f, idx) => {
    const percent = (f.confidence * 100).toFixed(1);
    const rows = Object.entries(CARE_LABELS).map(([key, label]) =>
      `<tr><th>${label}</th><td>${escapeHTML(f.care_guide[key])}</td></tr>`
    ).join("");

    contentHTML += `
      <div class="tab-pane ${idx === 0 ? 'active' : ''}" id="pane-${idx}">
        <div class="flower-card-inner">
          <h3 style="margin: 0.25rem 0 0.8rem; font-size: 1.35rem; font-weight: 800;">${escapeHTML(f.name)} — ${percent}%</h3>
          <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${percent}%"></div>
          </div>
          <table>${rows}</table>
        </div>
      </div>
    `;
  });
  contentHTML += `</div>`;

  // 3. Gộp tiêu đề "Kết quả nhận diện", hàng nút Tab và khung nội dung
  card.innerHTML = `
    <h2 class="sec-title" style="margin-bottom: 1rem;"><i data-lucide="flower-2"></i> Kết quả nhận diện</h2>
    ${tabsHTML}
    ${contentHTML}
  `;

  // Gắn vào Grid chính
  gridContainer.appendChild(card);
  if (window.lucide) {
    window.lucide.createIcons();
  }

  // 4. Lập trình sự kiện click chuyển Tab mượt mà
  const tabBtns = card.querySelectorAll('.tab-btn');
  const tabPanes = card.querySelectorAll('.tab-pane');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));
      
      btn.classList.add('active');
      const idx = btn.getAttribute('data-index');
      card.querySelector(`#pane-${idx}`).classList.add('active');
    });
  });
}

// ===== Vẽ bảng 5 mô hình dự đoán ảnh vừa tải lên =====
function renderModelPredictions(modelPredictions, flowers) {
  if (!modelPredictions || modelPredictions.length === 0) {
    modelPredSection.hidden = true;
    return;
  }

  // Cột bảng = đúng các loại hoa đã phát hiện, lấy tên hiển thị từ kết quả chính
  const cols = flowers.map(f => ({ id: f.id, name: f.name }));

  const header = `<tr><th>Mô hình</th>${cols.map(c => `<th>${escapeHTML(c.name)}</th>`).join("")}</tr>`;
  const rows = modelPredictions.map(m => {
    const cells = cols.map(c => {
      const p = m.predictions[c.id];
      return `<td>${p !== undefined ? (p * 100).toFixed(1) + "%" : "—"}</td>`;
    }).join("");
    // Đã bỏ chữ (được chọn) theo yêu cầu trước đó
    const label = escapeHTML(m.model_name) + (m.is_selected ? " 🏆" : "");
    return `<tr class="${m.is_selected ? "selected" : ""}"><td>${label}</td>${cells}</tr>`;
  }).join("");

  modelPredTable.innerHTML = `<table>${header}${rows}</table>`;
  modelPredSection.hidden = false;
}

// ===== Gửi ảnh tới POST /predict =====
predictBtn.addEventListener("click", async () => {
  clearMessages();

  // EF-03: chưa chọn ảnh thì không gửi request
  const file = imageInput.files[0];
  if (!file) {
    showMessage("Vui lòng chọn một ảnh trước khi nhận diện.", "error");
    return;
  }

  const formData = new FormData();
  formData.append("image", file);          // tên trường phải là "image"

  predictBtn.disabled = true;
  predictBtn.textContent = "Đang nhận diện...";

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    // Cố đọc JSON; nếu server trả nội dung không phải JSON thì dùng thông báo chung
    let data = null;
    try { data = await response.json(); } catch (_) {}

    if (!response.ok) {
      // EF-01, EF-02, EF-05: server trả lỗi kèm trường "error"
      const msg = (data && data.error) ? data.error : `Lỗi từ máy chủ (mã ${response.status}).`;
      showMessage(msg, "error");
      return;
    }

    if (!data.detected_flowers || data.detected_flowers.length === 0) {
      // FR-08: không lớp nào vượt ngưỡng
      showMessage("Không nhận diện được loại hoa nào trong ảnh này, vui lòng thử ảnh khác.", "info");
      return;
    }

    renderResults(data.detected_flowers);
    renderModelPredictions(data.model_predictions, data.detected_flowers);

  } catch (err) {
    // EF-04: không kết nối được tới backend
    showMessage("Không kết nối được tới máy chủ. Hãy kiểm tra backend đã chạy chưa.", "error");
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Nhận diện";
  }
});

// ===== Bảng so sánh 5 mô hình (FR-11) =====
async function loadComparison() {
  try {
    const response = await fetch(`${API_URL}/models/comparison`);
    if (!response.ok) throw new Error("bad status");
    const data = await response.json();

    const fmt = v => (v * 100).toFixed(2) + "%";
    const rows = data.models.map(m => `
      <tr class="${m.is_selected ? "selected" : ""}">
        <td>${escapeHTML(m.name)}${m.is_selected ? " 🏆" : ""}</td>
        <td>${fmt(m.val_accuracy)}</td>
        <td>${m.val_auc.toFixed(4)}</td>
        <td>${fmt(m.precision)}</td>
        <td>${fmt(m.recall)}</td>
      </tr>`).join("");

    comparisonDiv.innerHTML = `
      <table>
        <tr><th>Mô hình</th><th>Accuracy</th><th>AUC</th><th>Precision</th><th>Recall</th></tr>
        ${rows}
      </table>`;
  } catch (err) {
    // EF-06: không có dữ liệu so sánh thì không hiển thị bảng rỗng
    comparisonDiv.textContent = "Chưa có dữ liệu so sánh mô hình.";
  }
}

loadComparison();   // chạy ngay khi mở trang