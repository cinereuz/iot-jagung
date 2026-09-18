// Corn Kernel Mold Detection Frontend Application

document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("file-input");
    const dropZone = document.getElementById("drop-zone");
    const sampleButtonsContainer = document.getElementById("sample-buttons");
    const loadingOverlay = document.getElementById("loading-overlay");
    const resultsSection = document.getElementById("results-section");

    let currentResults = null;

    // Load available samples from server
    async function loadSamples() {
        try {
            const res = await fetch("/api/samples");
            const data = await res.json();
            renderSampleButtons(data.samples);
        } catch (err) {
            console.error("Gagal memuat sampel:", err);
        }
    }

    function renderSampleButtons(samples) {
        sampleButtonsContainer.innerHTML = "";
        
        if (!samples || samples.length === 0) {
            sampleButtonsContainer.innerHTML = "<span style='color: var(--text-muted); font-size: 0.85rem;'>Tidak ada sampel lokal ditemukan.</span>";
            return;
        }

        samples.forEach(s => {
            const btn = document.createElement("button");
            btn.className = `btn-sample ${s.category === "sehat" ? "healthy" : "moldy"}`;
            btn.innerHTML = `<span class="dot ${s.category === "sehat" ? "green" : "red"}"></span> ${s.name}`;
            btn.addEventListener("click", () => detectFromSample(s.path));
            sampleButtonsContainer.appendChild(btn);
        });
    }

    // Drag & drop handlers
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            detectFromUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            detectFromUpload(e.target.files[0]);
        }
    });

    // Detect from uploaded file
    async function detectFromUpload(file) {
        showLoading(true);
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/detect/upload", {
                method: "POST",
                body: formData
            });
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Terjadi kesalahan saat memproses citra.");
            }
            const data = await res.json();
            displayResults(data);
        } catch (err) {
            alert("Error: " + err.message);
        } finally {
            showLoading(false);
        }
    }

    // Detect from server sample path
    async function detectFromSample(samplePath) {
        showLoading(true);
        try {
            const res = await fetch("/api/detect/sample", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sample_path: samplePath })
            });
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Gagal memproses sampel.");
            }
            const data = await res.json();
            displayResults(data);
        } catch (err) {
            alert("Error: " + err.message);
        } finally {
            showLoading(false);
        }
    }

    function showLoading(show) {
        if (show) {
            loadingOverlay.classList.add("active");
        } else {
            loadingOverlay.classList.remove("active");
        }
    }

    // BARU -- Helper kecil terpusat untuk menentukan "state" tiap kernel (biji).
    // Dipakai di renderKernelCards() dan renderFeatureTable() supaya logic
    // penentuan warna/label/teks tidak ditulis dobel di dua tempat.
    // Return: { stateClass, label, colorHex }
    function getKernelState(k) {
        if (k.prediction === "bukan_jagung") {
            // Abu-abu -- menandakan objek ini DIABAIKAN dari penilaian mutu,
            // bukan salah satu dari sehat/terkontaminasi. Warnanya didefinisikan
            // di style.css lewat variabel --notcorn-gray.
            return { stateClass: "notcorn", label: "Bukan Jagung", colorHex: "var(--notcorn-gray)" };
        }
        if (k.is_moldy) {
            return { stateClass: "moldy", label: "Terkontaminasi Jamur", colorHex: "var(--mold-red)" };
        }
        return { stateClass: "healthy", label: "Sehat", colorHex: "var(--healthy-green)" };
    }

    // Render results
    function displayResults(data) {
        currentResults = data;
        const summary = data.summary;
        const images = data.images;
        const kernels = data.kernels;

        // BARU -- summary.bukan_jagung mungkin tidak ada kalau responsenya dari
        // pipeline.py versi lama (backward compatible). Default ke 0 kalau kosong.
        const bukanJagungCount = summary.bukan_jagung || 0;
        // BARU -- total biji yang BENAR-BENAR jagung (dipakai untuk kalimat ringkasan
        // yang lebih akurat, karena summary.total_biji sekarang bisa termasuk
        // objek yang bukan jagung).
        const totalJagung = summary.sehat + summary.terkontaminasi;

        // 1. Status Banner
        const statusBanner = document.getElementById("status-banner");
        // style.css sekarang sudah punya rule .status-banner.secondary untuk kasus
        // "semua objek bukan jagung", jadi cukup ganti class-nya saja seperti
        // badge success/warning/danger yang sudah ada.
        statusBanner.className = `status-banner ${summary.badge}`;
        document.getElementById("status-title").textContent = summary.status_mutu;

        // BARU -- kalimat ringkasan sekarang menyebut jumlah bukan_jagung juga,
        // dan pakai totalJagung (bukan summary.total_biji) supaya persentase
        // kontaminasi yang disebut tetap konsisten dengan angka yang dihitung
        // pipeline (kontaminasi dihitung dari objek yang memang jagung saja).
        let descText = totalJagung > 0
            ? `Dari ${totalJagung} biji jagung yang terdeteksi, ${summary.sehat} biji sehat dan ${summary.terkontaminasi} biji terinfeksi jamur (${summary.persen_kontaminasi}% kontaminasi).`
            : `Tidak ada biji jagung yang terdeteksi pada citra ini.`;
        if (bukanJagungCount > 0) {
            descText += ` ${bukanJagungCount} objek lain terdeteksi bukan biji jagung dan diabaikan dari penilaian mutu.`;
        }
        document.getElementById("status-desc").textContent = descText;

        // 2. Metrics Cards
        document.getElementById("metric-total").textContent = summary.total_biji;
        document.getElementById("metric-healthy").textContent = summary.sehat;
        document.getElementById("metric-moldy").textContent = summary.terkontaminasi;
        document.getElementById("metric-rate").textContent = `${summary.persen_kontaminasi}%`;

        // BARU -- kartu "Bukan Jagung" cuma ditampilkan kalau memang ada objek
        // yang terdeteksi bukan jagung, supaya tidak mengganggu tampilan hasil
        // deteksi normal (yang isinya cuma sehat/terkontaminasi).
        const notCornCard = document.getElementById("metric-card-notcorn");
        if (notCornCard) {
            if (bukanJagungCount > 0) {
                document.getElementById("metric-notcorn").textContent = bukanJagungCount;
                notCornCard.style.display = "";
            } else {
                notCornCard.style.display = "none";
            }
        }

        // 3. Stage Visualizer Setup
        setupStages(images);

        // 4. Kernel Cards Grid
        renderKernelCards(kernels);

        // 5. Feature Table
        renderFeatureTable(kernels);

        // Reveal section and smooth scroll
        resultsSection.style.display = "flex";
        resultsSection.scrollIntoView({ behavior: "smooth" });
    }

    function setupStages(images) {
        const stageImg = document.getElementById("stage-image");
        const stageDesc = document.getElementById("stage-desc");
        const stageButtons = document.querySelectorAll(".stage-btn");

        const stageData = {
            "annotated": {
                img: images.annotated,
                desc: "Hasil akhir klasifikasi Support Vector Machine (SVM): Bounding box hijau menunjukkan biji sehat, merah menunjukkan biji terkontaminasi jamur, dan abu-abu menunjukkan objek yang terdeteksi BUKAN biji jagung (diabaikan dari penilaian mutu)."
            },
            "original": {
                img: images.original,
                desc: "Citra asli biji jagung resolusi tinggi sebelum pra-pemrosesan."
            },
            "gray-blur": {
                img: images.blur,
                desc: "Tahap 1 Preprocessing: Konversi citra ke Grayscale dan reduksi derau (noise reduction) menggunakan Gaussian Blur."
            },
            "otsu": {
                img: images.otsu_mask,
                desc: "Tahap 2A Segmentasi: Pemisahan objek biji dari latar belakang (background) menggunakan Otsu Thresholding dan filter morfologi elips."
            }
        };

        // Set default to annotated
        stageImg.src = stageData["annotated"].img;
        stageDesc.textContent = stageData["annotated"].desc;

        stageButtons.forEach(btn => {
            btn.onclick = () => {
                stageButtons.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const stageKey = btn.dataset.stage;
                if (stageData[stageKey]) {
                    stageImg.src = stageData[stageKey].img;
                    stageDesc.textContent = stageData[stageKey].desc;
                }
            };
        });
    }

    function renderKernelCards(kernels) {
        const container = document.getElementById("kernels-grid");
        container.innerHTML = "";

        kernels.forEach(k => {
            // BARU -- pakai helper getKernelState() alih-alih cuma cek k.is_moldy,
            // supaya ada cabang ketiga untuk "bukan_jagung".
            const state = getKernelState(k);

            const card = document.createElement("div");
            // style.css sekarang sudah punya rule .kernel-card.notcorn (border atas
            // abu-abu), jadi cukup pasang class-nya saja -- sama seperti healthy/moldy.
            card.className = `kernel-card ${state.stateClass}`;

            // BARU -- confidence bisa null untuk bukan_jagung, jadi tampilkan "-"
            // alih-alih literal teks "null%".
            const confidenceText = (k.confidence !== null && k.confidence !== undefined)
                ? `${k.confidence}%`
                : null;

            // BARU -- untuk bukan_jagung, "Area Jamur X%" tidak relevan (itu cuma
            // hasil clustering warna, bukan indikasi jamur beneran) -- diganti
            // dengan info jarak outlier supaya tetap informatif & jujur secara metodologis.
            const thirdThumbLabel = state.stateClass === "notcorn"
                ? `Bukan Jagung (jarak ${k.outlier_distance ?? "-"} vs batas ${k.outlier_threshold ?? "-"})`
                : `Area Jamur (${k.mold_ratio}%)`;

            card.innerHTML = `
                <div class="kernel-header">
                    <span class="kernel-title">Biji #${k.kernel_id}</span>
                    <span class="badge ${state.stateClass}">${state.label}</span>
                </div>
                <div class="kernel-visuals">
                    <div class="thumb-box">
                        <img src="${k.crop_b64}" alt="Crop Asli">
                        <span>Crop Biji</span>
                    </div>
                    <div class="thumb-box">
                        <img src="${k.kmeans_b64 || k.crop_b64}" alt="K-Means Cluster">
                        <span>K-Means (Warna)</span>
                    </div>
                    <div class="thumb-box">
                        <img src="${k.mold_mask_b64 || k.crop_b64}" alt="Mask Jamur">
                        <span>${thirdThumbLabel}</span>
                    </div>
                </div>
                <div class="kernel-features-mini">
                    <div>Contrast: <span class="feature-val">${k.features.contrast}</span></div>
                    <div>Homogeneity: <span class="feature-val">${k.features.homogeneity}</span></div>
                    <div>Energy: <span class="feature-val">${k.features.energy}</span></div>
                    <div>Correlation: <span class="feature-val">${k.features.correlation}</span></div>
                    <div>Hue: <span class="feature-val">${k.features.hue}</span></div>
                    <div>Sat: <span class="feature-val">${k.features.saturation}</span></div>
                </div>
                ${confidenceText === null ? `<div style="font-size:0.78rem;color:${state.colorHex};margin-top:0.5rem;">Confidence tidak berlaku -- objek ditolak sebelum tahap klasifikasi SVM (lihat detail jarak di atas)</div>` : ""}
            `;
            container.appendChild(card);
        });
    }

    function renderFeatureTable(kernels) {
        const tbody = document.querySelector("#feature-table tbody");
        tbody.innerHTML = "";

        kernels.forEach(k => {
            const tr = document.createElement("tr");
            // BARU -- pakai helper yang sama supaya konsisten dengan kartu kernel.
            const state = getKernelState(k);

            // BARU -- confidence null ditampilkan sebagai "-", bukan "null%".
            const confidenceCell = (k.confidence !== null && k.confidence !== undefined)
                ? `${k.confidence}%`
                : "-";

            // BARU -- label status diformat lebih rapi ("BUKAN JAGUNG" bukan "BUKAN_JAGUNG").
            const statusLabel = k.prediction.replace(/_/g, " ").toUpperCase();

            tr.innerHTML = `
                <td>#${k.kernel_id}</td>
                <td style="color: ${state.colorHex}; font-weight: 600;">${statusLabel}</td>
                <td>${confidenceCell}</td>
                <td>${k.mold_ratio}%</td>
                <td>${k.features.contrast}</td>
                <td>${k.features.correlation}</td>
                <td>${k.features.energy}</td>
                <td>${k.features.homogeneity}</td>
                <td>${k.features.hue}</td>
                <td>${k.features.saturation}</td>
                <td>${k.features.value}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    // Export CSV
    const exportBtn = document.getElementById("btn-export-csv");
    if (exportBtn) {
        exportBtn.addEventListener("click", () => {
            if (!currentResults || !currentResults.kernels) return;
            
            let csv = "ID,Status,Confidence(%),AreaJamur(%),Contrast,Correlation,Energy,Homogeneity,Hue,Saturation,Value\n";
            currentResults.kernels.forEach(k => {
                // BARU -- confidence null ditulis string kosong di CSV, bukan literal "null".
                const confVal = (k.confidence !== null && k.confidence !== undefined) ? k.confidence : "";
                csv += `${k.kernel_id},${k.prediction},${confVal},${k.mold_ratio},${k.features.contrast},${k.features.correlation},${k.features.energy},${k.features.homogeneity},${k.features.hue},${k.features.saturation},${k.features.value}\n`;
            });

            const blob = new Blob([csv], { type: "text/csv" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `corn_disease_features_${Date.now()}.csv`;
            a.click();
            URL.revokeObjectURL(url);
        });
    }

    loadSamples();
});