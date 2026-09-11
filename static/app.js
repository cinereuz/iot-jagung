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

    // Render results
    function displayResults(data) {
        currentResults = data;
        const summary = data.summary;
        const images = data.images;
        const kernels = data.kernels;

        // 1. Status Banner
        const statusBanner = document.getElementById("status-banner");
        statusBanner.className = `status-banner ${summary.badge}`;
        document.getElementById("status-title").textContent = summary.status_mutu;
        document.getElementById("status-desc").textContent = 
            `Dari total ${summary.total_biji} biji yang terdeteksi, ${summary.sehat} biji sehat dan ${summary.terkontaminasi} biji terinfeksi jamur (${summary.persen_kontaminasi}% kontaminasi).`;

        // 2. Metrics Cards
        document.getElementById("metric-total").textContent = summary.total_biji;
        document.getElementById("metric-healthy").textContent = summary.sehat;
        document.getElementById("metric-moldy").textContent = summary.terkontaminasi;
        document.getElementById("metric-rate").textContent = `${summary.persen_kontaminasi}%`;

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
                desc: "Hasil akhir klasifikasi Support Vector Machine (SVM): Bounding box hijau menunjukkan biji sehat, sedangkan merah menunjukkan biji terkontaminasi jamur beserta persentase confidence."
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
            const card = document.createElement("div");
            card.className = `kernel-card ${k.is_moldy ? "moldy" : "healthy"}`;

            const labelText = k.is_moldy ? "Terkontaminasi Jamur" : "Sehat";
            const badgeClass = k.is_moldy ? "moldy" : "healthy";

            card.innerHTML = `
                <div class="kernel-header">
                    <span class="kernel-title">Biji #${k.kernel_id}</span>
                    <span class="badge ${badgeClass}">${labelText}</span>
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
                        <span>Area Jamur (${k.mold_ratio}%)</span>
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
            `;
            container.appendChild(card);
        });
    }

    function renderFeatureTable(kernels) {
        const tbody = document.querySelector("#feature-table tbody");
        tbody.innerHTML = "";

        kernels.forEach(k => {
            const tr = document.createElement("tr");
            const statusColor = k.is_moldy ? "var(--mold-red)" : "var(--healthy-green)";
            
            tr.innerHTML = `
                <td>#${k.kernel_id}</td>
                <td style="color: ${statusColor}; font-weight: 600;">${k.prediction.toUpperCase()}</td>
                <td>${k.confidence}%</td>
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
                csv += `${k.kernel_id},${k.prediction},${k.confidence},${k.mold_ratio},${k.features.contrast},${k.features.correlation},${k.features.energy},${k.features.homogeneity},${k.features.hue},${k.features.saturation},${k.features.value}\n`;
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
