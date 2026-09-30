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
        // yang terdeteksi bukan jagung, supaya tidak mengganggu tampilan normal.
        const notcornCard = document.getElementById("metric-card-notcorn");
        if (bukanJagungCount > 0) {
            notcornCard.style.display = "";
            document.getElementById("metric-notcorn").textContent = bukanJagungCount;
        } else {
            notcornCard.style.display = "none";
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
                    <div>Correlation: <span class="feature-val">${k.features.correlation}</span></div>
                    <div>Energy: <span class="feature-val">${k.features.energy}</span></div>
                    <div>Hue: <span class="feature-val">${k.features.hue}</span></div>
                    <div>Sat: <span class="feature-val">${k.features.saturation}</span></div>
                </div>
                ${confidenceText === null ? `<div style="font-size:0.78rem;color:${state.colorHex};margin-top:0.5rem;">Confidence tidak berlaku -- objek ditolak sebelum tahap klasifikasi SVM (lihat detail jarak di atas)</div>` : ""}
            `;
            container.appendChild(card);
        });
    }

    // "bukan_jagung" -> "Bukan Jagung", "sehat" -> "Sehat" (Title Case sesuai desain)
    function formatStatus(prediction) {
        return String(prediction)
            .split("_")
            .map(w => w.charAt(0).toUpperCase() + w.slice(1))
            .join(" ");
    }

    // confidence null ditampilkan "-", bukan "null%"
    function formatConfidence(k) {
        return (k.confidence !== null && k.confidence !== undefined) ? `${k.confidence}%` : "-";
    }

    // --- State pager (tampilan mobile: 1 biji per halaman) ---
    let pagerKernels = [];
    let pagerIndex = 0;
    const pagerCard = document.getElementById("pager-card");
    const pagerInfo = document.getElementById("pager-info");
    const pagerPrev = document.getElementById("pager-prev");
    const pagerNext = document.getElementById("pager-next");

    function renderPager() {
        const total = pagerKernels.length;
        if (total === 0) {
            pagerCard.innerHTML = `<div class="pager-empty">Belum ada biji terdeteksi.</div>`;
            pagerInfo.textContent = "0 dari 0";
            pagerPrev.disabled = true;
            pagerNext.disabled = true;
            return;
        }

        const k = pagerKernels[pagerIndex];
        const state = getKernelState(k);
        const rows = [
            ["Status SVM", `<span class="st-${state.stateClass}">${formatStatus(k.prediction)}</span>`],
            ["Conf", formatConfidence(k)],
            ["% Jamur", `${k.mold_ratio}%`],
            ["Contrast", k.features.contrast],
            ["Correlation", k.features.correlation],
            ["Energy", k.features.energy],
            ["Homogeneity", k.features.homogeneity],
            ["Hue", k.features.hue],
            ["Saturation", k.features.saturation],
            ["Value", k.features.value]
        ];

        pagerCard.innerHTML =
            `<div class="pager-head">
                <img class="pager-thumb" src="${k.crop_b64}" alt="Crop biji #${k.kernel_id}">
                <span class="pager-id">#${k.kernel_id}</span>
            </div>` +
            rows.map(r => `<div class="pager-row"><span class="pager-label">${r[0]}</span><span class="pager-value">${r[1]}</span></div>`).join("");

        pagerInfo.textContent = `${pagerIndex + 1} dari ${total}`;
        pagerPrev.disabled = pagerIndex === 0;
        pagerNext.disabled = pagerIndex === total - 1;
    }

    pagerPrev.addEventListener("click", () => {
        if (pagerIndex > 0) { pagerIndex--; renderPager(); }
    });
    pagerNext.addEventListener("click", () => {
        if (pagerIndex < pagerKernels.length - 1) { pagerIndex++; renderPager(); }
    });

    function renderFeatureTable(kernels) {
        const tbody = document.querySelector("#feature-table tbody");
        tbody.innerHTML = "";

        kernels.forEach(k => {
            const tr = document.createElement("tr");
            const state = getKernelState(k);

            tr.innerHTML = `
                <td>#${k.kernel_id}</td>
                <td class="st-${state.stateClass}">${formatStatus(k.prediction)}</td>
                <td>${formatConfidence(k)}</td>
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

        // Isi pager mobile dengan data yang sama, mulai dari biji pertama
        pagerKernels = kernels;
        pagerIndex = 0;
        renderPager();
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

    // ============================
    // BARU -- REALTIME WEBCAM DETECTION
    // ============================
    const webcamVideo = document.getElementById("webcam-video");
    const webcamCanvas = document.getElementById("webcam-canvas");
    const realtimeOutput = document.getElementById("realtime-output");
    const realtimeSummary = document.getElementById("realtime-summary");
    const realtimeStatus = document.getElementById("realtime-status");
    const btnStartWebcam = document.getElementById("btn-start-webcam");
    const btnStopWebcam = document.getElementById("btn-stop-webcam");

    let webcamStream = null;       // menyimpan stream kamera aktif, supaya bisa dimatikan nanti
    let realtimeIntervalId = null; // id dari setInterval, supaya bisa dihentikan saat klik stop
    let isProcessingFrame = false; // "kunci" -- cegah kirim frame baru sebelum frame sebelumnya selesai diproses server

    const REALTIME_INTERVAL_MS = 1000; // jeda ambil frame: 1000ms = 1 kali per detik (bisa diubah)

    async function startWebcam() {
        try {
            // { video: true } artinya kita cuma minta akses video, bukan audio/mikrofon.
            // Browser akan menampilkan dialog izin ke user di sini.
            webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
        } catch (err) {
            alert("Tidak bisa mengakses kamera: " + err.message);
            return;
        }

        webcamVideo.srcObject = webcamStream; // sambungkan stream kamera ke elemen <video>

        btnStartWebcam.disabled = true;
        btnStopWebcam.disabled = false;
        realtimeStatus.textContent = "Kamera aktif, memulai deteksi...";
        realtimeStatus.classList.remove("is-live", "is-error");

        // setInterval memanggil captureAndDetectFrame() berulang-ulang tiap
        // REALTIME_INTERVAL_MS milidetik, selama kamera masih menyala.
        realtimeIntervalId = setInterval(captureAndDetectFrame, REALTIME_INTERVAL_MS);
    }

    function stopWebcam() {
        if (realtimeIntervalId) {
            clearInterval(realtimeIntervalId); // hentikan pengambilan frame berikutnya
            realtimeIntervalId = null;
        }
        if (webcamStream) {
            // Matikan tiap "track" (jalur video) di dalam stream -- ini yang benar-benar
            // mematikan lampu indikator kamera di laptop/HP, bukan cuma menyembunyikan videonya.
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        webcamVideo.srcObject = null;

        btnStartWebcam.disabled = false;
        btnStopWebcam.disabled = true;
        realtimeStatus.textContent = "Kamera belum aktif";
        realtimeStatus.classList.remove("is-live", "is-error");
    }

    async function captureAndDetectFrame() {
        // Kalau frame sebelumnya masih diproses server, jangan kirim frame baru dulu --
        // supaya request tidak menumpuk kalau server ternyata lebih lambat dari interval.
        if (isProcessingFrame) return;
        // Video belum siap (ukuran masih 0) -> tunggu interval berikutnya
        if (!webcamVideo.videoWidth) return;

        // Samakan ukuran canvas dengan ukuran asli video, lalu "gambar ulang" frame
        // video saat ini ke dalam canvas -- ini teknik standar untuk "menjepret"
        // 1 frame diam dari video yang sedang berjalan.
        webcamCanvas.width = webcamVideo.videoWidth;
        webcamCanvas.height = webcamVideo.videoHeight;
        const ctx = webcamCanvas.getContext("2d");
        ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);

        // Ubah isi canvas jadi teks base64 JPEG. Kualitas 0.8 (dari maksimal 1.0)
        // dipilih supaya ukuran datanya tidak terlalu besar tapi kualitasnya tetap layak.
        const frameDataUrl = webcamCanvas.toDataURL("image/jpeg", 0.8);

        isProcessingFrame = true;
        try {
            const res = await fetch("/api/detect/frame", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: frameDataUrl })
            });

            if (!res.ok) {
                // Kalau 1 frame gagal diproses, jangan hentikan seluruh sesi realtime --
                // cukup catat di status, lanjut coba lagi di frame berikutnya.
                realtimeStatus.textContent = "Frame ini gagal diproses, mencoba lagi...";
                realtimeStatus.classList.add("is-error");
                return;
            }

            const data = await res.json();
            renderRealtimeResult(data);
        } catch (err) {
            realtimeStatus.textContent = "Koneksi ke server bermasalah...";
            realtimeStatus.classList.add("is-error");
        } finally {
            isProcessingFrame = false;
        }
    }

    function renderRealtimeResult(data) {
        // Kalau kamera sudah dihentikan saat request masih jalan, abaikan hasilnya
        if (!webcamStream) return;
        realtimeOutput.src = data.annotated;

        const s = data.summary;
        realtimeStatus.textContent = `Live -- ${s.total_biji} biji terdeteksi`;
        realtimeStatus.classList.remove("is-error");
        realtimeStatus.classList.add("is-live");

        let text = s.total_biji > 0
            ? `${s.sehat} sehat, ${s.terkontaminasi} terkontaminasi jamur (${s.persen_kontaminasi}%)`
            : "Belum ada biji jagung terdeteksi di frame ini.";
        if (s.bukan_jagung > 0) {
            text += ` -- ${s.bukan_jagung} objek bukan jagung diabaikan.`;
        }
        realtimeSummary.textContent = text;
    }

    btnStartWebcam.addEventListener("click", startWebcam);
    btnStopWebcam.addEventListener("click", stopWebcam);

    // Kalau user pindah/tutup tab tanpa klik "Hentikan Kamera" dulu, pastikan
    // kamera tetap dimatikan supaya lampu indikator kamera tidak terus menyala.
    window.addEventListener("beforeunload", stopWebcam);

    loadSamples();
});