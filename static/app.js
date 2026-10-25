// Corn Kernel Mold Detection Frontend Application

document.addEventListener('DOMContentLoaded', () => {
  const fileInput = document.getElementById('file-input');
  const dropZone = document.getElementById('drop-zone');
  const sampleButtonsContainer = document.getElementById('sample-buttons');
  const loadingOverlay = document.getElementById('loading-overlay');
  const resultsSection = document.getElementById('results-section');

  let currentResults = null;

  // Load available samples from server
  async function loadSamples() {
    if (!sampleButtonsContainer) return;
    try {
      const res = await fetch('/api/samples');
      const data = await res.json();
      renderSampleButtons(data.samples);
    } catch (err) {
      console.error('Gagal memuat sampel:', err);
    }
  }

  function renderSampleButtons(samples) {
    if (!sampleButtonsContainer) return;
    sampleButtonsContainer.innerHTML = '';

    if (!samples || samples.length === 0) {
      sampleButtonsContainer.innerHTML = "<span style='color: var(--text-muted); font-size: 0.85rem;'>No local samples found.</span>";
      return;
    }

    samples.forEach((s) => {
      const btn = document.createElement('button');
      btn.className = `btn-sample ${s.category === 'sehat' ? 'healthy' : 'moldy'}`;
      btn.innerHTML = `<span class="dot ${s.category === 'sehat' ? 'green' : 'red'}"></span> ${s.name}`;
      btn.addEventListener('click', () => detectFromSample(s.path));
      sampleButtonsContainer.appendChild(btn);
    });
  }

  // Drag & drop handlers
  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      detectFromUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      detectFromUpload(e.target.files[0]);
    }
  });

  // Detect from uploaded file
  async function detectFromUpload(file) {
    showLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/detect/upload', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'An error occurred while processing the image.');
      }
      const data = await res.json();
      displayResults(data);
    } catch (err) {
      alert('Error: ' + err.message);
    } finally {
      showLoading(false);
    }
  }

  // Detect from server sample path
  async function detectFromSample(samplePath) {
    showLoading(true);
    try {
      const res = await fetch('/api/detect/sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sample_path: samplePath }),
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Failed to process the sample.');
      }
      const data = await res.json();
      displayResults(data);
    } catch (err) {
      alert('Error: ' + err.message);
    } finally {
      showLoading(false);
    }
  }

  function showLoading(show) {
    if (show) {
      loadingOverlay.classList.add('active');
    } else {
      loadingOverlay.classList.remove('active');
    }
  }

  // BARU -- Helper kecil terpusat untuk menentukan "state" tiap kernel (biji).
  // Dipakai di renderKernelCards() dan renderFeatureTable() supaya logic
  // penentuan warna/label/teks tidak ditulis dobel di dua tempat.
  // Return: { stateClass, label, colorHex }
  function getKernelState(k) {
    if (k.prediction === 'bukan_jagung') {
      // Abu-abu -- menandakan objek ini DIABAIKAN dari penilaian mutu,
      // bukan salah satu dari sehat/terkontaminasi. Warnanya didefinisikan
      // di style.css lewat variabel --notcorn-gray.
      return { stateClass: 'notcorn', label: 'Not Corn', colorHex: 'var(--notcorn-gray)' };
    }
    if (k.is_moldy) {
      return { stateClass: 'moldy', label: 'Mold Contaminated', colorHex: 'var(--mold-red)' };
    }
    return { stateClass: 'healthy', label: 'Healthy', colorHex: 'var(--healthy-green)' };
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
    const statusBanner = document.getElementById('status-banner');
    // style.css sekarang sudah punya rule .status-banner.secondary untuk kasus
    // "semua objek bukan jagung", jadi cukup ganti class-nya saja seperti
    // badge success/warning/danger yang sudah ada.
    statusBanner.className = `status-banner ${summary.badge}`;
    document.getElementById('status-title').textContent = summary.status_mutu;

    // BARU -- kalimat ringkasan sekarang menyebut jumlah bukan_jagung juga,
    // dan pakai totalJagung (bukan summary.total_biji) supaya persentase
    // kontaminasi yang disebut tetap konsisten dengan angka yang dihitung
    // pipeline (kontaminasi dihitung dari objek yang memang jagung saja).
    let descText = '';
    if (totalJagung === 0) {
      descText = 'No corn kernels were detected in this image.';
      if (bukanJagungCount > 0) {
        descText += ` Objects detected as NOT corn: ${bukanJagungCount}.`;
      }
    } else if (bukanJagungCount > totalJagung) {
      descText = `Not Corn: detected ${bukanJagungCount} non-corn object(s) vs only ${totalJagung} corn-like object(s) (${summary.sehat} healthy, ${summary.terkontaminasi} mold-infected). Because non-corn objects predominate, the overall result is classified as Not Corn.`;
    } else {
      descText = `Corn kernels detected: ${totalJagung} (${summary.sehat} healthy, ${summary.terkontaminasi} mold-infected). Contamination rate: ${summary.persen_kontaminasi}%.`;
      if (bukanJagungCount > 0) {
        descText += ` Other objects not recognized as corn kernels: ${bukanJagungCount} (excluded from the quality assessment).`;
      }
    }
    document.getElementById('status-desc').textContent = descText;

    // 2. Metrics Cards
    document.getElementById('metric-total').textContent = summary.total_biji;
    document.getElementById('metric-healthy').textContent = summary.sehat;
    document.getElementById('metric-moldy').textContent = summary.terkontaminasi;
    document.getElementById('metric-rate').textContent = `${summary.persen_kontaminasi}%`;

    // BARU -- kartu "Bukan Jagung" cuma ditampilkan kalau memang ada objek
    // yang terdeteksi bukan jagung, supaya tidak mengganggu tampilan normal.
    const notcornCard = document.getElementById('metric-card-notcorn');
    if (bukanJagungCount > 0) {
      notcornCard.style.display = '';
      document.getElementById('metric-notcorn').textContent = bukanJagungCount;
    } else {
      notcornCard.style.display = 'none';
    }

    // 3. Stage Visualizer Setup
    setupStages(images);

    // 4. Kernel Cards Grid
    renderKernelCards(kernels);

    // 5. Feature Table
    renderFeatureTable(kernels);

    // Reveal section and smooth scroll
    resultsSection.style.display = 'flex';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
  }

  function setupStages(images) {
    const stageImg = document.getElementById('stage-image');
    const stageDesc = document.getElementById('stage-desc');
    const stageButtons = document.querySelectorAll('.stage-btn');

    const stageData = {
      annotated: {
        img: images.annotated,
        desc: 'Final classification result of the Support Vector Machine (SVM): green bounding boxes indicate healthy kernels, red indicates mold-contaminated kernels, and gray indicates objects detected as NOT corn kernels (excluded from the quality assessment).',
      },
      original: {
        img: images.original,
        desc: 'Original high-resolution corn kernel image before preprocessing.',
      },
      'gray-blur': {
        img: images.blur,
        desc: 'Preprocessing Step 1: Conversion of the image to grayscale and noise reduction using Gaussian Blur.',
      },
      otsu: {
        img: images.otsu_mask,
        desc: 'Segmentation Step 2A: Separation of kernel objects from the background using Otsu Thresholding and an elliptical morphological filter.',
      },
    };

    // Set default to annotated
    stageImg.src = stageData['annotated'].img;
    stageDesc.textContent = stageData['annotated'].desc;

    stageButtons.forEach((btn) => {
      btn.onclick = () => {
        stageButtons.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const stageKey = btn.dataset.stage;
        if (stageData[stageKey]) {
          stageImg.src = stageData[stageKey].img;
          stageDesc.textContent = stageData[stageKey].desc;
        }
      };
    });
  }

  function renderKernelCards(kernels) {
    const container = document.getElementById('kernels-grid');
    container.innerHTML = '';

    kernels.forEach((k) => {
      // BARU -- pakai helper getKernelState() alih-alih cuma cek k.is_moldy,
      // supaya ada cabang ketiga untuk "bukan_jagung".
      const state = getKernelState(k);

      const card = document.createElement('div');
      // style.css sekarang sudah punya rule .kernel-card.notcorn (border atas
      // abu-abu), jadi cukup pasang class-nya saja -- sama seperti healthy/moldy.
      card.className = `kernel-card ${state.stateClass}`;

      // BARU -- confidence bisa null untuk bukan_jagung, jadi tampilkan "-"
      // alih-alih literal teks "null%".
      const confidenceText = k.confidence !== null && k.confidence !== undefined ? `${k.confidence}%` : null;

      // BARU -- untuk bukan_jagung, "Area Jamur X%" tidak relevan (itu cuma
      // hasil clustering warna, bukan indikasi jamur beneran) -- diganti
      // dengan info jarak outlier supaya tetap informatif & jujur secara metodologis.
      const thirdThumbLabel = state.stateClass === 'notcorn' ? `Not Corn (distance ${k.outlier_distance ?? '-'} vs threshold ${k.outlier_threshold ?? '-'})` : `Mold Area (${k.mold_ratio}%)`;

      card.innerHTML = `
                <div class="kernel-header">
                    <span class="kernel-title">Kernel #${k.kernel_id}</span>
                    <span class="badge ${state.stateClass}">${state.label}</span>
                </div>
                <div class="kernel-visuals">
                    <div class="thumb-box">
                        <img src="${k.crop_b64}" alt="Original Crop">
                        <span>Kernel Crop</span>
                    </div>
                    <div class="thumb-box">
                        <img src="${k.kmeans_b64 || k.crop_b64}" alt="K-Means Cluster">
                        <span>K-Means (Color)</span>
                    </div>
                    <div class="thumb-box">
                        <img src="${k.mold_mask_b64 || k.crop_b64}" alt="Mold Mask">
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
                ${confidenceText === null ? `<div style="font-size:0.78rem;color:${state.colorHex};margin-top:0.5rem;">Confidence not applicable -- object rejected before the SVM classification stage (see the distance details above)</div>` : ''}
            `;
      container.appendChild(card);
    });
  }

  // Peta nilai "prediction" dari backend -> label Inggris untuk tampilan.
  // KEY (sehat / terkontaminasi / bukan_jagung) JANGAN diubah: itu nilai yang dikirim
  // pipeline.py dan dipakai logika lain. Yang diterjemahkan hanya teks yang tampil.
  const PREDICTION_LABELS = {
    sehat: 'Healthy',
    terkontaminasi: 'Contaminated',
    bukan_jagung: 'Not Corn',
  };
  function formatStatus(prediction) {
    return PREDICTION_LABELS[prediction] || String(prediction);
  }

  // confidence null ditampilkan "-", bukan "null%"
  function formatConfidence(k) {
    return k.confidence !== null && k.confidence !== undefined ? `${k.confidence}%` : '-';
  }

  // --- State pager (tampilan mobile: 1 biji per halaman) ---
  let pagerKernels = [];
  let pagerIndex = 0;
  const pagerCard = document.getElementById('pager-card');
  const pagerInfo = document.getElementById('pager-info');
  const pagerPrev = document.getElementById('pager-prev');
  const pagerNext = document.getElementById('pager-next');

  function renderPager() {
    const total = pagerKernels.length;
    if (total === 0) {
      pagerCard.innerHTML = `<div class="pager-empty">No kernels detected yet.</div>`;
      pagerInfo.textContent = '0 of 0';
      pagerPrev.disabled = true;
      pagerNext.disabled = true;
      return;
    }

    const k = pagerKernels[pagerIndex];
    const state = getKernelState(k);
    const rows = [
      ['SVM Status', `<span class="st-${state.stateClass}">${formatStatus(k.prediction)}</span>`],
      ['Conf', formatConfidence(k)],
      ['% Mold', `${k.mold_ratio}%`],
      ['Contrast', k.features.contrast],
      ['Correlation', k.features.correlation],
      ['Energy', k.features.energy],
      ['Homogeneity', k.features.homogeneity],
      ['Hue', k.features.hue],
      ['Saturation', k.features.saturation],
      ['Value', k.features.value],
    ];

    pagerCard.innerHTML =
      `<div class="pager-head">
                <img class="pager-thumb" src="${k.crop_b64}" alt="Kernel crop #${k.kernel_id}">
                <span class="pager-id">#${k.kernel_id}</span>
            </div>` + rows.map((r) => `<div class="pager-row"><span class="pager-label">${r[0]}</span><span class="pager-value">${r[1]}</span></div>`).join('');

    pagerInfo.textContent = `${pagerIndex + 1} of ${total}`;
    pagerPrev.disabled = pagerIndex === 0;
    pagerNext.disabled = pagerIndex === total - 1;
  }

  pagerPrev.addEventListener('click', () => {
    if (pagerIndex > 0) {
      pagerIndex--;
      renderPager();
    }
  });
  pagerNext.addEventListener('click', () => {
    if (pagerIndex < pagerKernels.length - 1) {
      pagerIndex++;
      renderPager();
    }
  });

  function renderFeatureTable(kernels) {
    const tbody = document.querySelector('#feature-table tbody');
    tbody.innerHTML = '';

    kernels.forEach((k) => {
      const tr = document.createElement('tr');
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
  const exportBtn = document.getElementById('btn-export-csv');
  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      if (!currentResults || !currentResults.kernels) return;

      let csv = 'ID,Status,Confidence(%),MoldArea(%),Contrast,Correlation,Energy,Homogeneity,Hue,Saturation,Value\n';
      currentResults.kernels.forEach((k) => {
        // BARU -- confidence null ditulis string kosong di CSV, bukan literal "null".
        const confVal = k.confidence !== null && k.confidence !== undefined ? k.confidence : '';
        csv += `${k.kernel_id},${formatStatus(k.prediction)},${confVal},${k.mold_ratio},${k.features.contrast},${k.features.correlation},${k.features.energy},${k.features.homogeneity},${k.features.hue},${k.features.saturation},${k.features.value}\n`;
      });

      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `corn_disease_features_${Date.now()}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  // ============================
  // BARU -- REALTIME WEBCAM DETECTION
  // ============================
  const webcamVideo = document.getElementById('webcam-video');
  const webcamCanvas = document.getElementById('webcam-canvas');
  const realtimeOutput = document.getElementById('realtime-output');
  const realtimeSummary = document.getElementById('realtime-summary');
  const realtimeStatus = document.getElementById('realtime-status');
  const btnStartWebcam = document.getElementById('btn-start-webcam');
  const btnStopWebcam = document.getElementById('btn-stop-webcam');

  let webcamStream = null; // menyimpan stream kamera aktif, supaya bisa dimatikan nanti
  let realtimeIntervalId = null; // id dari setInterval, supaya bisa dihentikan saat klik stop
  let isProcessingFrame = false; // "kunci" -- cegah kirim frame baru sebelum frame sebelumnya selesai diproses server
  let realtimeHasDetection = false; // flag: apakah ada biji terdeteksi selama sesi realtime ini
  let lastRealtimeFrame = null; // menyimpan frame terakhir (base64) untuk analisis penuh saat kamera dihentikan

  const REALTIME_INTERVAL_MS = 400; // jeda ambil frame: 400ms (~2.5 FPS) untuk deteksi realtime lebih cepat

  async function startWebcam() {
    // Cek dulu apakah browser menyediakan akses kamera
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert('The camera can only be used over HTTPS or localhost. Open the app via an https:// address or use one of the solutions in the guide.');
      return;
    }

    try {
      // Minta akses kamera belakang (environment). Browser akan menampilkan dialog izin.
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
      });
    } catch (err) {
      alert('Unable to access the camera: ' + err.message);
      return;
    }

    webcamVideo.srcObject = webcamStream; // sambungkan stream kamera ke elemen <video>

    btnStartWebcam.disabled = true;
    btnStopWebcam.disabled = false;
    realtimeStatus.textContent = 'Camera active, starting detection...';
    realtimeStatus.classList.remove('is-live', 'is-error');
    realtimeHasDetection = false;
    lastRealtimeFrame = null;

    // setInterval memanggil captureAndDetectFrame() berulang-ulang tiap
    // REALTIME_INTERVAL_MS milidetik, selama kamera masih menyala.
    realtimeIntervalId = setInterval(captureAndDetectFrame, REALTIME_INTERVAL_MS);
  }

  // ---- Helper: matikan kamera & hentikan interval (tanpa analisis) ----
  // Dipakai oleh beforeunload dan sebagai bagian dari stopWebcamAndAnalyze().
  function stopWebcamCleanup() {
    if (realtimeIntervalId) {
      clearInterval(realtimeIntervalId);
      realtimeIntervalId = null;
    }
    if (webcamStream) {
      webcamStream.getTracks().forEach((track) => track.stop());
      webcamStream = null;
    }
    webcamVideo.srcObject = null;
    isProcessingFrame = false;
  }

  // ---- Hentikan kamera + jalankan analisis penuh pada frame terakhir ----
  // Dipanggil saat user klik tombol "Hentikan Kamera". Kalau selama sesi
  // realtime ada biji yang terdeteksi, frame terakhir dari video akan
  // diproses oleh pipeline penuh (bukan lightweight) supaya hasilnya
  // bisa ditampilkan lengkap: tahapan preprocessing, kartu per-biji
  // dengan K-Means clustering, dan tabel fitur GLCM & HSV.
  function captureStandardizedSquare(videoEl, targetDim = 224, quality = 0.85) {
    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;
    const side = Math.min(vw, vh);
    const sx = Math.floor((vw - side) / 2);
    const sy = Math.floor((vh - side) / 2);
    webcamCanvas.width = targetDim;
    webcamCanvas.height = targetDim;
    const ctx = webcamCanvas.getContext('2d');
    ctx.drawImage(videoEl, sx, sy, side, side, 0, 0, targetDim, targetDim);
    return webcamCanvas.toDataURL('image/jpeg', quality);
  }

  async function stopWebcamAndAnalyze() {
    // 1. Tangkap frame terakhir SEBELUM kamera dimatikan (otomatis center-crop & resize 224x224)
    let finalFrame = null;
    if (webcamVideo.videoWidth) {
      finalFrame = captureStandardizedSquare(webcamVideo, 224, 0.92);
    }
    // Fallback ke frame terakhir yang tersimpan kalau capture gagal
    if (!finalFrame && lastRealtimeFrame) {
      finalFrame = lastRealtimeFrame;
    }

    // 2. Matikan kamera
    stopWebcamCleanup();

    btnStartWebcam.disabled = false;
    btnStopWebcam.disabled = true;
    realtimeStatus.textContent = finalFrame ? 'Camera stopped — processing the last frame...' : 'Camera stopped';
    realtimeStatus.classList.remove('is-live', 'is-error');

    // 3. Kalau ada frame dengan deteksi, jalankan pipeline PENUH
    //    (endpoint /api/detect/frame/full mengembalikan semua data lengkap:
    //    gambar tahapan, crop biji, K-Means, mold mask, fitur GLCM & HSV)
    if (finalFrame) {
      showLoading(true);
      try {
        const res = await fetch('/api/detect/frame/full', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: finalFrame }),
        });
        if (res.ok) {
          const data = await res.json();
          displayResults(data);
          realtimeStatus.textContent = 'Camera stopped — full analysis results are shown below';
        } else {
          realtimeStatus.textContent = 'Camera stopped (failed to analyze the last frame)';
          realtimeStatus.classList.add('is-error');
        }
      } catch (err) {
        console.error('Gagal analisis frame terakhir:', err);
        realtimeStatus.textContent = 'Camera stopped (server connection error)';
        realtimeStatus.classList.add('is-error');
      } finally {
        showLoading(false);
      }
    }

    // Reset untuk sesi kamera berikutnya
    realtimeHasDetection = false;
    lastRealtimeFrame = null;
  }

  async function captureAndDetectFrame() {
    // Kalau frame sebelumnya masih diproses server, jangan kirim frame baru dulu --
    // supaya request tidak menumpuk kalau server ternyata lebih lambat dari interval.
    if (isProcessingFrame) return;
    // Video belum siap (ukuran masih 0) -> tunggu interval berikutnya
    if (!webcamVideo.videoWidth) return;

    // Otomatis center-crop dan resize frame video ke resolusi standar 224x224
    // agar pengiriman data dan pemrosesan realtime jauh lebih cepat dan efisien.
    const frameDataUrl = captureStandardizedSquare(webcamVideo, 224, 0.82);
    lastRealtimeFrame = frameDataUrl; // simpan untuk analisis penuh saat kamera dihentikan

    isProcessingFrame = true;
    try {
      const res = await fetch('/api/detect/frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: frameDataUrl }),
      });

      if (!res.ok) {
        // Kalau 1 frame gagal diproses, jangan hentikan seluruh sesi realtime --
        // cukup catat di status, lanjut coba lagi di frame berikutnya.
        realtimeStatus.textContent = 'This frame failed to process, retrying...';
        realtimeStatus.classList.add('is-error');
        return;
      }

      const data = await res.json();
      renderRealtimeResult(data);
    } catch (err) {
      realtimeStatus.textContent = 'Server connection problem...';
      realtimeStatus.classList.add('is-error');
    } finally {
      isProcessingFrame = false;
    }
  }

  function renderRealtimeResult(data) {
    // Kalau kamera sudah dihentikan saat request masih jalan, abaikan hasilnya
    if (!webcamStream) return;
    realtimeOutput.src = data.annotated;

    const s = data.summary;
    realtimeStatus.textContent = `Live -- kernels detected: ${s.total_biji}`;
    realtimeStatus.classList.remove('is-error');
    realtimeStatus.classList.add('is-live');

    // Tandai bahwa ada OBJEK APAPUN terdeteksi di sesi ini -- supaya saat kamera
    // dihentikan, kita tahu perlu jalankan analisis penuh pada frame terakhir.
    // Pakai data.kernels.length (bukan s.total_biji) karena total_biji mungkin
    // tidak termasuk objek "bukan jagung", padahal kita tetap ingin menampilkan
    // analisis lengkap meskipun semua objek terdeteksi sebagai bukan jagung.
    if (data.kernels && data.kernels.length > 0) {
      realtimeHasDetection = true;
    }
    const totalJagung = (s.sehat || 0) + (s.terkontaminasi || 0);
    const bukanJagung = s.bukan_jagung || 0;
    let text = '';
    if (totalJagung === 0) {
      text = bukanJagung > 0 ? `Not Corn (${bukanJagung} non-corn object(s) detected).` : 'No corn kernels detected in this frame yet.';
    } else if (bukanJagung > totalJagung) {
      text = `Not Corn -- non-corn objects predominate (${bukanJagung} non-corn vs ${totalJagung} corn-like).`;
    } else {
      text = `${s.sehat} healthy, ${s.terkontaminasi} mold-contaminated (${s.persen_kontaminasi}%)`;
      if (bukanJagung > 0) {
        text += ` -- ignored non-corn objects: ${bukanJagung}.`;
      }
    }
    realtimeSummary.textContent = text;
  }

  btnStartWebcam.addEventListener('click', startWebcam);
  btnStopWebcam.addEventListener('click', stopWebcamAndAnalyze);

  // Kalau user pindah/tutup tab tanpa klik "Hentikan Kamera" dulu, pastikan
  // kamera tetap dimatikan supaya lampu indikator kamera tidak terus menyala.
  // Pakai stopWebcamCleanup (bukan stopWebcamAndAnalyze) karena tab sudah
  // ditutup -- tidak perlu kirim request analisis yang hasilnya tidak akan dilihat.
  window.addEventListener('beforeunload', stopWebcamCleanup);

  loadSamples();
});